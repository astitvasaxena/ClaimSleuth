"""
Stage A: the ADK pipeline reaches its tools through ADK's McpToolset and the real
ClaimSleuth MCP server (separate process), and stays safe when that server fails.

Evidence that calls really reached the server: the server appends every executed
call, with ITS process id, to CLAIMSLEUTH_MCP_CALL_LOG.
"""
import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from google.genai import types

from src.adk_app.pipeline import run_adk_investigation
from src.adk_app.tool_gateway import McpConfig, MCPUnavailableError
from src.core.audit import AuditLogger
from tests.test_adk_pipeline import ScriptedFakeLlm, call, load_claim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAULTY_SERVER = str(PROJECT_ROOT / "tests" / "fixtures" / "faulty_mcp_server.py")
SCENARIOS = {"scenario_1_valid": ("VERIFIED_PENDING_APPROVAL", "PROPOSED_PENDING_APPROVAL", 251.0),
             "scenario_2_expired": ("REJECTED_DETERMINISTICALLY", "NOT_PAYABLE", None),
             "scenario_3_conflict": ("CONFLICT_ESCALATED", "NOT_PAYABLE", None)}


class CallLogMixin:
    """Points the server's call log at a fresh temp file for each test."""

    def setUp(self):
        fd, self.call_log = tempfile.mkstemp(prefix="mcp_calls_", suffix=".jsonl")
        os.close(fd)
        self._env = mock.patch.dict(os.environ, {"CLAIMSLEUTH_MCP_CALL_LOG": self.call_log})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        os.remove(self.call_log)

    def server_calls(self):
        with open(self.call_log, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class TestAdkUsesRemoteMcpServer(CallLogMixin, unittest.TestCase):

    def test_three_scenarios_over_mcp_stdio_match_local(self):
        for scenario, (status, payout_status, proposed) in SCENARIOS.items():
            for backend in ("workflow", "sequential"):
                with self.subTest(scenario=scenario, backend=backend):
                    remote = run_adk_investigation(load_claim(scenario), backend=backend, tools="mcp").report
                    local = run_adk_investigation(load_claim(scenario), backend=backend, tools="local").report
                    self.assertEqual(remote.tool_backend, "mcp_stdio")
                    self.assertEqual((remote.status, remote.payout.payout_status, remote.payout.proposed_payout),
                                     (status, payout_status, proposed))
                    self.assertEqual(remote.status, local.status)
                    self.assertEqual(remote.payout.model_dump(), local.payout.model_dump())
                    self.assertEqual(remote.tool_errors, [])
                    self.assertIsNone(remote.payout.authorized_payout)

    def test_server_process_executed_the_calls(self):
        report = run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp").report
        server_calls = self.server_calls()
        client_calls = [c for c in report.tool_calls if c["via"] == "mcp_stdio"]
        self.assertEqual(len(server_calls), len(client_calls))
        self.assertEqual([c["tool"] for c in server_calls], [c["tool"] for c in client_calls])
        self.assertEqual({c["pid"] for c in server_calls} & {os.getpid()}, set())   # not this process
        self.assertEqual(len({c["pid"] for c in server_calls}), 1)                 # one server for the run
        self.assertIn("calculate_policy_settlement", [c["tool"] for c in server_calls])
        self.assertEqual({c["agent"] for c in client_calls},
                         {"IntakeAgent", "EvidenceAgent", "AnomalyAgent", "WarrantyAgent", "SettlementAgent"})

    def test_mcp_settlement_is_only_a_proposal(self):
        report = run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp").report
        self.assertEqual(report.payout.payout_status, "PROPOSED_PENDING_APPROVAL")
        self.assertIsNone(report.payout.authorized_payout)
        self.assertTrue(report.requires_human_approval)
        self.assertTrue(report.verifier.gate_2_arithmetic_matches)   # verifier recomputed locally and agreed

    def test_local_backend_makes_no_server_calls(self):
        run_adk_investigation(load_claim("scenario_1_valid"), tools="local")
        self.assertEqual(self.server_calls(), [])


class TestMcpServerFailures(CallLogMixin, unittest.TestCase):

    def faulty(self, fault):
        return McpConfig(transport="stdio", server_args=[FAULTY_SERVER, "--fault", fault])

    def test_tampered_settlement_is_caught_by_local_verifier(self):
        report = run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp",
                                       mcp=self.faulty("tamper_settlement")).report
        self.assertEqual(report.settlement_proposal.approved_payout, 751.0)   # what the server claimed
        self.assertFalse(report.verifier.gate_2_arithmetic_matches)
        self.assertEqual(report.status, "CONFLICT_ESCALATED")
        self.assertEqual(report.payout.payout_status, "NOT_PAYABLE")
        self.assertIsNone(report.payout.proposed_payout)
        self.assertIn("TOOL_RESULT_MISMATCH", report.risk_flags)

    def test_settlement_tool_error_fails_closed(self):
        report = run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp",
                                       mcp=self.faulty("settlement_error")).report
        self.assertIsNone(report.settlement_proposal)
        self.assertEqual(report.status, "NEEDS_EVIDENCE")
        self.assertEqual(report.payout.payout_status, "NOT_PAYABLE")
        self.assertIn("simulated settlement service failure", report.tool_errors[0])
        self.assertIn("TOOL_ERROR", report.risk_flags)

    def test_missing_tool_aborts_run(self):
        with self.assertRaises(MCPUnavailableError):
            run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp", mcp=self.faulty("missing_tool"))

    def test_server_that_cannot_start_aborts_without_result(self):
        reports_before = set(Path(os.environ["CLAIMSLEUTH_REPORTS_DIR"]).glob("CLM-2024-001_*.json"))
        bad = McpConfig(transport="stdio", server_args=["-m", "src.mcp.no_such_server"], timeout=10)
        started = time.time()
        with self.assertRaises(MCPUnavailableError):
            run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp", mcp=bad)
        self.assertLess(time.time() - started, 60)
        self.assertEqual(set(Path(os.environ["CLAIMSLEUTH_REPORTS_DIR"]).glob("CLM-2024-001_*.json")), reports_before)
        trail = AuditLogger("CLM-2024-001").get_trail()
        self.assertIn("MCP server unavailable", trail[-1]["details"]["conflict"])
        self.assertTrue(AuditLogger("CLM-2024-001").verify_chain()["valid"])

    def test_unreachable_http_server_aborts(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with self.assertRaises(MCPUnavailableError):
            run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp",
                                  mcp=McpConfig(transport="http", url=f"http://127.0.0.1:{port}/mcp", timeout=5))


class TestAdkOverMcpHttp(CallLogMixin, unittest.TestCase):
    """The same pipeline against an already-running streamable-HTTP server process."""

    def test_pipeline_over_http(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = subprocess.Popen([sys.executable, "-m", "src.mcp.remote_server", "--transport", "http",
                                   "--port", str(port)], cwd=str(PROJECT_ROOT), env=dict(os.environ),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                    break
                except OSError:
                    time.sleep(0.3)
            config = McpConfig(transport="http", url=f"http://127.0.0.1:{port}/mcp")
            for scenario, (status, payout_status, proposed) in SCENARIOS.items():
                report = run_adk_investigation(load_claim(scenario), tools="mcp", mcp=config).report
                self.assertEqual(report.tool_backend, "mcp_http")
                self.assertEqual((report.status, report.payout.payout_status, report.payout.proposed_payout),
                                 (status, payout_status, proposed))
            self.assertEqual({c["pid"] for c in self.server_calls()}, {server.pid})
        finally:
            server.terminate()
            server.wait(timeout=10)


class TestModelUsesMcpTools(CallLogMixin, unittest.TestCase):
    """TEST DOUBLE model + real MCP server: the model's MCP tool calls feed the Blackboard."""

    def run_model(self, script):
        model = ScriptedFakeLlm(model="fake-mcp-investigator", script=script)
        return run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp", model=model).report

    def script(self, *extra):
        return [call("list_claim_documents", claim_id="CLM-2024-001"),
                call("extract_claim_document_facts", claim_id="CLM-2024-001", document_name="invoice_001.txt",
                     fields=["purchase_date", "serial_number"]),
                call("extract_claim_document_facts", claim_id="CLM-2024-001",
                     document_name="repair_estimate_001.txt", fields=["repair_cost", "incident_cause", "serial_number"]),
                *extra, types.Part(text="Facts gathered.")]

    def test_model_mcp_calls_reach_server_and_blackboard(self):
        report = self.run_model(self.script())
        self.assertEqual(report.status, "VERIFIED_PENDING_APPROVAL")
        self.assertEqual(report.payout.proposed_payout, 251.0)
        self.assertEqual([c["tool"] for c in report.model_tool_calls],
                         ["list_claim_documents", "extract_claim_document_facts", "extract_claim_document_facts"])
        self.assertTrue(all(c["via"] == "mcp" and c["ok"] for c in report.model_tool_calls))
        server_tools = [c["tool"] for c in self.server_calls()]
        self.assertEqual(server_tools.count("extract_claim_document_facts"), 2 + 2)  # model's 2 + AnomalyAgent's 2

    def test_model_cannot_query_another_claim(self):
        report = self.run_model(self.script(call("extract_claim_document_facts", claim_id="CLM-2024-003",
                                                 document_name="invoice_003.txt", fields=["serial_number"])))
        blocked = [c for c in report.model_tool_calls if c["via"] == "blocked"]
        self.assertEqual(len(blocked), 1)
        self.assertNotIn("CLM-2024-003", json.dumps([c["args"] for c in self.server_calls()]))
        self.assertEqual(report.status, "VERIFIED_PENDING_APPROVAL")

    def test_model_has_no_settlement_tool_and_wrong_claim_escalates(self):
        report = self.run_model(self.script(call("calculate_policy_settlement", policy_id="POL-TECH-CARE-01",
                                                 repair_cost=5000),
                                            call("record_model_payout_claim", amount=4971.0)))
        self.assertNotIn("calculate_policy_settlement",
                         [c["tool"] for c in report.model_tool_calls if c.get("ok")])
        self.assertEqual(report.status, "CONFLICT_ESCALATED")
        self.assertEqual(report.payout.payout_status, "NOT_PAYABLE")


class TestAdkEndpointMcp(CallLogMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        import configs.settings
        importlib.reload(configs.settings)
        import src.ui.app as app_module
        cls.client = TestClient(importlib.reload(app_module).app)

    def test_endpoint_mcp_stdio(self):
        data = self.client.post("/api/adk/investigate/scenario_3_conflict?tools=mcp_stdio").json()
        self.assertEqual(data["report"]["tool_backend"], "mcp_stdio")
        self.assertEqual(data["report"]["status"], "CONFLICT_ESCALATED")
        self.assertTrue(self.server_calls())

    def test_endpoint_returns_503_when_mcp_http_unreachable(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with mock.patch.dict(os.environ, {"CLAIMSLEUTH_MCP_URL": f"http://127.0.0.1:{port}/mcp"}):
            res = self.client.post("/api/adk/investigate/scenario_1_valid?tools=mcp_http")
        self.assertEqual(res.status_code, 503)
        self.assertIn("unavailable", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
