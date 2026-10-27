"""
WP-3: MCP over real transports (stdio and streamable HTTP).

Every test talks to the server through an MCP ClientSession across a process
boundary: stdio spawns `python -m src.mcp.remote_server --transport stdio`, and the
HTTP tests start the server as a separate process on a free localhost port.
"""
import asyncio
import json
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from src.mcp.remote_client import MCPToolError, RemoteMCPClient, call_remote_tool
from src.mcp.remote_server import REMOTE_TOOL_NAMES
from src.tools.math_engine import compute_settlement
from src.tools.policy_rules import load_policy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICIES = json.loads((PROJECT_ROOT / "configs" / "policies.json").read_text(encoding="utf-8"))


async def exercise(client: RemoteMCPClient) -> dict:
    """Run every tool once (plus error cases) inside one session; return what came back."""
    out = {"server_name": client.server_name, "tools": await client.list_tools()}
    out["docs"] = await client.call_tool("list_claim_documents", {"claim_id": "CLM-2024-003"})
    out["facts"] = await client.call_tool("extract_claim_document_facts", {
        "claim_id": "CLM-2024-003", "document_name": "repair_estimate_003.txt",
        "fields": ["serial_number", "repair_cost"]})
    out["terms"] = await client.call_tool("get_policy_terms", {"policy_id": "POL-TECH-CARE-01"})
    out["coverage_expired"] = await client.call_tool("check_policy_coverage", {
        "policy_id": "POL-LAPTOP-STD-02", "purchase_date": "2023-01-10", "incident_date": "2024-04-12",
        "reported_cause": "Laptop fails to power on"})
    out["coverage_excluded"] = await client.call_tool("check_policy_coverage", {
        "policy_id": "POL-LAPTOP-STD-02", "purchase_date": "2024-01-10", "incident_date": "2024-02-01",
        "reported_cause": "keyboard stopped after water damage"})
    out["settlements"] = {pid: await client.call_tool("calculate_policy_settlement",
                                                      {"policy_id": pid, "repair_cost": 650.0})
                          for pid in POLICIES}
    out["serials"] = await client.call_tool("flag_serial_anomalies", {
        "claim_serial": "REF-HC-77100-A", "invoice_serial": "REF-HC-77100-A", "diagnostic_serial": "REF-HC-99400-B"})

    errors = {}
    for label, name, args in [
        ("unknown_policy", "get_policy_terms", {"policy_id": "NOPE"}),
        ("path_traversal", "extract_claim_document_facts",
         {"claim_id": "CLM-2024-001", "document_name": "../../../configs/settings.py", "fields": ["repair_cost"]}),
        ("bad_field", "extract_claim_document_facts",
         {"claim_id": "CLM-2024-001", "document_name": "invoice_001.txt", "fields": ["password"]}),
        ("bad_type", "calculate_policy_settlement", {"policy_id": "POL-TECH-CARE-01", "repair_cost": "lots"}),
        ("missing_arg", "calculate_policy_settlement", {"policy_id": "POL-TECH-CARE-01"}),
        ("negative_cost", "calculate_policy_settlement", {"policy_id": "POL-TECH-CARE-01", "repair_cost": -5}),
        ("bad_date", "check_policy_coverage",
         {"policy_id": "POL-TECH-CARE-01", "purchase_date": "15/03/2024", "incident_date": "2024-09-20"}),
        ("unknown_tool", "approve_payout", {"claim_id": "CLM-2024-001"}),
    ]:
        try:
            await client.call_tool(name, args)
            errors[label] = None
        except MCPToolError as exc:
            errors[label] = str(exc)
        except Exception as exc:  # protocol-level error (e.g. unknown tool)
            errors[label] = f"{type(exc).__name__}: {exc}"
    out["errors"] = errors
    return out


class RemoteToolAssertions:
    """Shared assertions for both transports."""
    results: dict

    def test_server_lists_all_tools_with_schemas(self):
        self.assertEqual(self.results["server_name"], "ClaimSleuthTools")
        tools = {t["name"]: t for t in self.results["tools"]}
        self.assertEqual(sorted(tools), sorted(REMOTE_TOOL_NAMES))
        self.assertIn("policy_id", tools["calculate_policy_settlement"]["input_schema"]["properties"])
        # Policy terms are not accepted from the caller
        self.assertNotIn("deductible", tools["calculate_policy_settlement"]["input_schema"]["properties"])

    def test_document_tools_return_provenance(self):
        self.assertEqual(self.results["docs"]["documents"], ["invoice_003.txt", "repair_estimate_003.txt"])
        serial = self.results["facts"]["facts"]["serial_number"]
        self.assertEqual(serial["extracted_value"], "REF-HC-99400-B")
        self.assertEqual(serial["status"], "VERIFIED")
        self.assertIn("REF-HC-99400-B", serial["raw_snippet"])
        self.assertEqual(self.results["facts"]["facts"]["repair_cost"]["extracted_value"], 650.0)

    def test_policy_lookup_and_coverage(self):
        self.assertEqual(self.results["terms"]["deductible"], 29.0)
        expired = self.results["coverage_expired"]
        self.assertFalse(expired["coverage"]["is_covered"])
        self.assertFalse(expired["eligible"])
        excluded = self.results["coverage_excluded"]
        self.assertTrue(excluded["coverage"]["is_covered"])
        self.assertTrue(excluded["excluded"])
        self.assertEqual(excluded["matched_exclusion"], "water damage")

    def test_settlement_matches_in_process_deterministic_code(self):
        for pid, remote in self.results["settlements"].items():
            p = load_policy(pid)
            local = compute_settlement(650.0, p.deductible, p.max_coverage_limit, p.depreciation_rate)
            self.assertEqual(remote["settlement"], local.model_dump(), pid)
            self.assertTrue(remote["authorization"].startswith("NOT_AUTHORIZED"))
        self.assertEqual(self.results["settlements"]["POL-APPLIANCE-PLUS-03"]["settlement"]["approved_payout"], 546.25)

    def test_serial_anomaly(self):
        self.assertTrue(self.results["serials"]["has_anomalies"])

    def test_invalid_requests_return_errors_not_results(self):
        errors = self.results["errors"]
        for label, message in errors.items():
            self.assertIsNotNone(message, f"{label} unexpectedly succeeded")
        self.assertIn("Unknown policy_id 'NOPE'", errors["unknown_policy"])
        self.assertIn("not in the dossier", errors["path_traversal"])
        self.assertIn("password", errors["bad_field"])
        self.assertIn("non-negative", errors["negative_cost"])


class TestMCPOverStdio(RemoteToolAssertions, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        async def run():
            async with RemoteMCPClient.stdio() as client:
                return await exercise(client)
        cls.results = asyncio.run(run())

    def test_one_shot_sync_helper(self):
        res = call_remote_tool("calculate_policy_settlement", {"policy_id": "POL-TECH-CARE-01", "repair_cost": 280})
        self.assertEqual(res["settlement"]["approved_payout"], 251.0)

    def test_call_really_crosses_the_process_boundary(self):
        # Sabotage the calculator inside THIS process. An in-process call would fail;
        # the stdio server is a separate interpreter and is unaffected.
        with mock.patch("src.mcp.remote_server.compute_settlement", side_effect=AssertionError("in-process call")), \
             mock.patch("src.tools.math_engine.compute_settlement", side_effect=AssertionError("in-process call")):
            res = call_remote_tool("calculate_policy_settlement",
                                   {"policy_id": "POL-TECH-CARE-01", "repair_cost": 280})
        self.assertEqual(res["settlement"]["approved_payout"], 251.0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestMCPOverStreamableHttp(RemoteToolAssertions, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.url = f"http://127.0.0.1:{cls.port}/mcp"
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "src.mcp.remote_server", "--transport", "http", "--port", str(cls.port)],
            cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 30
        while time.time() < deadline:
            if cls.server.poll() is not None:
                raise RuntimeError(f"MCP HTTP server exited early with code {cls.server.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", cls.port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.3)
        else:
            cls.server.kill()
            raise RuntimeError("MCP HTTP server did not start within 30s")

        async def run():
            async with RemoteMCPClient.http(cls.url) as client:
                return await exercise(client)
        cls.results = asyncio.run(run())

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.server.kill()

    def test_server_is_a_separate_process(self):
        self.assertIsNone(self.server.poll(), "server should still be running")
        self.assertNotEqual(self.server.pid, __import__("os").getpid())

    def test_two_independent_sessions(self):
        async def run():
            async with RemoteMCPClient.http(self.url) as a, RemoteMCPClient.http(self.url) as b:
                return (await a.call_tool("get_policy_terms", {"policy_id": "POL-LAPTOP-STD-02"}),
                        await b.call_tool("get_policy_terms", {"policy_id": "POL-APPLIANCE-PLUS-03"}))
        laptop, appliance = asyncio.run(run())
        self.assertEqual(laptop["duration_months"], 12)
        self.assertEqual(appliance["duration_months"], 36)


class TestConnectionFailure(unittest.TestCase):

    def test_unreachable_http_server_raises(self):
        url = f"http://127.0.0.1:{_free_port()}/mcp"   # nothing listening
        started = time.time()
        with self.assertRaises(BaseException) as ctx:
            call_remote_tool("get_policy_terms", {"policy_id": "POL-TECH-CARE-01"}, transport="http",
                             url=url, timeout=5)
        self.assertNotIsInstance(ctx.exception, MCPToolError)
        self.assertLess(time.time() - started, 30)


if __name__ == "__main__":
    unittest.main()
