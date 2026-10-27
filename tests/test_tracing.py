"""
C-5 / C-9 tracing and monitoring: one W3C trace id per investigation across ADK spans,
the audit ledger and the MCP server process; per-agent metrics; structured errors.
(A2A propagation to the CrewAI service is tested in tests/test_crewai_a2a.py.)
"""
import importlib
import json
import os
import tempfile
import unittest
from unittest import mock

from src.adk_app.pipeline import run_adk_investigation
from src.core.audit import AuditLogger
from src.core.tracing import read_trace, trace_id_from_traceparent
from tests.test_adk_pipeline import load_claim


class TestTraceIds(unittest.TestCase):

    def test_one_trace_id_links_spans_and_audit_entries(self):
        report = run_adk_investigation(load_claim("scenario_3_conflict")).report
        trace_id = report.trace_id
        self.assertRegex(trace_id, r"^[0-9a-f]{32}$")
        spans = read_trace(trace_id)
        self.assertTrue(spans)
        self.assertEqual({s["trace_id"] for s in spans}, {trace_id})
        root = [s for s in spans if s["name"] == "claimsleuth.investigation"]
        self.assertEqual(len(root), 1)
        self.assertIsNone(root[0]["parent_span_id"])
        self.assertEqual(root[0]["attributes"]["claim_id"], "CLM-2024-003")
        ids = {s["span_id"] for s in spans}
        self.assertTrue(all(s["parent_span_id"] in ids for s in spans if s is not root[0]))   # one connected tree
        self.assertTrue(any(s["name"].startswith("invoke_node") for s in spans))              # ADK's own spans
        stamped = [e for e in AuditLogger("CLM-2024-003").get_trail() if e.get("trace_id") == trace_id]
        self.assertGreaterEqual(len(stamped), 7)
        self.assertTrue(AuditLogger("CLM-2024-003").verify_chain()["valid"])

    def test_each_run_gets_its_own_trace(self):
        a = run_adk_investigation(load_claim("scenario_1_valid")).report.trace_id
        b = run_adk_investigation(load_claim("scenario_1_valid")).report.trace_id
        self.assertNotEqual(a, b)

    def test_trace_id_reaches_the_mcp_server_process(self):
        fd, log = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with mock.patch.dict(os.environ, {"CLAIMSLEUTH_MCP_CALL_LOG": log}):
                report = run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp").report
            calls = [json.loads(line) for line in open(log, encoding="utf-8") if line.strip()]
        finally:
            os.remove(log)
        self.assertEqual(len(calls), len(report.tool_calls))
        self.assertEqual({c["trace_id"] for c in calls}, {report.trace_id})
        self.assertNotIn(os.getpid(), {c["pid"] for c in calls})
        self.assertTrue(any(s["name"].startswith("MCP") or "mcp" in s["name"].lower()
                            for s in read_trace(report.trace_id)))

    def test_traceparent_parsing(self):
        self.assertEqual(trace_id_from_traceparent("00-" + "a" * 32 + "-" + "b" * 16 + "-01"), "a" * 32)
        self.assertIsNone(trace_id_from_traceparent("garbage"))
        self.assertIsNone(trace_id_from_traceparent(None))

    def test_trace_endpoint(self):
        from fastapi.testclient import TestClient
        os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        import configs.settings
        importlib.reload(configs.settings)
        import src.ui.app as app_module
        client = TestClient(importlib.reload(app_module).app)
        data = client.post("/api/adk/investigate/scenario_2_expired").json()
        res = client.get(f"/api/trace/{data['report']['trace_id']}").json()
        self.assertGreater(res["span_count"], 5)
        self.assertEqual(client.get("/api/trace/not-a-trace").status_code, 422)
        self.assertEqual(client.get("/api/trace/" + "0" * 32).status_code, 404)


class TestPerAgentMetricsAndErrors(unittest.TestCase):

    def test_metrics_are_measured_not_invented(self):
        report = run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp").report
        m = report.agent_metrics
        self.assertEqual(set(m) - {"ReportAgent"}, {"IntakeAgent", "EvidenceAgent", "AnomalyAgent", "WarrantyAgent",
                                                    "SettlementAgent", "VerifierAgent"})
        self.assertTrue(all(m[a]["latency_ms"] is not None for a in m if a != "ReportAgent"))
        self.assertEqual(sum(v["tool_calls"] for v in m.values()), len(report.tool_calls))
        # No live model ran: tokens are None everywhere, never zero or estimated
        self.assertTrue(all(v["input_tokens"] is None and v["llm_calls"] is None for v in m.values()))

    def test_errors_are_attributed_to_agents(self):
        report = run_adk_investigation(load_claim("scenario_1_valid"), forensics="crewai_a2a",
                                       a2a_url="http://127.0.0.1:9", a2a_timeout=3).report
        kinds = {(e["agent"], e["kind"]) for e in report.errors}
        self.assertIn(("CrewAIForensics", "a2a_error"), kinds)
        self.assertIn(("ForensicsReviewAgent", "tool_error"), kinds)
        self.assertGreaterEqual(report.agent_metrics["ForensicsReviewAgent"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
