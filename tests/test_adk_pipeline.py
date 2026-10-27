"""
WP-2 acceptance tests: the Google ADK execution path.

Everything here runs offline with no API key. Model-mode tests use ScriptedFakeLlm,
a TEST DOUBLE subclassing google.adk.models.BaseLlm: it proves the ADK wiring
(LlmAgent -> tool calls -> session state -> verifier), NOT a live Gemini integration.
Live Gemini is tested separately in tests/live/.
"""
import importlib
import json
import os
import socket
import unittest
from pathlib import Path
from unittest import mock

from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src.adk_app.pipeline import run_adk_investigation
from src.adk_app.report import FinalReport
from src.agents.investigator import LeadInvestigatorAgent
from src.core.adjudication import payout_view
from src.core.audit import AuditLogger
from src.core.state import ClaimInvestigationState

CLAIMS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_claims"
BACKENDS = ("workflow", "sequential")
EXPECTED_SEQUENCE = ["IntakeAgent", "EvidenceAgent", "PolicyRetrievalAgent", "AnomalyAgent", "WarrantyAgent",
                     "SettlementAgent", "VerifierAgent", "ReportAgent"]
REQUIRED_KEYS = ["bb:run", "bb:claim", "bb:policy", "bb:documents", "bb:evidence", "bb:conflicts",
                 "bb:missing", "bb:anomalies", "bb:eligibility", "bb:settlement_proposal",
                 "bb:model_payout_claim", "bb:model_tool_calls", "bb:verifier", "bb:status",
                 "bb:requires_human_approval", "bb:payout", "bb:final_report", "bb:report_files",
                 "bb:agent_sequence"]


def load_claim(scenario_id: str) -> dict:
    with open(CLAIMS_DIR / scenario_id / "claim.json", encoding="utf-8") as f:
        return json.load(f)


def call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


class ScriptedFakeLlm(BaseLlm):
    """TEST DOUBLE: returns a fixed script of responses, one per model call."""
    script: list = []
    calls_made: int = 0

    @classmethod
    def supported_models(cls):
        return [r"fake-.*"]

    async def generate_content_async(self, llm_request, stream=False):
        part = self.script[min(self.calls_made, len(self.script) - 1)]
        self.calls_made += 1
        if isinstance(part, Exception):
            raise part
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def investigator_script(invoice, diagnostic, payout_claim=None):
    """A sensible investigation: list docs, extract from both documents, optionally state a payout."""
    script = [call("list_dossier_documents"),
              call("extract_document_facts", document_name=invoice, fields=["purchase_date", "serial_number"]),
              call("extract_document_facts", document_name=diagnostic,
                   fields=["repair_cost", "incident_cause", "serial_number"])]
    if payout_claim is not None:
        script.append(call("record_model_payout_claim", amount=payout_claim))
    script.append(types.Part(text="Facts gathered."))
    return script


class TestScenarioParity(unittest.TestCase):
    """The ADK path reaches the same outcome as the original /api/investigate loop."""

    def check(self, scenario_id, status, payout_status, proposed):
        claim = load_claim(scenario_id)
        legacy = LeadInvestigatorAgent(use_llm=False).run_investigation(ClaimInvestigationState(**claim))
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                report = run_adk_investigation(claim, backend=backend).report
                self.assertEqual(report.status, status)
                self.assertEqual(report.status, legacy.status)
                self.assertEqual(report.payout.payout_status, payout_status)
                self.assertEqual(report.payout.payout_status, payout_view(legacy)["payout_status"])
                self.assertEqual(report.payout.proposed_payout, proposed)
                self.assertIsNone(report.payout.authorized_payout)

    def test_scenario_1_valid_proposes_251_pending_approval(self):
        self.check("scenario_1_valid", "VERIFIED_PENDING_APPROVAL", "PROPOSED_PENDING_APPROVAL", 251.0)

    def test_scenario_2_expired_rejected_not_payable(self):
        self.check("scenario_2_expired", "REJECTED_DETERMINISTICALLY", "NOT_PAYABLE", None)

    def test_scenario_3_conflict_escalated_not_payable(self):
        self.check("scenario_3_conflict", "CONFLICT_ESCALATED", "NOT_PAYABLE", None)
        report = run_adk_investigation(load_claim("scenario_3_conflict")).report
        self.assertIsNone(report.settlement_proposal)
        self.assertEqual(report.payout.diagnostic_calculation["approved_payout"], 546.25)
        self.assertEqual(report.recommended_action, "ESCALATE_TO_HUMAN")


class TestSessionStateAndOrdering(unittest.TestCase):

    def test_blackboard_keys_present_on_both_backends(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                state = run_adk_investigation(load_claim("scenario_1_valid"), backend=backend).state
                missing = [k for k in REQUIRED_KEYS if k not in state]
                self.assertEqual(missing, [])
                self.assertEqual(state["bb:settlement_proposal"]["approved_payout"], 251.0)
                self.assertEqual(state["bb:evidence"]["repair_cost"]["source_document"], "repair_estimate_001.txt")

    def test_workflow_node_order_matches_agent_sequence(self):
        result = run_adk_investigation(load_claim("scenario_1_valid"), backend="workflow")
        self.assertEqual(result.report.agent_sequence, EXPECTED_SEQUENCE)
        nodes = [e["node_path"].split("/")[-1].split("@")[0] for e in result.events if e["node_path"]]
        self.assertEqual(nodes, EXPECTED_SEQUENCE)

    def test_sequential_event_authors_match_agent_sequence(self):
        result = run_adk_investigation(load_claim("scenario_1_valid"), backend="sequential")
        self.assertEqual([e["author"] for e in result.events], EXPECTED_SEQUENCE)
        self.assertEqual(result.report.pipeline_backend, "sequential_agent")


class TestReportOutput(unittest.TestCase):

    def test_report_files_written_and_valid(self):
        result = run_adk_investigation(load_claim("scenario_1_valid"))
        json_path, md_path = Path(result.files["json"]), Path(result.files["markdown"])
        self.assertTrue(json_path.exists() and md_path.exists())
        self.assertEqual(json_path.parent.resolve(), Path(os.environ["CLAIMSLEUTH_REPORTS_DIR"]).resolve())
        reloaded = FinalReport.model_validate_json(json_path.read_text(encoding="utf-8"))
        self.assertEqual(reloaded.run_id, result.report.run_id)
        self.assertEqual(reloaded.planner, "deterministic")
        markdown = md_path.read_text(encoding="utf-8")
        self.assertIn("CLM-2024-001", markdown)
        self.assertIn("PROPOSED_PENDING_APPROVAL", markdown)

    def test_audit_chain_valid_across_adk_and_legacy_runs(self):
        claim = load_claim("scenario_3_conflict")
        run_adk_investigation(claim, backend="workflow")
        LeadInvestigatorAgent(use_llm=False).run_investigation(ClaimInvestigationState(**claim))
        report = run_adk_investigation(claim, backend="sequential").report
        self.assertTrue(report.audit_chain_valid)
        self.assertTrue(AuditLogger(claim["claim_id"]).verify_chain()["valid"])


class TestModelModeWithFakeLlm(unittest.TestCase):
    """TEST DOUBLE model: verifies wiring and safety rules, not live Gemini."""

    def run_model(self, script, scenario="scenario_1_valid", backend="workflow", **kw):
        model = ScriptedFakeLlm(model="fake-investigator", script=script)
        return run_adk_investigation(load_claim(scenario), backend=backend, model=model, **kw), model

    def test_fake_model_tool_calls_execute_real_tools(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                result, model = self.run_model(
                    investigator_script("invoice_001.txt", "repair_estimate_001.txt"), backend=backend)
                report = result.report
                self.assertEqual(report.planner, "llm:fake-investigator")
                self.assertEqual([c["tool"] for c in report.model_tool_calls],
                                 ["list_dossier_documents", "extract_document_facts", "extract_document_facts"])
                self.assertEqual(report.agent_sequence[:2], ["IntakeAgent", "InvestigatorLLM"])
                self.assertNotIn("EvidenceAgent", report.agent_sequence)
                self.assertEqual(report.status, "VERIFIED_PENDING_APPROVAL")
                self.assertEqual(report.payout.proposed_payout, 251.0)
                self.assertEqual(result.state["bb:model_summary"], "Facts gathered.")

    def test_model_stated_wrong_payout_is_rejected_and_escalated(self):
        result, _ = self.run_model(investigator_script("invoice_001.txt", "repair_estimate_001.txt", 999.0))
        report = result.report
        self.assertEqual(report.model_payout_claim, 999.0)
        self.assertEqual(report.status, "CONFLICT_ESCALATED")
        self.assertEqual(report.payout.payout_status, "NOT_PAYABLE")
        self.assertIsNone(report.payout.proposed_payout)
        self.assertIn("MODEL_PAYOUT_MISMATCH", report.risk_flags)
        self.assertFalse(report.verifier.is_approved_by_verifier)
        # The deterministic proposal still exists, but cannot be paid while the conflict is open
        self.assertEqual(report.settlement_proposal.approved_payout, 251.0)

    def test_model_stated_correct_payout_does_not_change_outcome(self):
        report = self.run_model(investigator_script("invoice_001.txt", "repair_estimate_001.txt", 251.0))[0].report
        self.assertEqual(report.payout.payout_status, "PROPOSED_PENDING_APPROVAL")
        self.assertEqual(report.payout.proposed_payout, 251.0)

    def test_model_skipping_evidence_fails_closed(self):
        report = self.run_model([call("list_dossier_documents"), types.Part(text="Done.")])[0].report
        self.assertEqual(report.status, "NEEDS_EVIDENCE")
        self.assertEqual(report.payout.payout_status, "NOT_PAYABLE")
        self.assertIn("repair_cost", report.missing_evidence)

    def test_model_cannot_read_files_outside_dossier(self):
        script = [call("extract_document_facts", document_name="../../configs/settings.py", fields=["repair_cost"]),
                  types.Part(text="Done.")]
        report = self.run_model(script)[0].report
        self.assertEqual(report.model_tool_calls[0]["ok"], False)
        self.assertEqual(report.evidence, [])
        self.assertEqual(report.status, "NEEDS_EVIDENCE")

    def test_model_error_falls_back_to_deterministic_and_says_so(self):
        result, _ = self.run_model([RuntimeError("simulated model outage")])
        self.assertIn("fallback", result.report.planner)
        self.assertIn("simulated model outage", result.model_error)
        self.assertEqual(result.report.status, "VERIFIED_PENDING_APPROVAL")

    def test_no_fallback_when_disabled(self):
        with self.assertRaises(Exception):
            self.run_model([RuntimeError("simulated model outage")], fallback_on_model_error=False)


class TestOfflineNoKey(unittest.TestCase):
    """The no-key path must not touch the network: any non-loopback connection raises."""

    def test_pipeline_runs_with_network_blocked(self):
        real_connect = socket.socket.connect

        def guarded_connect(sock, address):
            host = address[0] if isinstance(address, tuple) else address
            if host not in ("127.0.0.1", "::1", "localhost"):
                raise AssertionError(f"Network access attempted: {address}")
            return real_connect(sock, address)

        def blocked_getaddrinfo(host, *args, **kwargs):
            raise AssertionError(f"DNS lookup attempted: {host}")

        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(socket.socket, "connect", guarded_connect), \
             mock.patch.object(socket, "getaddrinfo", blocked_getaddrinfo):
            for backend in BACKENDS:
                report = run_adk_investigation(load_claim("scenario_1_valid"), backend=backend, use_llm=True).report
                self.assertEqual(report.planner, "deterministic")
                self.assertEqual(report.payout.proposed_payout, 251.0)


class TestAdkEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        import configs.settings
        importlib.reload(configs.settings)
        import src.ui.app as app_module
        cls.client = TestClient(importlib.reload(app_module).app)

    def test_adk_endpoint_both_backends(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                res = self.client.post(f"/api/adk/investigate/scenario_3_conflict?backend={backend}")
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["report"]["status"], "CONFLICT_ESCALATED")
                self.assertEqual(data["report"]["payout"]["payout_status"], "NOT_PAYABLE")
                self.assertIn("bb:verifier", data["blackboard_keys"])
                self.assertTrue(Path(data["report_files"]["json"]).exists())

    def test_adk_endpoint_validation(self):
        self.assertEqual(self.client.post("/api/adk/investigate/nope").status_code, 404)
        self.assertEqual(self.client.post("/api/adk/investigate/scenario_1_valid?backend=other").status_code, 422)

    def test_runtime_status_reports_adk_execution_separately_from_gemini(self):
        self.client.post("/api/adk/investigate/scenario_1_valid")
        status = self.client.get("/api/runtime-status").json()
        self.assertTrue(status["google_adk"]["executed_in_this_process"])
        self.assertEqual(status["google_adk"]["installed_version"], "2.12.0")
        if not os.environ.get("GOOGLE_API_KEY"):
            self.assertFalse(status["gemini_configured_for_adk"])

    def test_existing_endpoint_unchanged(self):
        data = self.client.post("/api/investigate/scenario_1_valid").json()
        self.assertEqual(set(data), {"state", "payout", "audit_trail", "blackboard", "monitoring", "mcp_tools", "runtime"})
        self.assertEqual(data["payout"]["proposed_payout"], 251.0)


if __name__ == "__main__":
    unittest.main()
