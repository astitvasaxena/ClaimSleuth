"""
Stage B: ADK investigation results go through the SAME validated human-adjudication
logic as the original loop, and only a human approval can make a payout AUTHORIZED.
"""
import importlib
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from src.adk_app.pipeline import run_adk_investigation
from src.adk_app.report import FinalReport
from tests.test_adk_pipeline import ScriptedFakeLlm, investigator_script, load_claim

LEGACY_KEYS = {"state", "payout", "audit_trail", "blackboard", "monitoring", "mcp_tools", "runtime"}


def load_app(token=None):
    env = {"CLAIMSLEUTH_ADJUDICATION_TOKEN": token} if token else {}
    with mock.patch.dict(os.environ, env):
        if not token:
            os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        import configs.settings
        importlib.reload(configs.settings)
        import src.ui.app as app_module
        app_module = importlib.reload(app_module)
    return app_module


class AdkAdjudicationBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app_module = load_app()
        cls.client = TestClient(cls.app_module.app)

    def adk(self, scenario, tools="local"):
        res = self.client.post(f"/api/adk/investigate/{scenario}?tools={tools}")
        self.assertEqual(res.status_code, 200, res.text)
        return res.json()

    def decide(self, claim_id, decision, headers=None):
        return self.client.post(f"/api/adjudicate/{claim_id}", json={"decision": decision, "notes": "test"},
                                headers=headers or {})


class TestAdkApprovalFlow(AdkAdjudicationBase):

    def test_investigation_never_reports_authorized(self):
        for scenario in ("scenario_1_valid", "scenario_2_expired", "scenario_3_conflict"):
            data = self.adk(scenario)
            self.assertNotEqual(data["payout"]["payout_status"], "AUTHORIZED")
            self.assertIsNone(data["payout"]["authorized_payout"])
            self.assertNotEqual(data["report"]["payout"]["payout_status"], "AUTHORIZED")
            self.assertIsNone(data["report"]["human_decision"])

    def test_approve_verified_adk_claim_authorizes_payout(self):
        data = self.adk("scenario_1_valid")
        self.assertEqual(data["payout"]["payout_status"], "PROPOSED_PENDING_APPROVAL")
        res = self.decide("CLM-2024-001", "APPROVED")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "CLOSED_SETTLED")
        self.assertEqual(body["payout"]["payout_status"], "AUTHORIZED")
        self.assertEqual(body["payout"]["authorized_payout"], 251.0)
        self.assertEqual(body["investigation_source"], "adk")
        self.assertEqual(body["run_id"], data["report"]["run_id"])

        decided = FinalReport.model_validate_json(Path(body["adjudicated_report_files"]["json"]).read_text("utf-8"))
        self.assertEqual(decided.payout.payout_status, "AUTHORIZED")
        self.assertEqual(decided.human_decision, "APPROVED")
        self.assertTrue(decided.audit_chain_valid)
        self.assertIn("Human decision:** APPROVED", Path(body["adjudicated_report_files"]["markdown"]).read_text("utf-8"))
        # The original investigation report on disk is unchanged
        original = FinalReport.model_validate_json(Path(data["report_files"]["json"]).read_text("utf-8"))
        self.assertEqual(original.payout.payout_status, "PROPOSED_PENDING_APPROVAL")
        self.assertIsNone(original.human_decision)

        self.assertEqual(self.decide("CLM-2024-001", "APPROVED").status_code, 409)   # closed
        verify = self.client.get("/api/audit/CLM-2024-001/verify").json()
        self.assertTrue(verify["valid"], verify["error"])

    def test_conflicted_adk_claim_cannot_be_approved(self):
        self.adk("scenario_3_conflict")
        self.assertEqual(self.decide("CLM-2024-003", "APPROVED").status_code, 409)
        res = self.decide("CLM-2024-003", "REJECTED").json()
        self.assertEqual(res["status"], "CLOSED_REJECTED")
        self.assertEqual(res["payout"]["payout_status"], "NOT_PAYABLE")
        decided = json.loads(Path(res["adjudicated_report_files"]["json"]).read_text("utf-8"))
        self.assertEqual(decided["payout"]["payout_status"], "NOT_PAYABLE")
        self.assertEqual(self.decide("CLM-2024-003", "APPROVED").status_code, 409)

    def test_info_request_on_expired_then_reject(self):
        self.adk("scenario_2_expired")
        res = self.decide("CLM-2024-002", "INFO_REQUESTED").json()
        self.assertEqual(res["status"], "NEEDS_EVIDENCE")
        self.assertEqual(res["payout"]["payout_status"], "NOT_PAYABLE")
        self.assertEqual(self.decide("CLM-2024-002", "APPROVED").status_code, 409)
        self.assertEqual(self.decide("CLM-2024-002", "REJECTED").json()["status"], "CLOSED_REJECTED")

    def test_info_request_withdraws_proposal(self):
        self.adk("scenario_1_valid")
        res = self.decide("CLM-2024-001", "INFO_REQUESTED").json()
        self.assertEqual(res["payout"]["payout_status"], "NOT_PAYABLE")
        self.assertEqual(self.decide("CLM-2024-001", "APPROVED").status_code, 409)

    def test_invalid_decision_rejected(self):
        self.adk("scenario_1_valid")
        self.assertEqual(self.decide("CLM-2024-001", "PAY_NOW").status_code, 422)

    def test_approval_through_mcp_path(self):
        data = self.adk("scenario_1_valid", tools="mcp_stdio")
        self.assertEqual(data["report"]["tool_backend"], "mcp_stdio")
        body = self.decide("CLM-2024-001", "APPROVED").json()
        self.assertEqual(body["payout"]["authorized_payout"], 251.0)


class TestModelMismatchCannotBeApproved(AdkAdjudicationBase):

    def test_registered_model_mismatch_blocked(self):
        model = ScriptedFakeLlm(model="fake", script=investigator_script(
            "invoice_001.txt", "repair_estimate_001.txt", payout_claim=999.0))
        result = run_adk_investigation(load_claim("scenario_1_valid"), model=model)
        # Register exactly as the endpoint does
        self.app_module.active_states["CLM-2024-001"] = result.claim_state
        self.app_module.active_runs["CLM-2024-001"] = {"source": "adk", "run_id": result.report.run_id,
                                                       "report": result.report}
        self.assertEqual(self.decide("CLM-2024-001", "APPROVED").status_code, 409)


class TestEngineInterplayAndCompatibility(AdkAdjudicationBase):

    def test_legacy_endpoint_unchanged_and_takes_over_claim(self):
        self.adk("scenario_1_valid")
        legacy = self.client.post("/api/investigate/scenario_1_valid").json()
        self.assertEqual(set(legacy), LEGACY_KEYS)
        body = self.decide("CLM-2024-001", "APPROVED").json()
        self.assertEqual(body["investigation_source"], "legacy")
        self.assertIsNone(body["adjudicated_report_files"])
        self.assertEqual(body["payout"]["payout_status"], "AUTHORIZED")

    def test_adk_response_is_dashboard_compatible(self):
        data = self.adk("scenario_3_conflict")
        self.assertTrue(LEGACY_KEYS - {"mcp_tools"} <= set(data))
        steps = data["state"]["steps_history"]
        self.assertEqual([s["description"].split(" ")[0] for s in steps],
                         ["IntakeAgent", "EvidenceAgent", "AnomalyAgent", "WarrantyAgent",
                          "SettlementAgent", "VerifierAgent", "ReportAgent"])
        self.assertEqual(steps[-1]["phase"], "STOP")
        self.assertIn("bb:conflicts", data["blackboard"]["facts"])
        self.assertTrue(all(m["intent"] == "STATE_DELTA" for m in data["blackboard"]["messages"]))
        self.assertIsNone(data["monitoring"]["avg_step_latency_ms"])
        self.assertEqual(data["state"]["status"], "CONFLICT_ESCALATED")
        self.assertTrue(data["audit_trail"])

    def test_dashboard_offers_engine_selector(self):
        html = self.client.get("/").text
        self.assertIn('id="engineSelect"', html)
        self.assertIn("/api/adk/investigate/", html)
        self.assertNotIn("|| 12.0", html)


class TestTokenAppliesToAdkRuns(unittest.TestCase):

    def test_adk_run_adjudication_requires_token_when_configured(self):
        app_module = load_app(token="adk-test-token")
        try:
            client = TestClient(app_module.app)
            self.assertEqual(client.post("/api/adk/investigate/scenario_1_valid").status_code, 200)
            body = {"decision": "APPROVED"}
            self.assertEqual(client.post("/api/adjudicate/CLM-2024-001", json=body).status_code, 401)
            self.assertEqual(client.post("/api/adjudicate/CLM-2024-001", json=body,
                                         headers={"Authorization": "Bearer wrong"}).status_code, 403)
            ok = client.post("/api/adjudicate/CLM-2024-001", json=body,
                             headers={"Authorization": "Bearer adk-test-token"})
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(ok.json()["payout"]["payout_status"], "AUTHORIZED")
        finally:
            load_app()


if __name__ == "__main__":
    unittest.main()
