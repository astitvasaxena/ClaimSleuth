"""
Regression tests (27 Oct 2026, follow-up to WP-1):

1. Scenario 3 showed a "$546.25 settlement" on an escalated claim. The verifier had
   written its own recalculation into state.deterministic_settlement. It now reports
   it as a diagnostic only, and the API exposes an explicit payout view.
2. Legacy audit ledgers (committed with broken chains) are classified by re-running
   verification, and new ledgers are written to a separate location.
"""
import importlib
import json
import os
import unittest
from pathlib import Path

from src.core.state import ClaimInvestigationState, EvidenceItem, VerifierReport
from src.agents.investigator import LeadInvestigatorAgent
from src.verifier.independent_verifier import DualGateVerifier
from src.tools.policy_rules import load_policy

CLAIMS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_claims"


def load_claim(scenario_id: str) -> dict:
    with open(CLAIMS_DIR / scenario_id / "claim.json", encoding="utf-8") as f:
        return json.load(f)


def load_app_module():
    os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
    import configs.settings
    importlib.reload(configs.settings)
    import src.ui.app as app_module
    return importlib.reload(app_module)


class TestVerifierDoesNotSupplyPayout(unittest.TestCase):

    def test_conflicted_claim_has_no_settlement_only_diagnostic(self):
        state = LeadInvestigatorAgent(use_llm=False).run_investigation(
            ClaimInvestigationState(**load_claim("scenario_3_conflict")))
        self.assertEqual(state.status, "CONFLICT_ESCALATED")
        self.assertIsNone(state.deterministic_settlement)
        self.assertFalse(state.verifier_report.is_approved_by_verifier)
        # (650 - 75) * 0.95 = 546.25, kept only as the verifier's diagnostic
        self.assertEqual(state.verifier_report.recalculated_settlement.approved_payout, 546.25)

    def test_missing_proposed_settlement_blocks_verifier_approval(self):
        state = ClaimInvestigationState(**load_claim("scenario_1_valid"))
        for field, value, snippet in (("purchase_date", "2024-03-15", "Purchase Date: 2024-03-15"),
                                      ("serial_number", "APPL-IPH15-998821", "Serial Number: APPL-IPH15-998821"),
                                      ("repair_cost", 280.0, "Estimated Cost: $280.00")):
            state.evidence[field] = EvidenceItem(evidence_id=f"ev_{field}", source_document="doc.txt",
                                                 field_name=field, extracted_value=value,
                                                 raw_snippet=snippet, confidence=0.95, status="VERIFIED")
        report = DualGateVerifier().verify_investigation(state, load_policy("POL-TECH-CARE-01"))
        self.assertFalse(report.is_approved_by_verifier)
        self.assertIsNone(state.deterministic_settlement)
        self.assertTrue(any("Missing proposed settlement" in d for d in report.discrepancies))
        self.assertEqual(report.recalculated_settlement.approved_payout, 251.0)


class TestPayoutView(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(load_app_module().app)

    def investigate(self, scenario_id):
        return self.client.post(f"/api/investigate/{scenario_id}").json()

    def decide(self, claim_id, decision):
        return self.client.post(f"/api/adjudicate/{claim_id}", json={"decision": decision})

    def test_conflicted_claim_payout_is_not_payable_with_labelled_diagnostic(self):
        data = self.investigate("scenario_3_conflict")
        payout = data["payout"]
        self.assertIsNone(data["state"]["deterministic_settlement"])
        self.assertEqual(payout["payout_status"], "NOT_PAYABLE")
        self.assertIsNone(payout["authorized_payout"])
        self.assertIsNone(payout["proposed_payout"])
        self.assertEqual(payout["diagnostic_calculation"]["approved_payout"], 546.25)
        self.assertIn("diagnostic calculation only", payout["explanation"])

    def test_conflicted_claim_stays_not_payable_after_human_rejection(self):
        self.investigate("scenario_3_conflict")
        res = self.decide("CLM-2024-003", "REJECTED").json()
        self.assertEqual(res["payout"]["payout_status"], "NOT_PAYABLE")
        self.assertIsNone(res["payout"]["authorized_payout"])

    def test_verified_claim_is_only_proposed_until_human_approval(self):
        payout = self.investigate("scenario_1_valid")["payout"]
        self.assertEqual(payout["payout_status"], "PROPOSED_PENDING_APPROVAL")
        self.assertEqual(payout["proposed_payout"], 251.0)
        self.assertIsNone(payout["authorized_payout"])
        res = self.decide("CLM-2024-001", "APPROVED").json()
        self.assertEqual(res["payout"]["payout_status"], "AUTHORIZED")
        self.assertEqual(res["payout"]["authorized_payout"], 251.0)

    def test_info_request_withdraws_the_proposal(self):
        self.investigate("scenario_1_valid")
        res = self.decide("CLM-2024-001", "INFO_REQUESTED").json()
        self.assertEqual(res["payout"]["payout_status"], "NOT_PAYABLE")
        self.assertIsNone(res["payout"]["proposed_payout"])

    def test_expired_claim_not_payable(self):
        payout = self.investigate("scenario_2_expired")["payout"]
        self.assertEqual(payout["payout_status"], "NOT_PAYABLE")
        self.assertIsNone(payout["authorized_payout"])


class TestAuditLogLocations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(load_app_module().app)

    def test_legacy_logs_reported_invalid_by_real_verification(self):
        data = self.client.get("/api/audit-legacy").json()
        self.assertEqual(len(data["logs"]), 4)
        for entry in data["logs"]:
            self.assertEqual(entry["classification"], "LEGACY_INVALID", entry["file"])
            self.assertFalse(entry["verify_chain"]["valid"])

    def test_default_runtime_dir_is_separate_from_legacy_dir(self):
        import configs.settings as settings
        saved = os.environ.pop("CLAIMSLEUTH_AUDIT_LOG_DIR")
        try:
            importlib.reload(settings)
            self.assertNotEqual(settings.AUDIT_LOG_DIR.resolve(), settings.LEGACY_AUDIT_LOG_DIR.resolve())
            self.assertEqual(settings.AUDIT_LOG_DIR.parts[-2:], ("runtime", "audit_logs"))
        finally:
            os.environ["CLAIMSLEUTH_AUDIT_LOG_DIR"] = saved
            importlib.reload(settings)

    def test_new_chain_valid_across_runs_and_decisions(self):
        for _ in range(3):
            self.client.post("/api/investigate/scenario_3_conflict")
        self.client.post("/api/adjudicate/CLM-2024-003", json={"decision": "APPROVED"})   # blocked, logged
        self.client.post("/api/adjudicate/CLM-2024-003", json={"decision": "REJECTED"})
        verify = self.client.get("/api/audit/CLM-2024-003/verify").json()
        self.assertTrue(verify["valid"], verify["error"])
        self.assertGreater(verify["entries_checked"], 30)


if __name__ == "__main__":
    unittest.main()
