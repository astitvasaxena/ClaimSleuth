"""
ClaimSleuth — WP-1 regression tests for the defects found in the 27 Oct 2026 baseline.

D-3  Audit hash chain restarted at GENESIS whenever a new AuditLogger was created.
D-2  MCP compute_settlement result was dropped; the loop repeated the tool until the step cap.
D-1/D-11  CrewAI placeholder invented evidence and approved conflicting/nonexistent claims.
C4   Human decisions were free strings with no state-transition guard.
C6   API/UI reported ADK/CrewAI as "Active" regardless of what actually ran.
"""
import importlib
import json
import os
import unittest
from pathlib import Path

from src.core.audit import AuditLogger
from src.core.state import ClaimInvestigationState, SettlementCalc
from src.agents.investigator import LeadInvestigatorAgent
from src.core.blackboard import Blackboard
from src.frameworks.crewai_adapter import ClaimInvestigationCrew
from configs.settings import MAX_INVESTIGATION_STEPS

CLAIMS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_claims"


def load_claim(scenario_id: str) -> dict:
    with open(CLAIMS_DIR / scenario_id / "claim.json", encoding="utf-8") as f:
        return json.load(f)


def load_app_module():
    """Reload settings + app so no adjudication token leaks in from other tests."""
    os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
    import configs.settings
    importlib.reload(configs.settings)
    import src.ui.app as app_module
    return importlib.reload(app_module)


class TestAuditChainContinuity(unittest.TestCase):
    """D-3: the chain must stay valid across logger instances and repeated runs."""

    def test_second_logger_instance_continues_chain(self):
        first = AuditLogger("TEST-CONT-001")
        first.log_step(1, "PLAN", "first instance")
        second = AuditLogger("TEST-CONT-001")
        second.log_step(2, "PLAN", "second instance")
        result = second.verify_chain()
        self.assertTrue(result["valid"], result["error"])
        self.assertEqual(result["entries_checked"], 3)

    def test_interleaved_instances_keep_chain_valid(self):
        a = AuditLogger("TEST-CONT-002")
        b = AuditLogger("TEST-CONT-002")
        a.log_step(1, "PLAN", "from a")
        b.log_step(2, "PLAN", "from b")
        a.log_step(3, "PLAN", "from a again")
        self.assertTrue(a.verify_chain()["valid"])

    def test_repeated_investigations_keep_chain_valid(self):
        agent = LeadInvestigatorAgent(use_llm=False)
        claim = load_claim("scenario_1_valid")
        agent.run_investigation(ClaimInvestigationState(**claim))
        agent.run_investigation(ClaimInvestigationState(**claim))
        result = AuditLogger(claim["claim_id"]).verify_chain()
        self.assertTrue(result["valid"], result["error"])

    def test_tampering_still_detected_after_fix(self):
        logger = AuditLogger("TEST-CONT-TAMPER")
        logger.log_step(1, "PLAN", "original")
        lines = logger.log_file.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[1])
        entry["details"]["description"] = "edited"
        lines[1] = json.dumps(entry)
        logger.log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertFalse(AuditLogger("TEST-CONT-TAMPER").verify_chain()["valid"])


class TestSettlementReachesStop(unittest.TestCase):
    """D-2: settlement must be computed once and stored by the loop itself."""

    def setUp(self):
        agent = LeadInvestigatorAgent(use_llm=False)
        self.state = agent.run_investigation(ClaimInvestigationState(**load_claim("scenario_1_valid")))

    def test_settlement_tool_called_exactly_once(self):
        calls = [s for s in self.state.steps_history if s.tool_called == "compute_settlement"]
        self.assertEqual(len(calls), 1)

    def test_loop_stops_before_step_cap(self):
        self.assertLess(len(self.state.steps_history), MAX_INVESTIGATION_STEPS)
        self.assertIn("settlement computed", self.state.steps_history[-1].reasoning)

    def test_settlement_stored_as_model_with_expected_payout(self):
        self.assertIsInstance(self.state.deterministic_settlement, SettlementCalc)
        self.assertEqual(self.state.deterministic_settlement.approved_payout, 251.0)
        self.assertEqual(self.state.status, "VERIFIED_PENDING_APPROVAL")


class TestCrewPlaceholderFailsClosed(unittest.TestCase):
    """D-1 / D-11: no invented facts, conflicts escalate, policy terms come from the policy."""

    def kickoff(self, claim: dict) -> dict:
        return ClaimInvestigationCrew(claim["claim_id"], Blackboard(claim["claim_id"])).kickoff(claim)

    def test_nonexistent_claim_needs_evidence_without_payout(self):
        res = self.kickoff({"claim_id": "CLM-DOES-NOT-EXIST", "incident_date": "2023-08-10"})
        self.assertEqual(res["status"], "NEEDS_EVIDENCE")
        self.assertIsNone(res.get("settlement"))
        self.assertIn("purchase_date", res["missing_fields"])

    def test_serial_conflict_is_escalated_not_approved(self):
        res = self.kickoff(load_claim("scenario_3_conflict"))
        self.assertEqual(res["status"], "CONFLICT_ESCALATED")
        self.assertTrue(res["conflicts"])
        self.assertIsNone(res.get("settlement"))

    def test_valid_scenario_matches_main_pipeline(self):
        res = self.kickoff(load_claim("scenario_1_valid"))
        self.assertEqual(res["status"], "VERIFIED_PENDING_APPROVAL")
        self.assertEqual(res["settlement"]["approved_payout"], 251.0)

    def test_expired_scenario_rejected(self):
        res = self.kickoff(load_claim("scenario_2_expired"))
        self.assertEqual(res["status"], "REJECTED_DETERMINISTICALLY")

    def test_depreciation_from_policy_is_applied(self):
        # Laptop policy: deductible 50, depreciation 10% -> (450 - 50) * 0.9 = 360
        res = self.kickoff({
            "claim_id": "CLM-CREW-DEP", "policy_id": "POL-LAPTOP-STD-02",
            "purchase_date": "2024-01-10", "incident_date": "2024-04-12", "repair_cost": 450.0,
        })
        self.assertEqual(res["settlement"]["approved_payout"], 360.0)

    def test_result_declares_placeholder_implementation(self):
        res = self.kickoff(load_claim("scenario_1_valid"))
        self.assertIn("placeholder", res["implementation"].lower())


class TestAdjudicationGuards(unittest.TestCase):
    """C4: decisions are an enum and must respect the claim's state."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.app_module = load_app_module()
        cls.client = TestClient(cls.app_module.app)

    def investigate(self, scenario_id: str) -> dict:
        res = self.client.post(f"/api/investigate/{scenario_id}")
        self.assertEqual(res.status_code, 200)
        return res.json()["state"]

    def decide(self, claim_id: str, decision: str):
        return self.client.post(f"/api/adjudicate/{claim_id}", json={"decision": decision, "notes": "test"})

    def test_unknown_decision_rejected(self):
        self.investigate("scenario_1_valid")
        self.assertEqual(self.decide("CLM-2024-001", "LOLWHAT").status_code, 422)

    def test_unknown_claim_404(self):
        self.assertEqual(self.decide("CLM-NOPE", "APPROVED").status_code, 404)

    def test_conflicted_claim_cannot_be_approved(self):
        self.investigate("scenario_3_conflict")
        self.assertEqual(self.decide("CLM-2024-003", "APPROVED").status_code, 409)

    def test_expired_claim_cannot_be_approved(self):
        self.investigate("scenario_2_expired")
        self.assertEqual(self.decide("CLM-2024-002", "APPROVED").status_code, 409)

    def test_closed_claim_cannot_be_reopened(self):
        self.investigate("scenario_3_conflict")
        res = self.decide("CLM-2024-003", "REJECTED")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "CLOSED_REJECTED")
        self.assertEqual(self.decide("CLM-2024-003", "APPROVED").status_code, 409)

    def test_verified_claim_approval_then_audit_chain_valid(self):
        self.investigate("scenario_1_valid")
        res = self.decide("CLM-2024-001", "APPROVED")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "CLOSED_SETTLED")
        verify = self.client.get("/api/audit/CLM-2024-001/verify").json()
        self.assertTrue(verify["valid"], verify["error"])


class TestHonestStatusAndHealth(unittest.TestCase):
    """C6 + operability: truthful runtime labels, health check, bounded audit payloads, CORS."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(load_app_module().app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def test_runtime_status_is_not_hardcoded_active(self):
        data = self.client.post("/api/investigate/scenario_1_valid").json()
        runtime = data["runtime"]
        self.assertNotIn("Active", json.dumps(runtime))
        self.assertNotIn("frameworks", data)
        if not os.environ.get("GOOGLE_API_KEY"):
            self.assertEqual(runtime["planner_mode"], "deterministic_fallback")

    def test_investigation_returns_only_this_runs_audit_entries(self):
        self.client.post("/api/investigate/scenario_1_valid")
        data = self.client.post("/api/investigate/scenario_1_valid").json()
        trail = data["audit_trail"]
        self.assertLess(len(trail), 40)
        self.assertEqual(trail[0]["details"].get("step_num"), 0)

    def test_audit_endpoint_is_bounded(self):
        self.client.post("/api/investigate/scenario_1_valid")
        self.assertLessEqual(len(self.client.get("/api/audit/CLM-2024-001?limit=5").json()), 5)

    def test_cors_rejects_unlisted_origin(self):
        res = self.client.get("/api/scenarios", headers={"Origin": "http://evil.example"})
        self.assertNotIn("access-control-allow-origin", {k.lower() for k in res.headers})


if __name__ == "__main__":
    unittest.main()
