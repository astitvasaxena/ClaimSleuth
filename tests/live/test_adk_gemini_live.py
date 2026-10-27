"""
LIVE integration test: ADK pipeline with a real Gemini model.

Skipped unless BOTH are set:
    GOOGLE_API_KEY            (never printed or logged)
    CLAIMSLEUTH_LIVE_TESTS=1  (explicit opt-in; makes paid/quota-limited API calls)

Run separately:
    python -m pytest tests/live -v -rs

Fallback is disabled, so a Gemini failure fails this test instead of silently
passing through the deterministic planner.
"""
import json
import os
import unittest
from pathlib import Path

from configs.settings import GEMINI_MODEL
from src.adk_app.pipeline import run_adk_investigation

LIVE = bool(os.environ.get("GOOGLE_API_KEY")) and os.environ.get("CLAIMSLEUTH_LIVE_TESTS") == "1"
CLAIMS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample_claims"


@unittest.skipUnless(LIVE, "Live Gemini test: set GOOGLE_API_KEY and CLAIMSLEUTH_LIVE_TESTS=1 to run")
class TestAdkGeminiLive(unittest.TestCase):

    def run_scenario(self, scenario_id):
        claim = json.loads((CLAIMS_DIR / scenario_id / "claim.json").read_text(encoding="utf-8"))
        return run_adk_investigation(claim, backend="workflow", model=GEMINI_MODEL,
                                     fallback_on_model_error=False).report

    def test_gemini_investigator_uses_tools_and_verifier_decides(self):
        report = self.run_scenario("scenario_1_valid")
        self.assertEqual(report.planner, f"llm:{GEMINI_MODEL}")
        self.assertIn("InvestigatorLLM", report.agent_sequence)
        self.assertTrue(report.model_tool_calls, "Gemini made no tool calls")
        # Whatever the model did, the payout can only be the deterministic one or none at all.
        self.assertIn(report.payout.payout_status, ("PROPOSED_PENDING_APPROVAL", "NOT_PAYABLE"))
        if report.payout.proposed_payout is not None:
            self.assertEqual(report.payout.proposed_payout, 251.0)
        self.assertIsNone(report.payout.authorized_payout)

    def test_gemini_cannot_clear_a_serial_conflict(self):
        report = self.run_scenario("scenario_3_conflict")
        self.assertIn(report.status, ("CONFLICT_ESCALATED", "NEEDS_EVIDENCE"))
        self.assertEqual(report.payout.payout_status, "NOT_PAYABLE")


if __name__ == "__main__":
    unittest.main()
