"""
Unit tests for the CrewAI forensics service. These need crewai, so they run in the
separate environment (pytest's main-env collection skips this folder):

    .venv-crewai/Scripts/python -m unittest tests.crew_service.test_forensics_flow -v

tests/test_crewai_a2a.py runs this command as part of the main suite.
"""
import json
import os
import socket
import unittest
from pathlib import Path
from unittest import mock

import crewai
from pydantic import ValidationError

from src.core.forensics_schema import ForensicsReport
from src.crew_service.a2a_server import build_agent_card, extract_claim_id, load_claim
from src.crew_service.forensics_flow import ForensicsFlow, run_forensics
from tests.fixtures import fake_crewai_llm

CLAIMS = Path(__file__).resolve().parent.parent.parent / "data" / "sample_claims"


def claim(scenario):
    return json.loads((CLAIMS / scenario / "claim.json").read_text(encoding="utf-8"))


class TestCrewAIFlow(unittest.TestCase):

    def test_real_crewai_flow_runs_every_step(self):
        self.assertTrue(issubclass(ForensicsFlow, crewai.Flow))
        report = run_forensics(claim("scenario_1_valid"))
        self.assertEqual(report.flow_steps, ["load_dossier", "extract_facts", "check_serials",
                                             "check_chronology", "scan_exclusions", "write_narrative"])
        self.assertEqual(report.crewai_version, crewai.__version__)
        self.assertEqual(report.recommendation, "NO_FORENSIC_ISSUES")
        self.assertEqual(report.narrative_source, "not_generated")

    def test_serial_conflict_flagged_with_evidence(self):
        report = run_forensics(claim("scenario_3_conflict"))
        serial = next(f for f in report.findings if f.check == "serial_consistency")
        self.assertEqual(serial.result, "FLAG")
        self.assertTrue(any("REF-HC-99400-B" in e for e in serial.evidence))
        self.assertEqual(report.recommendation, "ESCALATE_FOR_REVIEW")

    def test_report_has_no_payout_fields(self):
        fields = set(ForensicsReport.model_fields)
        self.assertFalse(fields & {"approved_payout", "payout", "settlement", "authorize_payout", "decision"})
        data = json.loads(run_forensics(claim("scenario_1_valid")).model_dump_json())
        data["approved_payout"] = 999.0
        with self.assertRaises(ValidationError):
            ForensicsReport(**data)


class TestCrewAICrewWithFakeLLM(unittest.TestCase):
    """TEST DOUBLE LLM: proves the real Crew/Agent/Task path runs; not a live model."""

    def test_crew_writes_narrative(self):
        before = fake_crewai_llm.CALLS["count"]
        report = run_forensics(claim("scenario_3_conflict"), llm=fake_crewai_llm.FakeReviewerLLM())
        self.assertEqual(fake_crewai_llm.CALLS["count"], before + 1)
        self.assertEqual(report.narrative_source, "crewai_crew:fake-reviewer")
        self.assertIn("no payout decision", report.narrative)
        self.assertEqual(report.recommendation, "ESCALATE_FOR_REVIEW")   # narrative does not change findings


class TestOfflineAndTelemetry(unittest.TestCase):

    def test_telemetry_disabled(self):
        for var in ("CREWAI_DISABLE_TELEMETRY", "CREWAI_DISABLE_TRACKING", "OTEL_SDK_DISABLED"):
            self.assertEqual(os.environ.get(var), "true", var)

    def test_flow_and_crew_run_with_network_blocked(self):
        real_connect = socket.socket.connect

        def guarded(sock, address):
            host = address[0] if isinstance(address, tuple) else address
            if host not in ("127.0.0.1", "::1", "localhost"):
                raise AssertionError(f"network access attempted: {address}")
            return real_connect(sock, address)

        def no_dns(host, *a, **k):
            raise AssertionError(f"DNS lookup attempted: {host}")

        with mock.patch.object(socket.socket, "connect", guarded), mock.patch.object(socket, "getaddrinfo", no_dns):
            report = run_forensics(claim("scenario_1_valid"), llm=fake_crewai_llm.FakeReviewerLLM())
        self.assertEqual(report.narrative_source, "crewai_crew:fake-reviewer")


class TestServiceHelpers(unittest.TestCase):

    def test_claim_id_extraction_and_dossier_lookup(self):
        self.assertEqual(extract_claim_id('Investigate:\n{"claim_id": "CLM-2024-003", "x": 1}'), "CLM-2024-003")
        self.assertIsNone(extract_claim_id("no claim here"))
        self.assertIsNone(load_claim("CLM-NOPE-000"))
        self.assertEqual(load_claim("CLM-2024-002")["policy_id"], "POL-LAPTOP-STD-02")

    def test_agent_card(self):
        card = build_agent_card("http://127.0.0.1:8101/")
        self.assertEqual(card.supported_interfaces[0].protocol_binding, "JSONRPC")
        self.assertEqual(card.skills[0].id, "claim_forensics")


if __name__ == "__main__":
    unittest.main()
