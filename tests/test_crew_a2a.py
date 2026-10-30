"""ADK side of the A2A link: the request text, and how the crew's reply reaches the blackboard."""
import json
import unittest
from types import SimpleNamespace

from google.genai import types

from claimsleuth import tracing
from claimsleuth.agents import fraud_remote
from claimsleuth.agents.common import (BB_CLAIM, BB_CREW_METRICS, BB_DOCUMENTS, BB_FACTS, BB_FRAUD, BB_GUARDRAIL_HITS,
                                       read, submit_report)
from claimsleuth.models import ManagerSubmission

FRAUD = {"findings": [{"kind": "serial_number_mismatch", "severity": "high", "detail": "Serials differ",
                       "quotes": [{"doc_id": "CLM-2024-003/invoice_003.txt", "line": 9,
                                   "text": "Serial Number: REF-HC-77100-A"}]}], "flagged": True}
TRACE = "a" * 32


def state():
    return {BB_CLAIM: json.dumps({"id": "CLM-2024-003", "claimant": "David Sterling", "statement": "Freezer broke"}),
            BB_DOCUMENTS: json.dumps([{"doc_id": "CLM-2024-003/invoice_003.txt", "kind": "invoice", "filename": "x"}]),
            BB_FACTS: json.dumps({"facts": [], "missing": []}), BB_GUARDRAIL_HITS: "[]"}


def callback_context(st, reply_text=None, error=None):
    content = types.Content(role="model", parts=[types.Part(text=reply_text)]) if reply_text is not None else None
    event = SimpleNamespace(author="fraud_crew", content=content, error_message=error)
    return SimpleNamespace(state=st, _invocation_context=SimpleNamespace(session=SimpleNamespace(events=[event])),
                           agent_name="fraud_crew", actions=SimpleNamespace(skip_summarization=False))


class TestRequest(unittest.TestCase):

    def test_request_starts_with_the_claim_id_and_carries_the_documents(self):
        text = fraud_remote.request_text(state())
        self.assertTrue(text.startswith("Investigate claim CLM-2024-003\n"))
        body = json.loads(text.split("\n", 1)[1])
        self.assertEqual(body["documents"], ["CLM-2024-003/invoice_003.txt"])
        self.assertEqual(body["claimant"], "David Sterling")


class TestReply(unittest.TestCase):

    def test_valid_reply_becomes_bb_fraud_and_its_spans_are_stored(self):
        st = state()
        span = {"trace_id": TRACE, "span_id": "b" * 16, "parent_span_id": "c" * 16, "name": "crewai.crew",
                "start_ns": 1, "end_ns": 2, "duration_ms": 0.0, "status": "OK", "attributes": {}}
        reply = {"fraud": FRAUD, "metrics": {"seconds": 5.0, "per_agent": {}}, "spans": [span]}
        fraud_remote._capture_reply(callback_context(st, json.dumps(reply)))
        self.assertEqual(read(st, BB_FRAUD), FRAUD)
        self.assertEqual(read(st, BB_CREW_METRICS)["seconds"], 5.0)
        self.assertIn("crewai.crew", [s["name"] for s in tracing.read_trace(TRACE)])

    def test_unusable_reply_is_a_bad_output_hit_and_forces_full_review(self):
        for text, error in (("Fraud crew failed: TimeoutError", None), ('{"fraud": {"flagged": "maybe"}}', None),
                            (None, "connection refused")):
            with self.subTest(text=text, error=error):
                st = state()
                fraud_remote._capture_reply(callback_context(st, text, error))
                self.assertNotIn(BB_FRAUD, st)
                hit = read(st, BB_GUARDRAIL_HITS)[0]
                self.assertEqual((hit["kind"], hit["where"]), ("bad_output", "fraud_crew"))
                submit_report(ManagerSubmission(decision="reject", reasons=["no fraud result"]), callback_context(st))
                self.assertEqual(json.loads(st["bb_report"])["review"], "full_review")
