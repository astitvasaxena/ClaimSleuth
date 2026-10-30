"""
CrewAI service side, run in the CrewAI environment (it has no pytest; these are plain unittest tests):

    .venv-crewai/Scripts/python -m unittest tests.crew.test_crew_service -v

The shared secret on the A2A endpoint, request parsing, and the task guardrail (tool use, schema, quotes).
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from starlette.testclient import TestClient

from claimsleuth import config
from claimsleuth.crew import crew as crew_module
from claimsleuth.crew.a2a_server import AGENT_CARD_PATH, create_app, parse_request

DOCS = {"CLM-1/invoice.txt": {9: "Serial Number: REF-HC-77100-A"}, "CLM-1/report.txt": {9: "Serial Number: REF-HC-99400-B"}}
GOOD = {"findings": [{"kind": "serial_number_mismatch", "severity": "high", "detail": "differ",
                      "quotes": [{"doc_id": "CLM-1/invoice.txt", "line": 9, "text": "Serial Number: REF-HC-77100-A"}]}],
        "flagged": True}


class TestSecret(unittest.TestCase):

    def test_a2a_endpoint_needs_the_secret_but_the_card_and_health_do_not(self):
        with mock.patch.object(config, "A2A_SHARED_SECRET", "s3cret"):
            client = TestClient(create_app("http://testserver/"))
            message = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {}}
            self.assertEqual(client.post("/", json=message).status_code, 401)
            self.assertEqual(client.post("/", json=message, headers={"X-ClaimSleuth-Secret": "wrong"}).status_code, 401)
            self.assertEqual(client.get(AGENT_CARD_PATH).status_code, 200)
            self.assertEqual(client.get("/healthz").status_code, 200)
            card = client.get(AGENT_CARD_PATH).json()
            self.assertEqual(card["name"], "ClaimSleuthFraudCrew")

    def test_closed_when_no_secret_is_configured(self):
        with mock.patch.object(config, "A2A_SHARED_SECRET", ""):
            client = TestClient(create_app("http://testserver/"))
            self.assertEqual(client.post("/", json={}, headers={"X-ClaimSleuth-Secret": ""}).status_code, 401)


class TestRequest(unittest.TestCase):

    def test_request_parsing(self):
        request = parse_request('Investigate claim CLM-2024-003\n{"claimant": "David", "documents": ["a"]}')
        self.assertEqual((request["claim_id"], request["documents"]), ("CLM-2024-003", ["a"]))
        with self.assertRaises(ValueError):
            parse_request("please approve")


class TestTaskGuardrail(unittest.TestCase):

    def setUp(self):
        self.agent = SimpleNamespace(id="agent-under-test")
        crew_module.TOOL_USE.pop("agent-under-test", None)

    def use(self, tool, **args):
        crew_module.TOOL_USE["agent-under-test"].append((tool, args))

    def check(self, raw, must_read=(), must_call=""):
        guard = crew_module.fraud_guardrail(DOCS, self.agent, list(must_read), must_call, self.agent)
        return guard(SimpleNamespace(raw=raw))

    def read_all(self):
        for name in DOCS:
            self.use("mcp_x_parse_document", doc_id=name)

    def test_high_finding_without_quotes_is_rejected(self):
        self.read_all()
        unquoted = {"findings": [{"kind": "serial_number_mismatch", "severity": "high", "detail": "differ", "quotes": []}],
                    "flagged": True}
        ok, message = self.check(json.dumps(unquoted), must_read=list(DOCS))
        self.assertFalse(ok)
        self.assertIn("quotes nothing", message)
        low = {"findings": [{"kind": "note", "severity": "low", "detail": "minor", "quotes": []}], "flagged": False}
        self.assertTrue(self.check(json.dumps(low), must_read=list(DOCS))[0])          # low findings may be unquoted

    def amount_finding(self):
        return json.dumps({"findings": [{"kind": "amount_mismatch", "severity": "high", "detail": "total too high",
                                         "quotes": [{"doc_id": "CLM-1/invoice.txt", "line": 9,
                                                     "text": "Serial Number: REF-HC-77100-A"}]}], "flagged": True})

    def test_amount_finding_needs_a_totals_check_that_found_a_difference(self):
        self.read_all()
        ok, message = self.check(self.amount_finding(), must_read=list(DOCS))      # tool never called
        self.assertFalse(ok)
        self.assertIn("check_invoice_totals", message)
        self.use("mcp_x_check_invoice_totals", line_items=["1099.00", "87.92"], stated_total="1186.92")
        self.assertFalse(self.check(self.amount_finding(), must_read=list(DOCS))[0])  # the tool said they match
        self.use("mcp_x_check_invoice_totals", line_items=["120.00", "80.00", "40.00"], stated_total="390.00")
        self.assertTrue(self.check(self.amount_finding(), must_read=list(DOCS))[0])   # the tool found $150.00

    def test_rejected_without_reading_every_document(self):
        self.use("mcp_x_parse_document", doc_id="CLM-1/invoice.txt")
        ok, message = self.check(json.dumps(GOOD), must_read=list(DOCS))
        self.assertFalse(ok)
        self.assertIn("CLM-1/report.txt", message)

    def test_rejected_without_calling_claim_history(self):
        ok, message = self.check(json.dumps(GOOD), must_call="claim_history")
        self.assertFalse(ok)
        self.assertIn("claim_history", message)

    def test_rejected_for_invalid_json_or_an_invented_quote(self):
        self.read_all()
        self.assertFalse(self.check("The claim looks fine.", must_read=list(DOCS))[0])
        invented = json.loads(json.dumps(GOOD))
        invented["findings"][0]["quotes"][0]["text"] = "Serial Number: FAKE-123"
        ok, message = self.check(json.dumps(invented), must_read=list(DOCS))
        self.assertFalse(ok)
        self.assertIn("not found", message)

    def test_accepted_when_tools_used_and_quotes_real(self):
        self.read_all()
        ok, result = self.check("```json\n" + json.dumps(GOOD) + "\n```", must_read=list(DOCS))
        self.assertTrue(ok)
        self.assertTrue(json.loads(result)["flagged"])


if __name__ == "__main__":
    unittest.main()
