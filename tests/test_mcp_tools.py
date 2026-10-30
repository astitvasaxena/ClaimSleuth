"""The five MCP tools: normal, edge and error cases, and the shared-secret check on /mcp."""
import unittest
from datetime import date
from decimal import Decimal
from unittest import mock

from fastapi.testclient import TestClient
from mcp.server.mcpserver.exceptions import ToolError

from claimsleuth import config, db
from claimsleuth.api import app as api
from claimsleuth.mcp_server.server import (check_coverage, claim_history, compute_payout, parse_document,
                                           search_policy, server)

D = Decimal


class TestComputePayout(unittest.TestCase):

    def test_deductible_only(self):
        r = compute_payout(D("280.00"), D("29.00"), D("1200.00"))
        self.assertEqual(r.amount, D("251.00"))

    def test_limit_cap_and_depreciation(self):
        r = compute_payout(D("2000"), D("50"), D("1500"), D("10"))
        self.assertEqual(r.breakdown["capped_amount"], D("1500.00"))
        self.assertEqual(r.amount, D("1350.00"))

    def test_cents_rounding(self):
        r = compute_payout(D("100.00"), D("0"), D("1000"), D("33.333"))
        self.assertEqual(r.amount, D("66.67"))

    def test_deductible_larger_than_cost_pays_nothing(self):
        self.assertEqual(compute_payout(D("20"), D("29"), D("1200")).amount, D("0.00"))

    def test_invalid_amounts_are_errors(self):
        with self.assertRaises(ToolError):
            compute_payout(D("-1"), D("0"), D("10"))
        with self.assertRaises(ToolError):
            compute_payout(D("10"), D("0"), D("10"), D("100"))


class TestCheckCoverage(unittest.TestCase):

    def test_inside_window(self):
        r = check_coverage(date(2023, 6, 1), date(2024, 8, 10), 36, ["commercial use"], "compressor failure")
        self.assertTrue(r.within_window)
        self.assertEqual(r.days_since_purchase, 436)
        self.assertIsNone(r.matched_exclusion)

    def test_expired(self):
        r = check_coverage(date(2023, 1, 1), date(2024, 4, 12), 12, [], "power failure")
        self.assertFalse(r.within_window)
        self.assertLess(r.days_left, 0)

    def test_leap_year_end_date(self):
        r = check_coverage(date(2024, 2, 29), date(2025, 2, 28), 12, [], "x")
        self.assertEqual(r.warranty_end, date(2025, 2, 28))
        self.assertTrue(r.within_window)

    def test_incident_before_purchase_is_outside_window(self):
        r = check_coverage(date(2024, 5, 1), date(2024, 4, 1), 12, [], "x")
        self.assertFalse(r.within_window)
        self.assertLess(r.days_since_purchase, 0)

    def test_exact_phrase_exclusion_match(self):
        r = check_coverage(date(2024, 1, 1), date(2024, 3, 1), 12, ["water damage", "cosmetic scratches"],
                           "Customer reports Water Damage after a spill")
        self.assertEqual(r.matched_exclusion, "water damage")

    def test_bad_warranty_months(self):
        with self.assertRaises(ToolError):
            check_coverage(date(2024, 1, 1), date(2024, 3, 1), 0, [], "x")


class TestDocumentAndDatabaseTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        db.init()

    def test_parse_document_numbers_lines(self):
        doc = parse_document("CLM-2024-003/repair_estimate_003.txt")
        self.assertEqual(doc.kind, "diagnostic")
        serial = [l for l in doc.lines if "Serial Number" in l.text]
        self.assertEqual(serial[0].text.strip(), "Serial Number: REF-HC-99400-B")
        self.assertGreater(serial[0].n, 1)

    def test_unknown_document_is_an_error(self):
        with self.assertRaises(ToolError):
            parse_document("CLM-NOPE/nothing.txt")

    def test_search_policy_returns_the_claims_policy_with_lines(self):
        r = search_policy("CLM-2024-001", "What deductible applies to a repair claim?")
        self.assertTrue(r.passages)
        top = r.passages[0]
        self.assertEqual(top.doc_id, "policy:POL-TECH-CARE-01")
        self.assertTrue(top.section.endswith("Deductible"))
        self.assertTrue(any("$29.00" in l.text for l in top.lines))

    def test_claim_history_finds_the_serial_used_by_another_claimant(self):
        r = claim_history("REF-HC-99400-B", "David Sterling", "CLM-2024-003")
        self.assertEqual(r.same_serial_count, 1)
        self.assertEqual(r.past_claims[0].claimant, "Karen Whitfield")
        self.assertEqual(r.last_claim_days_ago, 189)          # 2024-02-03 -> incident 2024-08-10

    def test_claim_history_excludes_the_current_claim(self):
        r = claim_history("", "Elena Rostova", "CLM-2024-001")
        self.assertEqual(r.past_claims, [])


class TestServer(unittest.IsolatedAsyncioTestCase):

    async def test_exactly_five_tools(self):
        names = {t.name for t in await server.list_tools()}
        self.assertEqual(names, {"parse_document", "search_policy", "check_coverage", "compute_payout", "claim_history"})

    def test_mcp_route_needs_the_secret(self):
        with mock.patch.object(config, "A2A_SHARED_SECRET", "s3cret"), TestClient(api.app) as client:
            body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
            self.assertEqual(client.post("/mcp", json=body).status_code, 401)
            wrong = client.post("/mcp", json=body, headers={"X-ClaimSleuth-Secret": "nope"})
            self.assertEqual(wrong.status_code, 401)

    def test_mcp_is_closed_when_no_secret_is_configured(self):
        with mock.patch.object(config, "A2A_SHARED_SECRET", ""), TestClient(api.app) as client:
            response = client.post("/mcp", json={}, headers={"X-ClaimSleuth-Secret": ""})
            self.assertEqual(response.status_code, 401)
