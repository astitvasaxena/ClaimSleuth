"""Data models round-trip through JSON; the review rule; the database schema and seed data."""
import unittest
from decimal import Decimal

from pydantic import ValidationError

from claimsleuth import db
from claimsleuth.models import (ClaimRecord, Coverage, DocumentRecord, Fact, Facts, FinalReport, Fraud,
                                FraudFinding, GuardrailHit, ManagerSubmission, Payout, Quote, review_level,
                                to_markdown)

QUOTE = Quote(doc_id="CLM-2024-001/invoice_001.txt", line=6, text="Purchase Date: 2023-11-15")
PAYOUT = Payout(amount=Decimal("251.00"), breakdown={"repair_cost": Decimal("280.00"),
                                                     "deductible": Decimal("29.00"), "net": Decimal("251.00")})
FRAUD = Fraud(findings=[FraudFinding(kind="serial_mismatch", severity="high", detail="Serials differ",
                                     quotes=[QUOTE])], flagged=True)
COVERAGE = Coverage(covered=True, within_window=True, exclusion=None, clause=QUOTE, days_since_purchase=310)


def report(**overrides):
    values = dict(claim_id="CLM-2024-001", decision="approve", payout=PAYOUT, reasons=["Covered repair"],
                  evidence=[QUOTE], coverage=COVERAGE, fraud=Fraud(findings=[], flagged=False),
                  guardrail_hits=[GuardrailHit(kind="injection", where="statement", detail="ignore your rules")],
                  review="one_click", pattern="hierarchical", prompt_style="direct")
    values.update(overrides)
    return FinalReport(**values)


class TestRoundTrip(unittest.TestCase):

    def test_every_model_round_trips_through_json(self):
        samples = [
            QUOTE, Fact(name="serial", value="SN-1", quote=QUOTE), Facts(facts=[], missing=["invoice"]),
            COVERAGE, PAYOUT, FRAUD, GuardrailHit(kind="bad_quote", where="Evidence", detail="not in document"),
            ManagerSubmission(decision="reject", reasons=["Expired"], evidence=[QUOTE]), report(),
            ClaimRecord(id="CLM-1", source="upload", claimant="A", device="Phone", incident_date="2024-01-02",
                        statement="Broke"),
            DocumentRecord(id="CLM-1/a.pdf", claim_id="CLM-1", kind="invoice", filename="a.pdf",
                           path="uploads/CLM-1/a.pdf", sha256="0" * 64),
        ]
        for model in samples:
            with self.subTest(model=type(model).__name__):
                self.assertEqual(type(model).model_validate_json(model.model_dump_json()), model)

    def test_money_stays_exact(self):
        restored = Payout.model_validate_json(PAYOUT.model_dump_json())
        self.assertEqual(restored.amount, Decimal("251.00"))

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValidationError):
            Quote(doc_id="x", line=0, text="a")
        with self.assertRaises(ValidationError):
            Fact(name="colour", value="red", quote=QUOTE)
        with self.assertRaises(ValidationError):
            ManagerSubmission(decision="pay_now", reasons=["x"])
        with self.assertRaises(ValidationError):
            ManagerSubmission(decision="approve", reasons=[])

    def test_markdown_has_the_decision_and_quotes(self):
        md = to_markdown(report())
        self.assertIn("**Decision:** approve", md)
        self.assertIn("Purchase Date: 2023-11-15", md)


class TestReviewRule(unittest.TestCase):

    def test_small_clean_approval_is_one_click(self):
        small = Payout(amount=Decimal("150.00"), breakdown={})
        self.assertEqual(review_level("approve", small, Fraud(findings=[], flagged=False)), "one_click")

    def test_payout_over_150_needs_full_review(self):
        self.assertEqual(review_level("approve", PAYOUT, None), "full_review")

    def test_rejection_needs_full_review(self):
        self.assertEqual(review_level("reject", None, None), "full_review")

    def test_fraud_flag_needs_full_review(self):
        small = Payout(amount=Decimal("20.00"), breakdown={})
        self.assertEqual(review_level("approve", small, FRAUD), "full_review")


class TestDatabase(unittest.TestCase):

    def test_init_creates_tables_and_loads_demos_and_seed(self):
        result = db.init()
        with db.connect() as conn:
            demos = conn.execute("SELECT COUNT(*) FROM claims WHERE source = 'demo'").fetchone()[0]
            demo_docs = conn.execute("SELECT COUNT(*) FROM documents d JOIN claims c ON c.id = d.claim_id"
                                     " WHERE c.source = 'demo'").fetchone()[0]
        self.assertEqual((demos, demo_docs), (3, 6))
        self.assertGreaterEqual(result["counts"]["past_claims"], 1)
        again = db.init()                                   # idempotent
        self.assertEqual(again["demos_added"], 0)
        self.assertEqual(again["counts"], result["counts"])
        claim = db.get_claim("CLM-2024-003")
        self.assertEqual(claim["source"], "demo")
        docs = db.get_documents("CLM-2024-003")
        self.assertEqual({d["kind"] for d in docs}, {"invoice", "diagnostic"})
        for d in docs:
            self.assertTrue(db.resolve_document_path(d["path"]).is_file())
