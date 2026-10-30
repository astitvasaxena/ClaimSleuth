"""API: token required, upload limits and file checks, persistence across a restart, the review rules."""
import asyncio
import io
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from claimsleuth import config, db
from claimsleuth.api import app as api
from claimsleuth.models import FinalReport

TOKEN = "test-adjuster-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
FORM = {"claimant": "Ana Lima", "device": "Pixel 9", "incident_date": "2024-05-02", "purchase_date": "2024-01-10",
        "statement": "The screen stopped responding.", "policy_id": "POL-TECH-CARE-01"}


def pdf_bytes() -> bytes:
    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buffer)
    return buffer.getvalue()


class ApiTest(unittest.TestCase):

    def setUp(self):
        for p in (mock.patch.object(config, "ADJUSTER_TOKEN", TOKEN), mock.patch.object(config, "GOOGLE_API_KEY", "k")):
            p.start()
            self.addCleanup(p.stop)

    def client(self):
        return TestClient(api.app)

    def upload(self, client, files, **form):
        return client.post("/api/claims", data={**FORM, **form}, files=files, headers=AUTH)


class TestAuth(ApiTest):

    def test_token_required(self):
        with self.client() as c:
            self.assertEqual(c.get("/api/claims").status_code, 401)
            self.assertEqual(c.get("/api/claims", headers={"Authorization": "Bearer wrong"}).status_code, 401)
            self.assertEqual(c.get("/api/claims", headers=AUTH).status_code, 200)
            self.assertEqual(c.get("/healthz").status_code, 200)
            self.assertEqual(c.get("/").status_code, 200)                # the page shell holds no data

    def test_api_closed_without_a_configured_token(self):
        with mock.patch.object(config, "ADJUSTER_TOKEN", ""), self.client() as c:
            self.assertEqual(c.get("/api/claims", headers={"Authorization": "Bearer "}).status_code, 503)


class TestUploads(ApiTest):

    def test_upload_saves_claim_and_documents_and_survives_a_restart(self):
        files = [("files", ("invoice.txt", b"Invoice\nPurchase Date: 2024-01-10\nTotal: $120.00\n", "text/plain")),
                 ("files", ("report.pdf", pdf_bytes(), "application/pdf"))]
        with self.client() as c:
            response = self.upload(c, files)
            self.assertEqual(response.status_code, 200, response.text)
            claim_id = response.json()["claim_id"]
        with self.client() as c:                                         # a new app lifespan: same database
            claim = c.get(f"/api/claims/{claim_id}", headers=AUTH).json()
            self.assertEqual(claim["source"], "upload")
            docs = {d["filename"]: d for d in claim["documents"]}
            self.assertEqual(docs["invoice.txt"]["kind"], "invoice")
            self.assertFalse(docs["report.pdf"]["has_text"])               # blank page: no text layer
            lines = c.get(f"/api/documents/{claim_id}/invoice.txt", headers=AUTH).json()["lines"]
            self.assertEqual(lines[1]["text"], "Purchase Date: 2024-01-10")

    def test_file_type_is_checked_by_content(self):
        with self.client() as c:
            fake_pdf = [("files", ("invoice.pdf", b"MZ\x90\x00 not a pdf", "application/pdf"))]
            self.assertEqual(self.upload(c, fake_pdf).status_code, 415)
            binary_txt = [("files", ("invoice.txt", b"abc\x00def", "text/plain"))]
            self.assertEqual(self.upload(c, binary_txt).status_code, 415)
            exe = [("files", ("invoice.exe", b"MZ", "application/octet-stream"))]
            self.assertEqual(self.upload(c, exe).status_code, 415)

    def test_limits_and_required_fields(self):
        with self.client() as c:
            big = [("files", ("big.txt", b"a" * (5 * 1024 * 1024 + 1), "text/plain"))]
            self.assertEqual(self.upload(c, big).status_code, 413)
            many = [("files", (f"f{i}.txt", b"x", "text/plain")) for i in range(11)]
            self.assertEqual(self.upload(c, many).status_code, 413)
            one = [("files", ("a.txt", b"x", "text/plain"))]
            self.assertEqual(self.upload(c, one, policy_id="").status_code, 422)          # no policy at all
            self.assertEqual(self.upload(c, one, incident_date="02/05/2024").status_code, 422)


class TestReview(ApiTest):

    def add_report(self, claim_id, review_level):
        report = FinalReport(claim_id=claim_id, decision="approve", reasons=["ok"], evidence=[], review=review_level,
                             pattern="hierarchical", prompt_style="direct")
        with db.connect() as conn:
            conn.execute("INSERT INTO reports (claim_id, run_id, report_json, report_md, created_at) VALUES (?, ?, ?, ?, ?)",
                         (claim_id, f"r-{review_level}", report.model_dump_json(), "#", db.now()))

    def test_full_review_needs_a_note(self):
        with self.client() as c:
            self.add_report("CLM-2024-001", "full_review")
            body = {"action": "approve", "note": ""}
            self.assertEqual(c.post("/api/claims/CLM-2024-001/review", json=body, headers=AUTH).status_code, 422)
            body["note"] = "Checked the invoice and the repair estimate."
            self.assertEqual(c.post("/api/claims/CLM-2024-001/review", json=body, headers=AUTH).status_code, 200)
            self.assertEqual(c.get("/api/claims/CLM-2024-001/report", headers=AUTH).json()["review"]["action"], "approve")

    def test_one_click_review_and_bad_action(self):
        with self.client() as c:
            self.add_report("CLM-2024-003", "one_click")
            self.assertEqual(c.post("/api/claims/CLM-2024-003/review", json={"action": "approve"}, headers=AUTH).status_code, 200)
            self.assertEqual(c.post("/api/claims/CLM-2024-003/review", json={"action": "pay"}, headers=AUTH).status_code, 422)

    def test_no_report_no_review(self):
        with self.client() as c:
            self.assertEqual(c.post("/api/claims/CLM-2024-002/review", json={"action": "approve"}, headers=AUTH).status_code, 409)


class TestRuns(ApiTest):

    def test_second_run_on_the_same_claim_is_refused(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(3)
            return {"status": "completed"}
        with self.client() as c, mock.patch("claimsleuth.runner.run_claim", side_effect=slow):
            self.assertEqual(c.post("/api/claims/CLM-2024-002/investigate", headers=AUTH).status_code, 200)
            self.assertEqual(c.post("/api/claims/CLM-2024-002/investigate", headers=AUTH).status_code, 409)
            self.assertEqual(c.post("/api/claims/CLM-2024-002/investigate?pattern=star", headers=AUTH).status_code, 422)
