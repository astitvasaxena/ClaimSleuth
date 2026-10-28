"""
Evidence Dossier & Human Review, and the claim summary row (dashboard UI).

- Pure display helpers (<dossier-helpers> in index.html) run in Node.
- The server's `allowed_decisions` must match what /api/adjudicate actually accepts.
- A real headless-Chrome run of the dashboard (tests/fixtures/dashboard_browser_check.mjs)
  against a dedicated server process: documents, extracted facts and verbatim quotes,
  review controls, loading/error states, stale-response handling, and layout at several
  widths. Uses only the original loop and ADK with local tools (no MCP, CrewAI or model).
Node / Chrome-based parts are skipped with a reason if those programs are missing.
"""
import importlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "src" / "ui" / "templates" / "index.html").read_text(encoding="utf-8")
CLAIMS = ROOT / "data" / "sample_claims"
NODE = shutil.which("node")
CHROME = next((p for p in (os.environ.get("CLAIMSLEUTH_TEST_CHROME"),
                           r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                           r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                           shutil.which("google-chrome"), shutil.which("chromium"))
               if p and Path(p).exists()), None)
DECISIONS = ("APPROVED", "REJECTED", "INFO_REQUESTED")


def node_eval(program: str, payload):
    block = re.search(r"// <dossier-helpers>[^\n]*\n(.*?)// </dossier-helpers>", HTML, re.DOTALL).group(1)
    script = "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n" + block + "\n" + program
    res = subprocess.run([NODE, "-e", script], input=json.dumps(payload), capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout)


def client():
    os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
    import configs.settings
    importlib.reload(configs.settings)
    import src.ui.app as app_module
    return TestClient(importlib.reload(app_module).app)


@unittest.skipUnless(NODE, "needs Node.js to execute the dashboard's display helpers")
class TestDossierHelpers(unittest.TestCase):

    def test_locate_quote_reports_a_line_only_for_a_verbatim_match(self):
        doc = (CLAIMS / "scenario_1_valid" / "invoice_001.txt").read_text(encoding="utf-8")
        out = node_eval("""process.stdout.write(JSON.stringify({
            found: locateQuote(input.doc, 'Purchase Date: 2024-03-15'),
            serial: locateQuote(input.doc, 'Serial Number: APPL-IPH15-998821'),
            altered: locateQuote(input.doc, 'Purchase Date: 2024-03-16'),
            empty: locateQuote(input.doc, ''), noDoc: locateQuote(undefined, 'x') }));""", {"doc": doc})
        lines = doc.splitlines()
        self.assertEqual(lines[out["found"] - 1], "Purchase Date: 2024-03-15")
        self.assertIn("Serial Number: APPL-IPH15-998821", lines[out["serial"] - 1])
        self.assertIsNone(out["altered"])
        self.assertIsNone(out["empty"])
        self.assertIsNone(out["noDoc"])

    def test_review_summary_follows_server_fields(self):
        cases = {
            "proposed": ["VERIFIED_PENDING_APPROVAL", {"payout_status": "PROPOSED_PENDING_APPROVAL", "proposed_payout": 251},
                         ["APPROVED", "REJECTED", "INFO_REQUESTED"], True],
            "authorized": ["CLOSED_SETTLED", {"payout_status": "AUTHORIZED", "authorized_payout": 251}, [], False],
            "conflict": ["CONFLICT_ESCALATED", {"payout_status": "NOT_PAYABLE"}, ["REJECTED", "INFO_REQUESTED"], True],
            "unknown": ["VERIFIED_PENDING_APPROVAL", {"payout_status": "PROPOSED_PENDING_APPROVAL", "proposed_payout": 1}, None, True],
            "spoofed_order": ["NEEDS_EVIDENCE", {"payout_status": "NOT_PAYABLE"}, ["INFO_REQUESTED", "BOGUS", "REJECTED"], True],
        }
        out = node_eval("""const r = {}; for (const [k, a] of Object.entries(input)) r[k] = reviewSummary(...a);
            process.stdout.write(JSON.stringify(r));""", cases)
        self.assertEqual(out["proposed"]["decisions"], ["APPROVED", "REJECTED", "INFO_REQUESTED"])
        self.assertIn("Proposed settlement: $251.00", out["proposed"]["payoutLine"])
        self.assertIn("not paid unless a human approves", out["proposed"]["payoutLine"])
        self.assertIn("Authorized payout: $251.00", out["authorized"]["payoutLine"])
        self.assertEqual(out["authorized"]["decisions"], [])
        self.assertEqual(out["conflict"]["decisions"], ["REJECTED", "INFO_REQUESTED"])
        self.assertIn("Approval is not possible", out["conflict"]["reviewLine"])
        self.assertEqual(out["unknown"]["decisions"], [])               # no server list -> no controls
        self.assertEqual(out["spoofed_order"]["decisions"], ["REJECTED", "INFO_REQUESTED"])


class TestAllowedDecisionsMatchTheServerGuard(unittest.TestCase):
    """allowed_decisions is informational; the adjudication endpoint stays the authority."""

    def test_reported_decisions_are_exactly_the_accepted_ones(self):
        c = client()
        expected = {"scenario_1_valid": ["APPROVED", "REJECTED", "INFO_REQUESTED"],
                    "scenario_2_expired": ["REJECTED", "INFO_REQUESTED"],
                    "scenario_3_conflict": ["REJECTED", "INFO_REQUESTED"]}
        claim_ids = {"scenario_1_valid": "CLM-2024-001", "scenario_2_expired": "CLM-2024-002",
                     "scenario_3_conflict": "CLM-2024-003"}
        for engine, path in (("legacy", "/api/investigate/{}"), ("adk", "/api/adk/investigate/{}?tools=local")):
            for scenario, allowed in expected.items():
                for decision in DECISIONS:
                    with self.subTest(engine=engine, scenario=scenario, decision=decision):
                        data = c.post(path.format(scenario)).json()          # fresh state for every decision
                        self.assertEqual(data["allowed_decisions"], allowed)
                        res = c.post(f"/api/adjudicate/{claim_ids[scenario]}", json={"decision": decision})
                        self.assertEqual(res.status_code, 200 if decision in allowed else 409)
                        if res.status_code == 200:
                            self.assertEqual(res.json()["allowed_decisions"],
                                             [] if decision != "INFO_REQUESTED" else ["REJECTED", "INFO_REQUESTED"])


@unittest.skipUnless(NODE and CHROME, "needs Node.js and Chrome/Edge for the browser check")
class TestDossierInTheBrowser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cls.base = f"http://127.0.0.1:{port}"
        env = {**os.environ}
        env.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        cls.server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.ui.app:app", "--host", "127.0.0.1",
                                       "--port", str(port), "--log-level", "warning"], cwd=str(ROOT), env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if httpx.get(cls.base + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        res = subprocess.run([NODE, str(ROOT / "tests" / "fixtures" / "dashboard_browser_check.mjs"), cls.base, CHROME],
                             capture_output=True, text=True, encoding="utf-8", timeout=400)
        cls.out = json.loads(res.stdout.strip().splitlines()[-1])

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            cls.server.kill()

    def test_no_script_errors(self):
        self.assertIsNone(self.out.get("fatal"))
        self.assertEqual(self.out["errors"], [])

    def test_documents_show_the_selected_files_full_text(self):
        folder = CLAIMS / "scenario_1_valid"
        expected = {p.name: p.read_text(encoding="utf-8") for p in folder.iterdir() if p.suffix == ".txt"}
        self.assertEqual(sorted(self.out["initial"]["tabs"]), sorted(expected))
        self.assertEqual(self.out["docSwitch"], expected)                   # each tab shows its own file, in full

    def test_states_before_and_during_an_investigation(self):
        initial, loading = self.out["initial"], self.out["loading"]
        self.assertIn("will appear here after an investigation runs", initial["evidenceMessage"])
        self.assertEqual((initial["facts"], initial["visibleDecisions"], initial["notesVisible"]), ([], [], False))
        self.assertIn("in progress", loading["evidenceMessage"])
        self.assertIn("in progress", loading["outcome"])
        self.assertEqual(loading["visibleDecisions"], [])
        self.assertFalse(loading["runButtonEnabled"])

    def test_facts_show_verbatim_quotes_with_their_source_line(self):
        for key, scenario in (("s1", "scenario_1_valid"), ("s3", "scenario_3_conflict"), ("s2", "scenario_2_expired")):
            facts = self.out[key]["facts"]
            self.assertTrue(facts, key)
            for fact in facts:
                with self.subTest(case=key, field=fact["field"]):
                    match = re.match(r"Source: (\S+) · line (\d+)", fact["meta"])
                    self.assertIsNotNone(match, fact["meta"])
                    lines = (CLAIMS / scenario / match.group(1)).read_text(encoding="utf-8").splitlines()
                    quote = fact["quote"][1:-1]                              # strip the display quotes
                    self.assertIn(quote, lines[int(match.group(2)) - 1])     # verbatim, on the stated line

    def test_review_controls_match_the_claim_state(self):
        s1, approved, s2, s3 = self.out["s1"], self.out["s1AfterApprove"], self.out["s2"], self.out["s3"]
        self.assertEqual(s1["visibleDecisions"], ["APPROVED", "REJECTED", "INFO_REQUESTED"])
        self.assertIn("Proposed settlement: $251.00", s1["outcome"])
        self.assertIn("Human review required", s1["outcome"])
        self.assertEqual(approved["visibleDecisions"], [])
        self.assertIn("Authorized payout: $251.00", approved["outcome"])
        self.assertEqual(approved["pill"], "CLOSED SETTLED")
        for case in (s2, s3):
            self.assertEqual(case["visibleDecisions"], ["REJECTED", "INFO_REQUESTED"])
            self.assertIn("Not payable", case["outcome"])

    def test_forcing_approval_of_a_conflicted_claim_changes_nothing(self):
        forced = self.out["s3ForcedApprove"]
        self.assertTrue(any("HTTP 409" in d for d in forced["dialogs"]))
        self.assertEqual(forced["pill"], "CONFLICT ESCALATED")
        self.assertEqual(forced["payout"], "Not payable")
        self.assertEqual(forced["visibleDecisions"], ["REJECTED", "INFO_REQUESTED"])

    def test_stale_responses_never_overwrite_the_current_scenario(self):
        for key in ("staleRun", "staleDocs"):
            with self.subTest(case=key):
                snap = self.out[key]
                self.assertEqual(snap["claimId"], "CLM-2024-001")
                self.assertEqual(snap["tabs"], ["invoice_001.txt", "repair_estimate_001.txt"])
                self.assertEqual(snap["facts"], [])
                self.assertEqual(snap["visibleDecisions"], [])
                self.assertEqual(snap["pill"], "INITIALIZED")
                self.assertFalse(snap["resultsBannerVisible"])
                self.assertTrue(snap["runButtonEnabled"])

    def test_failed_investigation_shows_an_error_not_old_results(self):
        failed = self.out["failed"]
        self.assertTrue(failed["evidenceIsError"])
        self.assertIn("HTTP 500", failed["evidenceMessage"])
        self.assertEqual((failed["facts"], failed["visibleDecisions"]), ([], []))
        self.assertFalse(failed["resultsBannerVisible"])
        self.assertEqual(failed["pill"], "INVESTIGATION FAILED")

    def test_summary_row_and_dossier_stay_inside_their_containers(self):
        layout = self.out["layout"]
        for width in ("1920", "1440", "1366", "1280", "1024", "768", "420"):
            with self.subTest(width=width):
                m = layout[width]
                self.assertEqual(m["summaryCard"]["n"], 0)
                self.assertEqual(m["dossierPanel"]["n"], 0)
                self.assertEqual(m["chipOverlaps"], 0)
                self.assertFalse(m["investigationClipped"])
                self.assertNotEqual(m["investigationEllipsis"], "ellipsis")
                self.assertFalse(m["docHorizontalOverflow"])
                self.assertFalse(m["evidenceHorizontalOverflow"])
        for width in ("1920", "1440", "1366", "1280", "1024", "768"):     # laptop and desktop: nothing past the window
            self.assertEqual(layout[width]["beyondViewport"], 0, width)
        self.assertEqual(layout["1366"]["summaryColumns"], 3)
        self.assertEqual(layout["768"]["summaryColumns"], 2)
        self.assertEqual(layout["420"]["summaryColumns"], 1)


if __name__ == "__main__":
    unittest.main()
