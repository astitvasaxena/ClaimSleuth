"""
Technical Console (dashboard UI only): layout rules, readable JSON, display-only redaction
and badges that reflect the run's actual report.

The display helpers between the <console-helpers> markers in index.html are pure
JavaScript; they are executed here with Node (skipped if Node is not installed) on the
real Blackboard records of an ADK investigation.
"""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML = (PROJECT_ROOT / "src" / "ui" / "templates" / "index.html").read_text(encoding="utf-8")
NODE = shutil.which("node")


def helpers_js() -> str:
    match = re.search(r"// <console-helpers>[^\n]*\n(.*?)// </console-helpers>", HTML, re.DOTALL)
    assert match, "console helpers block not found"
    return match.group(1)


def run_node(program: str, payload) -> dict:
    script = "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n" + helpers_js() + "\n" + program
    res = subprocess.run([NODE, "-e", script], input=json.dumps(payload), capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout)


class TestConsoleLayoutRules(unittest.TestCase):

    def test_columns_can_shrink_and_values_wrap(self):
        self.assertRegex(HTML, r"\.tech-split \{[^}]*grid-template-columns: minmax\(0, 1fr\) minmax\(0, 1fr\)")
        value_rule = re.search(r"\.bb-fact-value, \.bb-msg-payload \{([^}]*)\}", HTML).group(1)
        for prop in ("white-space: pre-wrap", "overflow-wrap: anywhere", "max-width: 100%"):
            self.assertIn(prop, value_rule)
        self.assertRegex(HTML, r"\.console-list \{[^}]*overflow-x: hidden")
        self.assertIn('id="blackboardFactsContainer" class="console-list"', HTML)
        self.assertIn('id="a2aMessagesContainer" class="console-list"', HTML)

    def test_values_are_inserted_as_text_not_markup(self):
        render = HTML[HTML.index("function renderBlackboard"):HTML.index("function renderMonitoring")]
        self.assertNotIn("innerHTML = `", render.split("No facts in working memory")[1].split("// A2A Messages")[0])
        self.assertIn("pre.textContent = parts.text", render)
        self.assertIn("payload.textContent = JSON.stringify(redactForConsole(m.payload))", render)

    def test_badges_are_no_longer_hardcoded(self):
        self.assertNotIn(">MCP (in-process)<", HTML)
        self.assertNotIn(">CrewAI placeholder<", HTML)
        self.assertIn('id="mcpBadge"', HTML)
        self.assertIn('id="crewBadge"', HTML)
        self.assertIn("renderConsoleBadges(data.report || null, engine);", HTML)


@unittest.skipUnless(NODE, "needs Node.js to execute the dashboard's display helpers")
class TestConsoleHelpersWithRealRecords(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import src.ui.app as app_module
        client = TestClient(app_module.app)
        data = client.post("/api/adk/investigate/scenario_3_conflict?tools=local").json()
        cls.facts = {k: v["value"] for k, v in data["blackboard"]["facts"].items()}

    def test_long_records_are_formatted_and_preserved(self):
        out = run_node("""
            const result = {};
            for (const [k, v] of Object.entries(input)) {
                const before = JSON.stringify(v);
                const parts = consoleValueParts(v);
                result[k] = { parts, redacted: redactForConsole(v), unchanged: JSON.stringify(v) === before };
            }
            process.stdout.write(JSON.stringify(result));
        """, self.facts)
        for key in ("bb:claim", "bb:tool_calls", "bb:agent_timings", "bb:context_stats"):
            with self.subTest(key=key):
                self.assertIn(key, out)
                record = out[key]
                parts = record["parts"]
                self.assertEqual(parts["kind"], "json")
                self.assertIn("\n  ", parts["text"])                        # indented, multi-line
                self.assertEqual(json.loads(parts["text"]), record["redacted"])  # full value, nothing truncated
                self.assertTrue(record["unchanged"])                        # display copy only
                self.assertLess(len(parts["summary"]), 140)
        self.assertEqual(len(json.loads(out["bb:tool_calls"]["parts"]["text"])), len(self.facts["bb:tool_calls"]))
        self.assertEqual(len(json.loads(out["bb:agent_timings"]["parts"]["text"])), len(self.facts["bb:agent_timings"]))
        self.assertFalse(out["bb:tool_calls"]["parts"]["open"])               # long records start collapsed

    def test_claimant_details_are_redacted_in_the_console_only(self):
        claim = self.facts["bb:claim"]
        self.assertTrue(claim.get("claimant_name"))                         # the real data still has it
        out = run_node("process.stdout.write(JSON.stringify(consoleValueParts(input)));", claim)
        shown = json.loads(out["text"])
        self.assertEqual(shown["claimant_name"], "[redacted in console]")
        self.assertEqual(shown["claimant_id"], "[redacted in console]")
        self.assertEqual(shown["claim_id"], claim["claim_id"])               # identifiers needed for diagnosis stay
        self.assertNotIn(claim["claimant_name"], out["text"])
        text = run_node("process.stdout.write(JSON.stringify(redactForConsole(input)));",
                        {"line": "Customer: Elena Rostova\nSerial Number: APPL-1", "card": "Payment Method: VISA ending in 4402"})
        self.assertEqual(text["line"], "Customer: [redacted]\nSerial Number: APPL-1")
        self.assertEqual(text["card"], "Payment Method: [redacted]")

    def test_badges_follow_the_actual_report(self):
        reports = {   # report shapes as returned by /api/adk/investigate (fields used by the badges only)
            "mcp_no_crew": {"tool_backend": "mcp_stdio", "forensics_mode": "none"},
            "local": {"tool_backend": "local", "forensics_mode": "none"},
            "http": {"tool_backend": "mcp_http", "forensics_mode": "none"},
            "crew_llm": {"tool_backend": "mcp_stdio", "forensics_mode": "crewai_a2a@http://127.0.0.1:8101",
                         "a2a_call": {"ok": True}, "forensics": {"llm_review": {
                             "status": "COMPLETED", "llm_provider": "openrouter (OpenAI-compatible endpoint)",
                             "llm_model": "nvidia/nemotron-3.5-super-vl:free"}}},
            "crew_no_llm": {"tool_backend": "mcp_stdio", "forensics_mode": "crewai_a2a@x", "a2a_call": {"ok": True},
                            "forensics": {"llm_review": {"status": "LLM_NOT_CONFIGURED"}}},
            "crew_llm_failed": {"tool_backend": "mcp_stdio", "forensics_mode": "crewai_a2a@x", "a2a_call": {"ok": True},
                                "forensics": {"llm_review": {"status": "LLM_ERROR"}}},
            "crew_down": {"tool_backend": "mcp_stdio", "forensics_mode": "crewai_a2a@x", "a2a_call": {"ok": False},
                          "forensics": None},
        }
        out = run_node("""
            const r = {};
            for (const [k, v] of Object.entries(input)) r[k] = consoleBadgeText(v, 'adk_mcp_crewai');
            r.legacy = consoleBadgeText(null, 'legacy');
            r.first_load = consoleBadgeText(null, 'adk_local');
            process.stdout.write(JSON.stringify(r));
        """, reports)
        self.assertEqual(out["mcp_no_crew"], {"mcp": "MCP: stdio transport", "crew": "CrewAI: not used", "crewTitle": ""})
        self.assertEqual(out["local"]["mcp"], "MCP: not used (local tools)")
        self.assertEqual(out["http"]["mcp"], "MCP: streamable HTTP transport")
        self.assertEqual(out["crew_llm"]["crew"], "CrewAI Crew + LLM via A2A")
        self.assertIn("nvidia/nemotron-3.5-super-vl:free", out["crew_llm"]["crewTitle"])
        self.assertEqual(out["crew_no_llm"]["crew"], "CrewAI Flow via A2A (no LLM)")
        self.assertEqual(out["crew_llm_failed"]["crew"], "CrewAI via A2A: LLM failed (LLM_ERROR)")
        self.assertEqual(out["crew_down"]["crew"], "CrewAI via A2A: failed")
        self.assertEqual(out["legacy"]["mcp"], "MCP: in-process (original loop)")
        self.assertEqual(out["first_load"]["crew"], "CrewAI: no run yet")


if __name__ == "__main__":
    unittest.main()
