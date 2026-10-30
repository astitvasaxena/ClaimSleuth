"""The evaluation code without model calls: the claim set, the plan, scoring, configuration choice, charts.

The chart test uses a small made-up CSV in a temporary folder (TEST DATA, never written to eval/results/)."""
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))


def load(name):
    spec = importlib.util.spec_from_file_location(name, EVAL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_eval_set, run_eval, charts = load("check_eval_set"), load("run_eval"), load("charts")
from claimsleuth import ratelimit  # noqa: E402


def outcome(decision, flagged=False, payout=None, status="completed"):
    report = None if decision is None else {
        "decision": decision, "fraud": {"flagged": flagged, "findings": []},
        "payout": {"amount": payout} if payout else None,
        "guardrail_hits": [{"kind": "injection", "where": "customer_statement", "detail": "x"}]}
    return {"run": {"id": "r1", "model": "m", "status": status, "error": None, "seconds": 10.0, "prompt_tokens": 100,
                    "completion_tokens": 10, "cost_usd": 0.001},
            "agents": [{"agent": "claim_manager", "seconds": 10.0, "prompt_tokens": 100, "completion_tokens": 10,
                        "llm_calls": 2, "tool_calls": 1, "errors": 0, "guardrail_hits": 1}],
            "report": report, "result": {}}


class TestClaimSet(unittest.TestCase):

    def test_the_claim_set_is_complete(self):
        self.assertEqual(check_eval_set.check(), [])


class TestPlanAndScoring(unittest.TestCase):

    def test_plan_sizes(self):
        self.assertEqual(len(run_eval.plan_a()), 24)
        self.assertEqual(len(run_eval.plan_b(("hierarchical", "direct"))), 20)
        self.assertEqual(len({run_eval.key(p) for p in run_eval.plan_a()}), 24)
        self.assertEqual({p["claim"] for p in run_eval.plan_a()}, {"E1", "E3", "E4", "E6"})

    def test_scoring(self):
        p = {"experiment": "A", "claim": "E5", "persona": "honest", "pattern": "sequential", "prompt_style": "direct"}
        self.assertTrue(run_eval.to_row(p, outcome("approve", payout="760.00"), 0, "t")["correct"])
        self.assertFalse(run_eval.to_row(p, outcome("approve", payout="800.00"), 0, "t")["correct"])   # wrong amount
        p4 = {**p, "claim": "E4"}
        self.assertTrue(run_eval.to_row(p4, outcome("request_evidence", flagged=True), 0, "t")["correct"])
        self.assertFalse(run_eval.to_row(p4, outcome("request_evidence", flagged=False), 0, "t")["correct"])  # fraud missed
        failed = run_eval.to_row(p4, outcome(None, status="failed"), 0, "t")
        self.assertFalse(failed["correct"])
        self.assertEqual(run_eval.to_row(p4, outcome("reject", flagged=True), 0, "t")["hits_injection"], 1)

    def test_best_configuration_prefers_accuracy_then_cost(self):
        rows = []
        for pattern in run_eval.PATTERNS:
            for style in run_eval.STYLES:
                for i, claim in enumerate(run_eval.EVAL_CLAIMS):
                    good = (pattern, style) in (("parallel", "direct"), ("hierarchical", "react")) or i < 2
                    cost = "0.002" if pattern == "parallel" else "0.001"
                    rows.append({"experiment": "A", "pattern": pattern, "prompt_style": style, "correct": str(good),
                                 "cost_usd": cost, "seconds": "10"})
        self.assertEqual(run_eval.best_config(rows), ("hierarchical", "react"))   # tied accuracy, lower cost

    def test_rate_limit_detection(self):
        daily = ("_ResourceExhaustedError: 429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generativelanguage."
                 "googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash, description: "
                 "Generate content requests per day limit exceeded on Free Tier.")
        minute = "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: requests per minute, limit: 15"
        self.assertEqual(ratelimit.classify(daily), ratelimit.DAILY)
        self.assertEqual(ratelimit.classify(minute), ratelimit.MINUTE)
        self.assertIsNone(ratelimit.classify("TimeoutError: run stopped"))
        self.assertTrue(run_eval.is_daily_quota(f"{ratelimit.DAILY}: {ratelimit.quota_detail(daily)}"))
        self.assertFalse(run_eval.is_daily_quota(f"{ratelimit.MINUTE}: x"))
        self.assertTrue(run_eval.is_rate_limit(f"{ratelimit.MINUTE}: x"))
        self.assertIn("per day", ratelimit.quota_detail(daily))


class TestCharts(unittest.TestCase):

    def test_charts_and_summary_from_a_runs_file(self):
        out = Path(tempfile.mkdtemp(prefix="claimsleuth_test_charts_"))
        rows = []
        for i, p in enumerate(run_eval.plan_a()[:12] + run_eval.plan_b(("sequential", "direct"))[:10]):
            row = {c: "" for c in run_eval.COLUMNS}
            row.update(p, run_id=f"test{i}", model="test-model", target="test", status="completed", expected="approve",
                       decision="approve", correct=str(i % 3 != 0), seconds=str(20 + i), rate_limit_wait_s="0",
                       prompt_tokens="1000", completion_tokens="100", cost_usd="0.0005", llm_calls="12", tool_calls="9",
                       errors="0", guardrail_hits="1", hits_injection="1", hits_abuse="0", finished_at="2026-10-30T12:00:00",
                       claim_manager_seconds=str(10 + i), evidence_agent_seconds="3")
            rows.append(row)
        with (out / "runs.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=run_eval.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        written = charts.write_charts(charts.load(out / "runs.csv"), out)
        self.assertIn("boxplot_seconds_A_by_configuration.png", written)
        self.assertIn("boxplot_cost_usd_B_by_persona.png", written)
        self.assertIn("heatmap_agents_by_configuration.png", written)
        for name in written:
            self.assertTrue((out / name).read_bytes().startswith(b"\x89PNG"))
        summary = charts.write_summary(charts.load(out / "runs.csv"), written, out, 0).read_text(encoding="utf-8")
        self.assertIn("Experiment A", summary)
        self.assertIn("injection caught", summary)
