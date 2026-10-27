"""
C-10 benchmark: reproducible labelled dataset, held-out discipline, honest metrics, plots.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy

from src.eval.dataset import dataset_digest, generate
from src.eval.svgplot import box_stats, boxplot_svg, heatmap_svg, quantile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH = PROJECT_ROOT / "data" / "benchmark"


class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))

    def test_committed_dataset_matches_generator_and_manifest(self):
        self.assertEqual(dataset_digest(BENCH), self.manifest["dataset_sha256"])
        with tempfile.TemporaryDirectory() as tmp:
            regenerated = generate(Path(tmp))
            self.assertEqual(regenerated["dataset_sha256"], self.manifest["dataset_sha256"])

    def test_size_and_splits(self):
        self.assertEqual(self.manifest["counts"], {"dev": 48, "val": 48, "test": 48})
        self.assertEqual(self.manifest["held_out_split"], "test")
        bases = defaultdict(set)
        for split, ids in self.manifest["splits"].items():
            for cid in ids:
                label = json.loads((BENCH / "labels" / f"{cid}.json").read_text(encoding="utf-8"))
                self.assertEqual(label["split"], split)
                bases[label["base_id"]].add(split)
        self.assertTrue(all(len(s) == 1 for s in bases.values()))   # no base case crosses splits

    def test_labels_are_not_visible_to_the_pipeline(self):
        for folder in (BENCH / "claims").iterdir():
            names = {p.name for p in folder.iterdir()}
            self.assertNotIn("label.json", names)
            claim = json.loads((folder / "claim.json").read_text(encoding="utf-8"))
            self.assertFalse({"expected_status", "expected_payout", "case_type"} & set(claim))

    def test_every_persona_and_case_type_in_every_split(self):
        for split, ids in self.manifest["splits"].items():
            labels = [json.loads((BENCH / "labels" / f"{c}.json").read_text(encoding="utf-8")) for c in ids]
            self.assertEqual({l["persona"] for l in labels}, set(self.manifest["personas"]))
            self.assertEqual({l["case_type"] for l in labels}, set(self.manifest["case_types"]))


class TestPlots(unittest.TestCase):

    def test_quantiles_match_numpy(self):
        values = sorted([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.8, 9.7, 3.2])
        for q in (0.25, 0.5, 0.75, 0.9):
            self.assertAlmostEqual(quantile(values, q), float(numpy.quantile(values, q)))

    def test_box_stats_outliers(self):
        stats = box_stats([10, 11, 12, 13, 14, 100])
        self.assertEqual(stats["outliers"], [100])
        self.assertEqual(stats["whisker_high"], 14)

    def test_svg_output(self):
        svg = boxplot_svg({"a": [1, 2, 3], "b": [2, 3, 4]}, "t", "ms")
        self.assertTrue(svg.startswith("<svg") and svg.rstrip().endswith("</svg>"))
        heat = heatmap_svg(["x"], ["m1", "m2"], [[1.0, None]], "t")
        self.assertIn("n/a", heat)


class TestBenchmarkHarness(unittest.TestCase):

    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, "-W", "ignore", "-m", "src.eval.benchmark", *args],
                              cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=900, env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})})

    def test_refuses_held_out_without_confirmation(self):
        res = self.run_cli("--splits", "test")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--confirm-heldout", res.stderr)

    def test_refuses_prompt_variants_without_a_live_model(self):
        res = self.run_cli("--prompt-variants", "baseline,cot")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("live model", res.stderr)

    def test_live_model_run_is_skipped_without_key(self):
        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
        res = subprocess.run([sys.executable, "-m", "src.eval.benchmark", "--model", "gemini-2.0-flash"],
                             cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300, env=env)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("SKIPPED", res.stderr)

    def test_dev_run_writes_honest_results(self):
        with tempfile.TemporaryDirectory() as out:
            res = self.run_cli("--splits", "dev", "--patterns", "sequential,original_loop", "--out", out)
            self.assertEqual(res.returncode, 0, res.stderr[-1500:])
            rows = list(csv.DictReader(open(Path(out) / "runs.csv", encoding="utf-8")))
            self.assertEqual(len(rows), 96)
            self.assertTrue(all(r["authorized_without_human"] == "False" for r in rows))
            self.assertTrue(all(r["input_tokens"] == "" for r in rows))          # not measured -> empty, not 0
            summary = json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("not measured", summary["tokens_note"])
            self.assertEqual(summary["totals"]["exceptions"], 0)
            self.assertTrue((Path(out) / "plots" / "latency_by_pattern.svg").exists())
            self.assertTrue((Path(out) / "plots" / "heatmap_agent_metrics_sequential.svg").exists())
            agents = list(csv.DictReader(open(Path(out) / "agent_metrics.csv", encoding="utf-8")))
            self.assertTrue(agents and {a["agent"] for a in agents} >= {"IntakeAgent", "VerifierAgent"})


if __name__ == "__main__":
    unittest.main()
