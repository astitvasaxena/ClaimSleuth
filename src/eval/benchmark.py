"""
Benchmark evaluation (C-10) on the labelled synthetic dataset (src/eval/dataset.py).

    python -m src.eval.benchmark --splits dev,val                    # development splits
    python -m src.eval.benchmark --splits test --confirm-heldout     # held-out split, frozen config

Runs every claim of the chosen splits through each pattern (and the original loop as a
baseline), compares with the labels, and writes runs.csv, agent_metrics.csv, summary.md /
summary.json and SVG plots to data/runtime/eval/benchmark_<timestamp>/.

Honesty rules built in:
- Token and cost columns are filled ONLY from usage a live model reported. In the default
  deterministic mode no model is called, so they are reported as "not measured".
- Prompt variants change only what a live model is told, so they run only with --model.
- Held-out (test) runs need --confirm-heldout and are appended, with the code/config hash and
  dataset hash, to data/benchmark/heldout_runs.jsonl so repeated tuning on it is visible.
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = PROJECT_ROOT / "data" / "benchmark"
FROZEN_SOURCES = ["src/core/guardrails.py", "src/core/adjudication.py", "src/verifier/independent_verifier.py",
                  "src/adk_app/steps.py", "src/adk_app/patterns.py", "src/adk_app/pipeline.py",
                  "src/agents/investigator.py", "src/agents/extractor.py", "src/tools/policy_rules.py",
                  "src/tools/math_engine.py", "configs/policies.json"]
HUMAN_STATUSES = {"CONFLICT_ESCALATED", "NEEDS_EVIDENCE", "BLOCKED_BY_GUARDRAIL"}


def config_hash(extra: dict) -> str:
    """Hash of the decision-making code + run configuration: 'frozen config' identity for held-out runs."""
    h = hashlib.sha256(json.dumps(extra, sort_keys=True).encode())
    for rel in FROZEN_SOURCES:
        h.update((PROJECT_ROOT / rel).read_bytes())
    return h.hexdigest()


def _pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "—"


def _p(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    from src.eval.svgplot import quantile
    return round(quantile(values, q), 1)


def run_claim(pattern, claim, label, tools, model, prompt_variant):
    from src.adk_app.pipeline import run_adk_investigation
    from src.agents.investigator import LeadInvestigatorAgent
    from src.core.adjudication import payout_view
    from src.core.state import ClaimInvestigationState

    row = {"claim_id": label["claim_id"], "split": label["split"], "case_type": label["case_type"],
           "persona": label["persona"], "pattern": pattern, "expected_status": label["expected_status"],
           "expected_payout": label["expected_payout"], "expected_guardrail": label["expected_guardrail"]}
    agent_rows = []
    started = time.perf_counter()
    try:
        if pattern == "original_loop":
            st = LeadInvestigatorAgent(use_llm=False).run_investigation(ClaimInvestigationState(**claim))
            payout = payout_view(st)
            flags = st.guardrail_flags
            row.update(status=st.status, proposed_payout=payout["proposed_payout"],
                       authorized_payout=payout["authorized_payout"], trace_id=None, tool_calls=None,
                       agents_run=None, input_tokens=None, output_tokens=None,
                       guardrail=("block" if any(f.startswith("BLOCK:") for f in flags)
                                  else "redact" if flags else "allow"))
        else:
            result = run_adk_investigation(claim, pattern=pattern, tools=tools, model=model,
                                           prompt_variant=prompt_variant, fallback_on_model_error=False)
            rep = result.report
            tokens_in = [m.get("input_tokens") for m in rep.agent_metrics.values() if m.get("input_tokens") is not None]
            tokens_out = [m.get("output_tokens") for m in rep.agent_metrics.values() if m.get("output_tokens") is not None]
            row.update(status=rep.status, proposed_payout=rep.payout.proposed_payout,
                       authorized_payout=rep.payout.authorized_payout, trace_id=rep.trace_id,
                       tool_calls=len(rep.tool_calls) + len(rep.model_tool_calls), agents_run=len(rep.agent_sequence),
                       input_tokens=sum(tokens_in) if tokens_in else None,
                       output_tokens=sum(tokens_out) if tokens_out else None,
                       guardrail=(rep.guardrail or {}).get("action", "allow"))
            for agent, m in rep.agent_metrics.items():
                agent_rows.append({"claim_id": label["claim_id"], "split": label["split"], "persona": label["persona"],
                                   "pattern": pattern, "agent": agent, **m})
        row["error"] = ""
    except Exception as exc:
        row.update(status="EXCEPTION", proposed_payout=None, authorized_payout=None, guardrail=None,
                   error=f"{type(exc).__name__}: {exc}"[:300])
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    row["correct_status"] = row["status"] == label["expected_status"]
    row["correct_payout"] = row.get("proposed_payout") == label["expected_payout"]
    row["correct"] = row["correct_status"] and row["correct_payout"]
    row["unsafe_proposal"] = row.get("proposed_payout") is not None and not label["expected_payable"]
    row["authorized_without_human"] = row.get("authorized_payout") is not None
    row["missed_payable"] = label["expected_payable"] and row.get("proposed_payout") is None
    return row, agent_rows


def summarise(rows, agent_rows, meta):
    lines = [f"# ClaimSleuth benchmark ({meta['stamp']})", "",
             f"- Splits: {', '.join(meta['splits'])} | claims: {meta['claims']} | patterns: {', '.join(meta['patterns'])} "
             f"| runs: {len(rows)} | tools: {meta['tools']} | planner: {meta['planner']}",
             f"- Dataset sha256: `{meta['dataset_sha256'][:16]}…` (verified against manifest) | config hash: "
             f"`{meta['config_hash'][:16]}…`",
             f"- Tokens / cost: {meta['tokens_note']}",
             f"- Prompt variants: {meta['prompt_note']}", ""]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["pattern"])].append(r)
    lines += ["## Accuracy and safety by split and pattern", "",
              "| split | pattern | runs | status acc. | status+payout acc. | unsafe proposals | auto-authorized "
              "| missed payable | escalated to human | p50 ms | p90 ms |", "|" + "---|" * 11]
    for (split, pattern), cell in sorted(groups.items()):
        n = len(cell)
        lines.append(f"| {split} | {pattern} | {n} | {_pct(sum(r['correct_status'] for r in cell), n)} "
                     f"| {_pct(sum(r['correct'] for r in cell), n)} | {sum(r['unsafe_proposal'] for r in cell)} "
                     f"| {sum(r['authorized_without_human'] for r in cell)} | {sum(r['missed_payable'] for r in cell)} "
                     f"| {_pct(sum(r['status'] in HUMAN_STATUSES for r in cell), n)} "
                     f"| {_p([r['latency_ms'] for r in cell], .5)} | {_p([r['latency_ms'] for r in cell], .9)} |")
    lines.append("")

    # Guardrail confusion (expected vs observed action), all patterns
    lines += ["## Guardrails (expected vs observed action, all runs)", "",
              "| expected \\ observed | allow | redact | block | other |", "|---|---|---|---|---|"]
    conf = Counter((r["expected_guardrail"], r.get("guardrail")) for r in rows)
    for exp in ("allow", "redact", "block"):
        other = sum(v for (e, o), v in conf.items() if e == exp and o not in ("allow", "redact", "block"))
        lines.append(f"| {exp} | {conf[(exp, 'allow')]} | {conf[(exp, 'redact')]} | {conf[(exp, 'block')]} | {other} |")
    blocked = [r for r in rows if r["expected_guardrail"] == "block"]
    lines += ["", f"Blocked claims that became payable or authorized: "
                  f"{sum(1 for r in blocked if r.get('proposed_payout') is not None or r['authorized_without_human'])} "
                  f"of {len(blocked)}.", ""]

    for dim in ("persona", "case_type"):
        lines += [f"## Status+payout accuracy by {dim} (all patterns)", "", f"| {dim} | runs | accuracy |", "|---|---|---|"]
        by = defaultdict(list)
        for r in rows:
            by[r[dim]].append(r)
        for key, cell in sorted(by.items()):
            lines.append(f"| {key} | {len(cell)} | {_pct(sum(r['correct'] for r in cell), len(cell))} |")
        lines.append("")

    wrong = [r for r in rows if not r["correct"]]
    lines += ["## Incorrect runs", ""]
    lines += ([f"- {r['claim_id']} ({r['case_type']}, {r['persona']}, {r['pattern']}): expected "
               f"{r['expected_status']} / {r['expected_payout']}, got {r['status']} / {r.get('proposed_payout')} "
               f"{r.get('error', '')}" for r in wrong] or ["- None"])
    lines.append("")

    by_agent = defaultdict(list)
    for a in agent_rows:
        by_agent[(a["pattern"], a["agent"])].append(a)
    lines += ["## Per-agent metrics (ADK runs; mean per run)", "",
              "| pattern | agent | runs | latency ms | tool calls | errors | LLM tokens |", "|---|---|---|---|---|---|---|"]
    for (pattern, agent), cell in sorted(by_agent.items()):
        lat = [c["latency_ms"] for c in cell if c["latency_ms"] is not None]
        tok = [c["input_tokens"] for c in cell if c["input_tokens"] is not None]
        lines.append(f"| {pattern} | {agent} | {len(cell)} | {statistics.mean(lat):.1f} | "
                     f"{statistics.mean(c['tool_calls'] for c in cell):.2f} | "
                     f"{statistics.mean(c['errors'] for c in cell):.2f} | "
                     f"{statistics.mean(tok):.0f} |" if lat and tok else
                     f"| {pattern} | {agent} | {len(cell)} | {statistics.mean(lat):.1f} | "
                     f"{statistics.mean(c['tool_calls'] for c in cell):.2f} | "
                     f"{statistics.mean(c['errors'] for c in cell):.2f} | not measured |" if lat else
                     f"| {pattern} | {agent} | {len(cell)} | not measured | "
                     f"{statistics.mean(c['tool_calls'] for c in cell):.2f} | "
                     f"{statistics.mean(c['errors'] for c in cell):.2f} | not measured |")
    lines += ["", "## Plots", ""] + [f"- `{p}`" for p in meta["plots"]]
    return lines


def make_plots(rows, agent_rows, plot_dir: Path):
    from src.eval.svgplot import boxplot_svg, heatmap_svg
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(name, svg):
        (plot_dir / name).write_text(svg, encoding="utf-8")
        written.append(f"plots/{name}")

    ok_rows = [r for r in rows if r["status"] != "EXCEPTION"]
    patterns = list(dict.fromkeys(r["pattern"] for r in ok_rows))
    save("latency_by_pattern.svg", boxplot_svg({p: [r["latency_ms"] for r in ok_rows if r["pattern"] == p]
                                                for p in patterns},
                                               "End-to-end latency by pattern", "milliseconds per claim"))
    personas = list(dict.fromkeys(r["persona"] for r in ok_rows))
    save("latency_by_persona.svg", boxplot_svg({p: [r["latency_ms"] for r in ok_rows if r["persona"] == p
                                                    and r["pattern"] != "original_loop"] for p in personas},
                                               "ADK latency by claimant persona (all ADK patterns)", "milliseconds per claim"))
    agents = list(dict.fromkeys(a["agent"] for a in agent_rows))
    save("agent_latency_boxplot.svg", boxplot_svg(
        {a: [x["latency_ms"] for x in agent_rows if x["agent"] == a and x["latency_ms"] is not None] for a in agents},
        "Per-agent latency (all ADK runs)", "milliseconds", width=max(760, 95 * len(agents))))

    metrics = [("latency_ms", "latency ms"), ("tool_calls", "tool calls"), ("errors", "errors"),
               ("input_tokens", "LLM tokens")]
    for pattern in [p for p in patterns if p != "original_loop"]:
        cell = [a for a in agent_rows if a["pattern"] == pattern]
        names = list(dict.fromkeys(a["agent"] for a in cell))
        values = []
        for agent in names:
            mine = [a for a in cell if a["agent"] == agent]
            row = []
            for key, _ in metrics:
                vals = [a[key] for a in mine if a[key] is not None]
                row.append(round(statistics.mean(vals), 2) if vals else None)
            values.append(row)
        save(f"heatmap_agent_metrics_{pattern}.svg",
             heatmap_svg(names, [m[1] for m in metrics], values,
                         f"Per-agent metrics, {pattern} (mean per run; n/a = not measured)"))
    names = list(dict.fromkeys(a["agent"] for a in agent_rows))
    values = [[(lambda v: round(statistics.mean(v), 2) if v else None)(
        [a["errors"] for a in agent_rows if a["agent"] == agent and a["persona"] == persona])
        for persona in personas] for agent in names]
    save("heatmap_agent_errors_by_persona.svg",
         heatmap_svg(names, personas, values, "Errors per agent by persona (mean per run; includes guardrail blocks)",
                     fmt="{:.2f}", normalise="global"))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="ClaimSleuth benchmark evaluation")
    parser.add_argument("--splits", default="dev,val")
    parser.add_argument("--patterns", default="original_loop,sequential,parallel,hierarchical")
    parser.add_argument("--tools", default="local", choices=["local", "mcp"])
    parser.add_argument("--confirm-heldout", action="store_true", help="required to evaluate the held-out test split")
    parser.add_argument("--model", default=None, help="live model (e.g. gemini-2.0-flash); needs GOOGLE_API_KEY")
    parser.add_argument("--prompt-variants", default="baseline", help="comma list; only meaningful with --model")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "test" in splits and not args.confirm_heldout:
        sys.exit("Refusing to run the held-out 'test' split without --confirm-heldout "
                 "(evaluate it only on a frozen configuration).")
    if args.model and not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("SKIPPED: --model needs GOOGLE_API_KEY; no live-model evaluation was run.")
    variants = [v.strip() for v in args.prompt_variants.split(",") if v.strip()]
    if not args.model and variants != ["baseline"]:
        sys.exit("Prompt variants change only what a live model is told; run them with --model.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else PROJECT_ROOT / "data" / "runtime" / "eval" / f"benchmark_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["CLAIMSLEUTH_CLAIMS_DIR"] = str(BENCH_DIR / "claims")
    for var, sub in (("CLAIMSLEUTH_AUDIT_LOG_DIR", "audit"), ("CLAIMSLEUTH_REPORTS_DIR", "reports"),
                     ("CLAIMSLEUTH_TRACES_DIR", "traces")):
        os.environ[var] = str(out / sub)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.eval.dataset import dataset_digest

    manifest = json.loads((BENCH_DIR / "manifest.json").read_text(encoding="utf-8"))
    digest = dataset_digest(BENCH_DIR)
    if digest != manifest["dataset_sha256"]:
        sys.exit(f"Dataset files do not match the manifest (sha256 {digest[:16]} != {manifest['dataset_sha256'][:16]}); "
                 "regenerate with python -m src.eval.dataset")

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    run_config = {"patterns": patterns, "tools": args.tools, "model": args.model, "prompt_variants": variants}
    chash = config_hash(run_config)
    rows, agent_rows = [], []
    started = time.perf_counter()
    for split in splits:
        for claim_id in manifest["splits"][split]:
            label = json.loads((BENCH_DIR / "labels" / f"{claim_id}.json").read_text(encoding="utf-8"))
            claim = json.loads((BENCH_DIR / "claims" / claim_id / "claim.json").read_text(encoding="utf-8"))
            for pattern in patterns:
                for variant in variants:
                    if pattern == "original_loop" and args.model:
                        continue
                    row, arows = run_claim(pattern, claim, label, args.tools, args.model, variant)
                    row["prompt_variant"] = variant
                    rows.append(row)
                    agent_rows += arows

    plots = make_plots(rows, agent_rows, out / "plots")
    measured_tokens = any(r.get("input_tokens") is not None for r in rows)
    meta = {"stamp": stamp, "splits": splits, "patterns": patterns, "tools": args.tools,
            "claims": sum(len(manifest["splits"][s]) for s in splits),
            "planner": f"llm:{args.model}" if args.model else "deterministic (no LLM calls)",
            "dataset_sha256": digest, "config_hash": chash, "plots": plots,
            "tokens_note": ("measured from model usage metadata" if measured_tokens else
                            "not measured: no live model was called in this run, so no token or cost figures exist"),
            "prompt_note": (", ".join(variants) if args.model else
                            "not evaluated: prompt variants only change a live model's instructions")}
    fields = ["claim_id", "split", "case_type", "persona", "pattern", "prompt_variant", "expected_status",
              "status", "expected_payout", "proposed_payout", "authorized_payout", "correct_status", "correct_payout",
              "correct", "unsafe_proposal", "authorized_without_human", "missed_payable", "expected_guardrail",
              "guardrail", "latency_ms", "agents_run", "tool_calls", "input_tokens", "output_tokens", "trace_id", "error"]
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "agent_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["claim_id", "split", "persona", "pattern", "agent", "latency_ms", "tool_calls",
                                          "errors", "llm_calls", "input_tokens", "output_tokens"], extrasaction="ignore")
        w.writeheader()
        w.writerows(agent_rows)
    summary = summarise(rows, agent_rows, meta)
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    totals = {"runs": len(rows), "correct": sum(r["correct"] for r in rows),
              "unsafe_proposals": sum(r["unsafe_proposal"] for r in rows),
              "authorized_without_human": sum(r["authorized_without_human"] for r in rows),
              "exceptions": sum(r["status"] == "EXCEPTION" for r in rows),
              "wall_seconds": round(time.perf_counter() - started, 1)}
    (out / "summary.json").write_text(json.dumps({**meta, **run_config, "totals": totals}, indent=2), encoding="utf-8")
    if "test" in splits:
        with open(BENCH_DIR / "heldout_runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": stamp, "config_hash": chash, "dataset_sha256": digest, **run_config,
                                "runs": len(rows), "correct": totals["correct"],
                                "output": str(out.relative_to(PROJECT_ROOT))}) + "\n")
    print(f"{len(rows)} runs -> {out}")
    print(json.dumps(totals))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
