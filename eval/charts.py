"""
Charts and summary from eval/results/runs.csv (nothing here calls a model or makes up a number):

    python eval/charts.py                                   # charts + summary.md from runs.csv
    python eval/charts.py --url http://127.0.0.1:8000       # also download each run's heat map from the API

Writes to eval/results/: boxplots of seconds and cost per configuration (experiment A) and per persona (B), a
summary heat map (agents x configurations, mean seconds), heatmaps/<run_id>.png, and summary.md.
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import httpx

from claimsleuth import config
from claimsleuth.charts import boxplot, heatmap

RESULTS = Path(__file__).resolve().parent / "results"
AGENTS = ("claim_manager", "evidence_agent", "policy_agent", "settlement_agent", "fraud_crew", "forensics", "history")
PERSONA_ORDER = ("honest", "abusive", "vague", "fraudster", "injector")


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(value) -> float | None:
    return float(value) if value not in ("", None) else None


def config_name(r: dict) -> str:
    return f"{r['pattern']}/{r['prompt_style']}"


def grouped(rows: list[dict], by, field: str, order=None) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        value = num(r[field])
        if value is not None:
            groups.setdefault(by(r), []).append(value)
    keys = [k for k in (order or sorted(groups)) if k in groups]
    return {k: groups[k] for k in keys}


def mean(values) -> str:
    values = [v for v in values if v is not None]
    return f"{statistics.mean(values):.2f}" if values else "not measured"


def table(header: list[str], body: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + \
           ["| " + " | ".join(str(c) for c in row) + " |" for row in body]


def write_charts(rows: list[dict], out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    a = [r for r in rows if r["experiment"] == "A"]
    b = [r for r in rows if r["experiment"] == "B"]
    for rows_, by, name, label, order in ((a, config_name, "A_by_configuration", "configuration", None),
                                         (b, lambda r: r["persona"], "B_by_persona", "persona", PERSONA_ORDER)):
        for field, unit in (("seconds", "seconds per run"), ("cost_usd", "cost per run (USD)")):
            groups = grouped(rows_, by, field, order)
            if groups:
                (out / f"boxplot_{field}_{name}.png").write_bytes(
                    boxplot(groups, unit, f"{unit.capitalize()} by {label} ({len(rows_)} runs)"))
                written.append(f"boxplot_{field}_{name}.png")
    if a:
        configs = sorted({config_name(r) for r in a})
        summary_rows = []
        for agent in AGENTS:
            row = {"agent": agent}
            for c in configs:
                values = [num(r[f"{agent}_seconds"]) for r in a if config_name(r) == c]
                values = [v for v in values if v is not None]
                row[c] = statistics.mean(values) if values else 0.0
            summary_rows.append(row)
        (out / "heatmap_agents_by_configuration.png").write_bytes(
            heatmap(summary_rows, "Mean seconds per agent, by configuration (experiment A)", columns=configs))
        written.append("heatmap_agents_by_configuration.png")
    return written


def download_heatmaps(rows: list[dict], url: str, out: Path) -> int:
    folder = out / "heatmaps"
    folder.mkdir(parents=True, exist_ok=True)
    got = 0
    with httpx.Client(base_url=url.rstrip("/"), headers={"Authorization": f"Bearer {config.ADJUSTER_TOKEN}"}, timeout=60) as http:
        for r in rows:
            target = folder / f"{r['run_id']}.png"
            if target.is_file():
                got += 1
                continue
            response = http.get(f"/api/runs/{r['run_id']}/heatmap.png")
            if response.status_code == 200:
                target.write_bytes(response.content)
                got += 1
    return got


def write_summary(rows: list[dict], charts: list[str], out: Path, heatmaps: int) -> Path:
    a = [r for r in rows if r["experiment"] == "A"]
    b = [r for r in rows if r["experiment"] == "B"]
    price = config.PRICES.get(config.GEMINI_MODEL)
    models = sorted({r["model"] for r in rows})
    targets = sorted({r["target"] for r in rows})
    lines = ["# ClaimSleuth evaluation", "",
             f"Model: {', '.join(models)}. Target: {', '.join(targets)}. Runs: {len(rows)} "
             f"(experiment A {len(a)} of 24, experiment B {len(b)} of 20; claims E1, E3, E4, E6). Summary written {time.strftime('%Y-%m-%d')}; "
             f"runs finished {min(r['finished_at'] for r in rows)[:10]} to {max(r['finished_at'] for r in rows)[:10]}.", ""]
    if price:
        lines += [f"Prices used for cost: ${price['input_per_million']} per million input tokens and "
                  f"${price['output_per_million']} per million output tokens ({price['tier']}), checked {price['checked']} "
                  f"at {price['source']}.", ""]
    else:
        lines += ["No price is recorded for this model, so cost is not measured.", ""]
    total_cost = [num(r["cost_usd"]) for r in rows]
    lines += table(["Total runs", "Correct", "Model calls", "Prompt tokens", "Completion tokens", "Cost (USD)",
                    "Run time (h)", "Rate-limit wait (h)"],
                   [[len(rows), sum(r["correct"] == "True" for r in rows), int(sum(num(r["llm_calls"]) or 0 for r in rows)),
                     int(sum(num(r["prompt_tokens"]) or 0 for r in rows)), int(sum(num(r["completion_tokens"]) or 0 for r in rows)),
                     f"{sum(v for v in total_cost if v is not None):.4f}" if any(v is not None for v in total_cost) else "not measured",
                     f"{sum(num(r['seconds']) or 0 for r in rows) / 3600:.2f}",
                     f"{sum(num(r['rate_limit_wait_s']) or 0 for r in rows) / 3600:.2f}"]]) + [""]
    if a:
        lines += ["## Experiment A: pattern and prompt style (honest persona, 4 claims each)", ""]
        body = []
        for c in sorted({config_name(r) for r in a}):
            g = [r for r in a if config_name(r) == c]
            body.append([c, len(g), f"{sum(r['correct'] == 'True' for r in g)}/{len(g)}", mean(num(r["seconds"]) for r in g),
                         mean(num(r["cost_usd"]) for r in g), mean(num(r["llm_calls"]) for r in g),
                         sum(r["status"] != "completed" for r in g)])
        lines += table(["Configuration", "Runs", "Correct", "Mean seconds", "Mean cost (USD)", "Mean model calls", "Failed runs"],
                       body) + [""]
        lines += ["Per claim (correct runs out of the 6 configurations):", ""]
        lines += table(["Claim", "Expected", "Correct"],
                       [[p, next(r["expected"] for r in a if r["claim"] == p),
                         f"{sum(r['correct'] == 'True' for r in a if r['claim'] == p)}/{sum(r['claim'] == p for r in a)}"]
                        for p in sorted({r["claim"] for r in a})]) + [""]
    if b:
        best = f"{b[0]['pattern']}/{b[0]['prompt_style']}"
        lines += [f"## Experiment B: personas ({best}, 4 claims each)", ""]
        body = []
        for persona in PERSONA_ORDER:
            g = [r for r in b if r["persona"] == persona]
            if not g:
                continue
            caught = ""
            if persona == "injector":
                caught = f"{sum((num(r['hits_injection']) or 0) > 0 for r in g)}/{len(g)} injection caught"
            elif persona == "abusive":
                caught = f"{sum((num(r['hits_abuse']) or 0) > 0 for r in g)}/{len(g)} abuse caught"
            body.append([persona, len(g), f"{sum(r['correct'] == 'True' for r in g)}/{len(g)}", caught,
                         mean(num(r["seconds"]) for r in g), mean(num(r["cost_usd"]) for r in g)])
        lines += table(["Persona", "Runs", "Correct", "Guardrail catch rate", "Mean seconds", "Mean cost (USD)"], body) + [""]
    lines += ["## Charts", ""] + [f"- `{c}`" for c in charts]
    if heatmaps:
        lines.append(f"- `heatmaps/`: one heat map per run ({heatmaps} runs)")
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=str(RESULTS / "runs.csv"))
    parser.add_argument("--out", default=str(RESULTS))
    parser.add_argument("--url", default=None, help="API to download the per-run heat maps from")
    args = parser.parse_args()
    runs = Path(args.runs)
    if not runs.is_file() or not load(runs):
        print(f"no runs in {runs}; run eval/run_eval.py first")
        return 1
    rows, out = load(runs), Path(args.out)
    charts = write_charts(rows, out)
    heatmaps = download_heatmaps(rows, args.url, out) if args.url else 0
    print(f"wrote {write_summary(rows, charts, out, heatmaps)} and {len(charts)} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
