"""
The evaluation runner. It calls the ClaimSleuth API the way a client does, one run at a time, and appends one
row per run to eval/results/runs.csv. It is resumable: runs already in the file are skipped.

    python eval/run_eval.py --url http://127.0.0.1:8000 --experiment A     # 3 patterns x 2 prompt styles x 4 claims = 24
    python eval/run_eval.py --url http://127.0.0.1:8000 --experiment B     # best configuration of A x 4 claims x 5 personas = 20

The evaluation uses 4 of the 8 claim packets (E1 valid repair, E3 excluded cause, E4 serial mismatch, E6 missing
invoice), so the 44 runs fit the free tier of the pinned model; all 8 packets and 40 personas stay in eval/.
On a daily quota error the runner stops with "daily quota reached, resume tomorrow"; rerun it the next day.
    python eval/run_eval.py --dry-run                                       # print the plan; no requests

Needs ADJUSTER_TOKEN (from .env). Every row must come from the pinned model (claimsleuth.config.GEMINI_MODEL);
the runner refuses to continue if the server or an existing row uses another model.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

from claimsleuth import config, ratelimit
from claimsleuth.db import PERSONAS, eval_claim_id

RESULTS = Path(__file__).resolve().parent / "results"
RUNS_CSV = RESULTS / "runs.csv"
RUN_INFO = RESULTS / "run_info.json"
RETRIES_CSV = RESULTS / "rate_limit_retries.csv"
# The code every run must come from: the app, the agents, the crew, the evaluation scripts and the data they read.
CODE_PATHS = ["src", "configs", "data/sample_claims", "data/rag/policies", "data/past_claims.json", "eval/claims",
              "eval/personas", "eval/run_eval.py", "requirements.txt", "requirements-crewai-service.txt"]
REPO = Path(__file__).resolve().parents[1]
LABELS = json.loads((Path(__file__).resolve().parent / "claims" / "labels.json").read_text(encoding="utf-8"))
PACKETS = sorted(LABELS)
EVAL_CLAIMS = ("E1", "E3", "E4", "E6")              # the claims the 44-run evaluation uses
PATTERNS = ("sequential", "parallel", "hierarchical")
STYLES = ("direct", "react")
AGENTS = ("claim_manager", "evidence_agent", "policy_agent", "settlement_agent", "fraud_crew", "forensics", "history")
COLUMNS = (["run_id", "experiment", "claim", "persona", "pattern", "prompt_style", "model", "commit", "target", "status", "error",
            "expected", "decision", "correct", "fraud_flagged", "payout", "seconds", "rate_limit_wait_s",
            "prompt_tokens", "completion_tokens", "cost_usd", "llm_calls", "tool_calls", "errors", "guardrail_hits",
            "hits_injection", "hits_abuse", "finished_at"]
           + [f"{a}_{m}" for a in AGENTS for m in ("seconds", "tokens")])
BACKOFF_S = (60, 120, 240, 480, 900)


def plan_a() -> list[dict]:
    return [{"experiment": "A", "claim": p, "persona": "honest", "pattern": pat, "prompt_style": st}
            for pat in PATTERNS for st in STYLES for p in EVAL_CLAIMS]


def plan_b(best: tuple[str, str]) -> list[dict]:
    return [{"experiment": "B", "claim": p, "persona": per, "pattern": best[0], "prompt_style": best[1]}
            for p in EVAL_CLAIMS for per in PERSONAS]


def key(row: dict) -> tuple:
    return (row["experiment"], row["claim"], row["persona"], row["pattern"], row["prompt_style"])


def read_rows() -> list[dict]:
    if not RUNS_CSV.is_file():
        return []
    with RUNS_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_row(row: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = not RUNS_CSV.is_file()
    with RUNS_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in COLUMNS})


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def code_commit() -> str:
    """The commit the evaluation runs on. Refuses if the code (CODE_PATHS) has uncommitted changes."""
    dirty = git("status", "--porcelain", "--", *CODE_PATHS)
    if dirty:
        sys.exit(f"uncommitted changes in the evaluated code; commit or revert them first:\n{dirty}")
    return git("rev-parse", "HEAD")


def check_run_info(commit: str, target: str) -> dict:
    """First start: record the commit, model and start time. Later starts: the code must be the recorded commit's."""
    if RUN_INFO.is_file():
        info = json.loads(RUN_INFO.read_text(encoding="utf-8"))
        changed = git("diff", "--name-only", info["commit"], commit, "--", *CODE_PATHS)
        if changed or info["model"] != config.GEMINI_MODEL:
            sys.exit(f"the evaluation started on commit {info['commit'][:10]} with {info['model']}; the code or model "
                     f"has changed since ({changed or config.GEMINI_MODEL}). Not mixing.")
        return info
    info = {"commit": commit, "model": config.GEMINI_MODEL, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "target": target, "claims": list(EVAL_CLAIMS), "patterns": list(PATTERNS), "prompt_styles": list(STYLES),
            "personas": list(PERSONAS), "runs_planned": {"A": len(plan_a()), "B": len(EVAL_CLAIMS) * len(PERSONAS)}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    RUN_INFO.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def log_retry(planned: dict, outcome: dict, waited_s: int, kind: str) -> None:
    """A run cut off by a rate limit is not a result; it is logged here, and redone."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = not RETRIES_CSV.is_file()
    with RETRIES_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["at", "experiment", "claim", "persona", "pattern", "prompt_style", "discarded_run_id",
                             "kind", "wait_before_retry_s", "error"])
        writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), planned["experiment"], planned["claim"], planned["persona"],
                         planned["pattern"], planned["prompt_style"], outcome["run"]["id"], kind, waited_s,
                         (outcome["run"]["error"] or "")[:300]])


def best_config(rows: list[dict]) -> tuple[str, str]:
    """Highest accuracy in experiment A; ties go to the lower mean cost (then the lower mean time)."""
    a = [r for r in rows if r["experiment"] == "A"]
    if len(a) < len(plan_a()):
        raise SystemExit(f"experiment A has {len(a)} of {len(plan_a())} runs; finish it before experiment B")
    scores = []
    for pat in PATTERNS:
        for st in STYLES:
            group = [r for r in a if r["pattern"] == pat and r["prompt_style"] == st]
            accuracy = sum(r["correct"] == "True" for r in group) / len(group)
            costs = [float(r["cost_usd"]) for r in group if r["cost_usd"]]
            seconds = [float(r["seconds"]) for r in group if r["seconds"]]
            scores.append((-accuracy, sum(costs) / len(costs) if costs else float("inf"),
                           sum(seconds) / len(seconds) if seconds else float("inf"), pat, st))
    _, _, _, pattern, style = min(scores)
    return pattern, style


def is_rate_limit(error: str) -> bool:
    return ratelimit.classify(error) is not None


def is_daily_quota(error: str) -> bool:
    """The server stores rate-limit failures as 'RATE_LIMIT_DAILY: ...' or 'RATE_LIMIT_MINUTE: ...'."""
    return (error or "").startswith(ratelimit.DAILY) or ratelimit.classify(error) == ratelimit.DAILY


class Client:

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.http = httpx.Client(base_url=self.url, headers={"Authorization": f"Bearer {token}"}, timeout=60)

    def model(self) -> str:
        response = self.http.get("/api/status")
        response.raise_for_status()
        return response.json()["model"]

    def run(self, claim_id: str, pattern: str, style: str) -> dict:
        response = self.http.post(f"/api/claims/{claim_id}/investigate", params={"pattern": pattern, "prompt_style": style})
        response.raise_for_status()
        run_id = response.json()["run_id"]
        result = None
        with self.http.stream("GET", f"/api/runs/{run_id}/events", timeout=httpx.Timeout(60, read=None)) as stream:
            event = None
            for line in stream.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: ") and event == "done":
                    result = json.loads(line[6:])
                    break
        if result is None:
            raise RuntimeError(f"run {run_id}: the event stream ended without a result")
        details = self.http.get(f"/api/runs/{run_id}")
        details.raise_for_status()
        return {"result": result, **details.json()}


def to_row(planned: dict, outcome: dict, waited: float, target: str, commit: str = "") -> dict:
    run, report, result = outcome["run"], outcome["report"], outcome["result"]
    label = LABELS[planned["claim"]]
    decision = report["decision"] if report else ""
    flagged = bool(report and report.get("fraud") and report["fraud"]["flagged"])
    payout = report["payout"]["amount"] if report and report.get("payout") else ""
    correct = (run["status"] == "completed" and decision in label["accepted"]
               and (flagged or not label["fraud_expected"])
               and ("payout" not in label or payout == label["payout"]))
    hits = report["guardrail_hits"] if report else []
    agents = {a["agent"]: a for a in outcome["agents"]}
    totals = {m: sum(a[m] for a in outcome["agents"]) for m in ("llm_calls", "tool_calls", "errors", "guardrail_hits")}
    row = {**planned, "run_id": run["id"], "model": run["model"], "commit": commit, "target": target, "status": run["status"],
           "error": (run["error"] or "")[:300], "expected": "|".join(label["accepted"]), "decision": decision,
           "correct": correct, "fraud_flagged": flagged, "payout": payout, "seconds": run["seconds"],
           "rate_limit_wait_s": round(waited + float(run.get("wait_seconds") or 0), 1), "prompt_tokens": run["prompt_tokens"],
           "completion_tokens": run["completion_tokens"], "cost_usd": run["cost_usd"] if run["cost_usd"] is not None else "",
           "hits_injection": sum(h["kind"] == "injection" for h in hits), "hits_abuse": sum(h["kind"] == "abuse" for h in hits),
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **totals}
    for name in AGENTS:
        a = agents.get(name)
        row[f"{name}_seconds"] = a["seconds"] if a else ""
        row[f"{name}_tokens"] = (a["prompt_tokens"] + a["completion_tokens"]) if a else ""
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=config.APP_URL)
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    parser.add_argument("--target", default="local docker compose",
                        help="what the URL is, for the results: 'deployed <url>' or 'local docker compose'")
    parser.add_argument("--max-runs", type=int, default=None, help="stop after this many new runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_rows()
    models = {r["model"] for r in rows}
    if models - {config.GEMINI_MODEL}:
        sys.exit(f"runs.csv holds runs from {sorted(models)}; the pinned model is {config.GEMINI_MODEL}. Not mixing.")
    planned = plan_a() if args.experiment == "A" else plan_b(best_config(rows))
    done = {key(r) for r in rows}
    todo = [p for p in planned if key(p) not in done]
    print(f"experiment {args.experiment}: {len(planned)} runs planned, {len(planned) - len(todo)} done, {len(todo)} to go")
    if args.dry_run:
        for p in todo:
            print(f"  {p['claim']} {p['persona']:9} {p['pattern']:12} {p['prompt_style']}")
        return 0
    if not config.ADJUSTER_TOKEN:
        sys.exit("ADJUSTER_TOKEN is not set")
    commit = code_commit()
    info = check_run_info(commit, args.target)
    print(f"evaluation on commit {info['commit'][:10]}, model {info['model']}, started {info['started_at']}")
    client = Client(args.url, config.ADJUSTER_TOKEN)
    server_model = client.model()
    if server_model != config.GEMINI_MODEL:
        sys.exit(f"the server uses {server_model}; the pinned model is {config.GEMINI_MODEL}. Not mixing.")
    for count, p in enumerate(todo[:args.max_runs] if args.max_runs else todo, start=1):
        waited = 0.0
        for attempt in range(len(BACKOFF_S) + 1):
            outcome = client.run(eval_claim_id(p["claim"], p["persona"]), p["pattern"], p["prompt_style"])
            error = outcome["run"]["error"] or ""
            if outcome["run"]["status"] == "completed" or not is_rate_limit(error):
                break
            if is_daily_quota(error):
                log_retry(p, outcome, 0, "daily")
                print(f"daily quota reached, resume tomorrow. {len(read_rows())} runs are saved in {RUNS_CSV}; this run "
                      f"({p['claim']} {p['persona']} {p['pattern']}/{p['prompt_style']}) will be redone. [{error[:200]}]")
                return 3
            if attempt == len(BACKOFF_S):
                print(f"stopping: still rate limited after {len(BACKOFF_S)} waits ({error[:160]}). Finished runs are kept.")
                return 2
            log_retry(p, outcome, BACKOFF_S[attempt], "per-minute")
            print(f"  per-minute rate limit; waiting {BACKOFF_S[attempt]} s (not counted as agent time)")
            time.sleep(BACKOFF_S[attempt])
            waited += BACKOFF_S[attempt]
        row = to_row(p, outcome, waited, args.target, info["commit"])
        append_row(row)
        print(f"[{count}/{len(todo)}] {p['claim']} {p['persona']} {p['pattern']}/{p['prompt_style']}: "
              f"{row['status']} {row['decision'] or '-'} correct={row['correct']} {row['seconds']} s "
              f"{row['prompt_tokens']}+{row['completion_tokens']} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
