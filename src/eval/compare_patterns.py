"""
Pattern and messaging comparison (C-4 / C-1): runs the real ADK pipeline across

    pattern   x  messaging  x  claim  x  tool backend  x  repeats
    (+ CrewAI forensics over A2A when the service is reachable, + injected failures)

and writes every run to CSV plus a Markdown summary. Nothing is estimated: each row is
one executed investigation. Expected outcomes are the hand-checked results of the three
benchmark claims, so this measures consistency and safety, not accuracy on unseen data.

    python -m src.eval.compare_patterns --repeats 3
    python -m src.eval.compare_patterns --repeats 1 --tools local      # quick run

Output: data/runtime/eval/<timestamp>/patterns.csv and patterns.md (gitignored).
"""
import argparse
import csv
import json
import os
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPECTED = {"scenario_1_valid": ("VERIFIED_PENDING_APPROVAL", "PROPOSED_PENDING_APPROVAL", 251.0),
            "scenario_2_expired": ("REJECTED_DETERMINISTICALLY", "NOT_PAYABLE", None),
            "scenario_3_conflict": ("CONFLICT_ESCALATED", "NOT_PAYABLE", None)}
PATTERNS = ("sequential", "parallel", "hierarchical")
MESSAGING = ("blackboard", "direct", "broadcast")


def _claim(scenario):
    return json.loads((PROJECT_ROOT / "data" / "sample_claims" / scenario / "claim.json").read_text(encoding="utf-8"))


def _service_up(url: str) -> bool:
    import httpx
    try:
        return httpx.get(f"{url}/.well-known/agent-card.json", timeout=2).status_code == 200
    except Exception:
        return False


def _overlap_ms(report) -> float:
    span = {t["agent"]: (t["start_ms"], t["end_ms"]) for t in report.agent_timings}
    if "AnomalyAgent" not in span or "WarrantyAgent" not in span:
        return 0.0
    (a0, a1), (w0, w1) = span["AnomalyAgent"], span["WarrantyAgent"]
    return round(max(0.0, min(a1, w1) - max(a0, w0)), 1)


def run_one(run_adk_investigation, group, scenario, pattern, messaging, tools, forensics="none",
            fault="none", expected=None, **kwargs):
    started = time.perf_counter()
    row = {"group": group, "scenario": scenario, "pattern": pattern, "messaging": messaging, "tools": tools,
           "forensics": forensics, "fault": fault}
    try:
        result = run_adk_investigation(_claim(scenario), pattern=pattern, messaging=messaging,
                                       tools="mcp" if tools.startswith("mcp") else "local",
                                       forensics=forensics, **kwargs)
        rep = result.report
        got = (rep.status, rep.payout.payout_status, rep.payout.proposed_payout)
        m = rep.messaging_summary
        row.update(
            status=rep.status, payout_status=rep.payout.payout_status, proposed=rep.payout.proposed_payout,
            correct=(got == (expected or EXPECTED[scenario])),
            fail_closed=(rep.payout.payout_status == "NOT_PAYABLE" and rep.payout.authorized_payout is None),
            wall_ms=result.wall_ms, agents_run=len(rep.agent_sequence), tool_calls=len(rep.tool_calls),
            tool_errors=len(rep.tool_errors), conflicts=len(rep.conflicts),
            manager_decisions=len(rep.manager_decisions), branch_overlap_ms=_overlap_ms(rep),
            messages_sent=m.get("messages_sent"), deliveries=m.get("deliveries"),
            context_bytes_per_agent=m.get("context_bytes_per_agent"),
            risk_flags="|".join(rep.risk_flags), error="")
    except Exception as exc:
        row.update(status="EXCEPTION", correct=False, fail_closed=None,
                   wall_ms=round((time.perf_counter() - started) * 1000, 1), error=f"{type(exc).__name__}: {exc}"[:300])
    return row


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return "—", "—"
    values = sorted(values)
    p90 = values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))]
    return f"{statistics.median(values):.0f}", f"{p90:.0f}"


def summarise(rows):
    out = []
    groups = [("A. Patterns x tool backend (blackboard messaging)", "matrix", ("pattern", "tools"),
               lambda r: r["messaging"] == "blackboard"),
              ("B. Patterns with CrewAI forensics over A2A (local tools, blackboard)", "crewai", ("pattern",),
               lambda r: True)]
    for title, group, keys, extra in groups:
        sel = [r for r in rows if r["group"] == group and extra(r)]
        if not sel:
            out += [f"## {title}", "", "_Not run (see notes)._", ""]
            continue
        out += [f"## {title}", "", "| " + " | ".join(keys) + " | runs | correct | median ms | p90 ms | agents run "
                "| tool calls | branch overlap >0 |", "|" + "---|" * (len(keys) + 7)]
        combos = sorted({tuple(r[k] for k in keys) for r in sel})
        for combo in combos:
            cell = [r for r in sel if tuple(r[k] for k in keys) == combo]
            med, p90 = _stats([r["wall_ms"] for r in cell])
            ok = sum(1 for r in cell if r["correct"])
            agents = statistics.mean(r.get("agents_run") or 0 for r in cell)
            calls = statistics.mean(r.get("tool_calls") or 0 for r in cell)
            overlap = sum(1 for r in cell if (r.get("branch_overlap_ms") or 0) > 0)
            out.append(f"| {' | '.join(combo)} | {len(cell)} | {ok}/{len(cell)} | {med} | {p90} | {agents:.1f} "
                       f"| {calls:.1f} | {overlap}/{len(cell)} |")
        out.append("")

    sel = [r for r in rows if r["group"] == "matrix" and r["tools"] == "local"]
    out += ["## C. Messaging architectures (local tools)", "",
            "| pattern | messaging | runs | correct | messages sent | deliveries | context bytes / agent |",
            "|---|---|---|---|---|---|---|"]
    for pattern in PATTERNS:
        for messaging in MESSAGING:
            cell = [r for r in sel if r["pattern"] == pattern and r["messaging"] == messaging]
            if cell:
                out.append(f"| {pattern} | {messaging} | {len(cell)} | {sum(r['correct'] for r in cell)}/{len(cell)} "
                           f"| {statistics.mean(r['messages_sent'] or 0 for r in cell):.1f} "
                           f"| {statistics.mean(r['deliveries'] or 0 for r in cell):.1f} "
                           f"| {statistics.mean(r['context_bytes_per_agent'] or 0 for r in cell):.0f} |")
    out.append("")

    sel = [r for r in rows if r["group"] == "failure"]
    out += ["## D. Injected failures", "", "| pattern | messaging | fault | status | payout | fail-closed | flags |",
            "|---|---|---|---|---|---|---|"]
    for r in sel:
        out.append(f"| {r['pattern']} | {r['messaging']} | {r['fault']} | {r['status']} | {r.get('payout_status', '—')} "
                   f"| {'yes' if r.get('fail_closed') else 'NO'} | {r.get('risk_flags', '')} |")
    out.append("")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tools", default="local,mcp", help="comma list of: local, mcp")
    parser.add_argument("--crewai-url", default=os.environ.get("CLAIMSLEUTH_CREWAI_A2A_URL", "http://127.0.0.1:8101"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "data" / "runtime" / "eval" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep evaluation ledgers and reports apart from normal runs (must be set before project imports)
    os.environ["CLAIMSLEUTH_AUDIT_LOG_DIR"] = str(out_dir / "audit")
    os.environ["CLAIMSLEUTH_REPORTS_DIR"] = str(out_dir / "reports")
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.adk_app.pipeline import run_adk_investigation
    from src.adk_app.tool_gateway import McpConfig

    tools_list = [t.strip() for t in args.tools.split(",") if t.strip()]
    rows, notes = [], []
    t_start = time.perf_counter()
    for repeat in range(args.repeats):
        for tools in tools_list:
            for pattern in PATTERNS:
                for messaging in MESSAGING:
                    if tools == "mcp" and messaging != "blackboard":
                        continue   # messaging is compared on local tools; MCP is compared per pattern
                    for scenario in EXPECTED:
                        rows.append(run_one(run_adk_investigation, "matrix", scenario, pattern, messaging,
                                            "mcp_stdio" if tools == "mcp" else "local"))

    if _service_up(args.crewai_url):
        for repeat in range(args.repeats):
            for pattern in PATTERNS:
                for scenario in EXPECTED:
                    rows.append(run_one(run_adk_investigation, "crewai", scenario, pattern, "blackboard", "local",
                                        forensics="crewai_a2a", a2a_url=args.crewai_url))
    else:
        notes.append(f"CrewAI service not reachable at {args.crewai_url}: section B not run.")

    faulty_mcp = str(PROJECT_ROOT / "tests" / "fixtures" / "faulty_mcp_server.py")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    for pattern in PATTERNS:
        rows.append(run_one(run_adk_investigation, "failure", "scenario_1_valid", pattern, "blackboard", "mcp_stdio",
                            fault="mcp_settlement_error",
                            mcp=McpConfig(server_args=[faulty_mcp, "--fault", "settlement_error"]),
                            expected=("NEEDS_EVIDENCE", "NOT_PAYABLE", None)))
        rows.append(run_one(run_adk_investigation, "failure", "scenario_1_valid", pattern, "blackboard", "mcp_stdio",
                            fault="mcp_tampered_settlement",
                            mcp=McpConfig(server_args=[faulty_mcp, "--fault", "tamper_settlement"]),
                            expected=("CONFLICT_ESCALATED", "NOT_PAYABLE", None)))
        rows.append(run_one(run_adk_investigation, "failure", "scenario_1_valid", pattern, "blackboard", "local",
                            forensics="crewai_a2a", fault="crewai_service_down",
                            a2a_url=f"http://127.0.0.1:{dead_port}", a2a_timeout=5,
                            expected=("NEEDS_EVIDENCE", "NOT_PAYABLE", None)))
        rows.append(run_one(run_adk_investigation, "failure", "scenario_1_valid", pattern, "direct", "local",
                            fault="direct_route_dropped(Evidence->Settlement)",
                            drop_routes={("EvidenceAgent", "SettlementAgent")},
                            expected=("NEEDS_EVIDENCE", "NOT_PAYABLE", None)))

    fields = ["group", "scenario", "pattern", "messaging", "tools", "forensics", "fault", "status", "payout_status",
              "proposed", "correct", "fail_closed", "wall_ms", "agents_run", "tool_calls", "tool_errors", "conflicts",
              "manager_decisions", "branch_overlap_ms", "messages_sent", "deliveries", "context_bytes_per_agent",
              "risk_flags", "error"]
    with open(out_dir / "patterns.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    md = [f"# Pattern and messaging comparison ({stamp})", "",
          f"- Runs: {total} ({sum(1 for r in rows if r['correct'])} with the expected outcome); "
          f"exceptions: {sum(1 for r in rows if r['status'] == 'EXCEPTION')}; wall time "
          f"{time.perf_counter() - t_start:.0f}s; repeats per cell: {args.repeats}",
          "- Benchmark: the 3 hand-checked sample claims only. Latencies are single-machine wall times "
          "(Windows, Python 3.12); MCP stdio runs include spawning the server process per run.",
          "- Deterministic planner (no LLM). The hierarchical manager is a deterministic policy, not an LLM.",
          *[f"- {n}" for n in notes], ""] + summarise(rows)
    (out_dir / "patterns.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{total} runs written to {out_dir}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
