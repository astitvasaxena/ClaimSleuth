"""
run_claim(): one investigation of one claim, in one pattern and prompt style.

It seeds the blackboard (bb_claim, bb_documents, bb_run, bb_guardrail_hits), runs the ADK agents, and saves
the run and, if the Claim Manager submitted one, the FinalReport. A run without a submitted report ends
with status "failed"; no report is ever made up.

    python -m claimsleuth.runner --claim CLM-2024-001 --pattern hierarchical --style direct
(needs the app running for /mcp, and the CrewAI service.)
"""
import argparse
import asyncio
import json
import time
import uuid
from typing import Awaitable, Callable, Optional

from google.adk.agents import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from claimsleuth import audit, config, db, tracing
from claimsleuth.agents import patterns
from claimsleuth.callbacks import metrics
from claimsleuth.agents.common import (BB_CLAIM, BB_DOCUMENTS, BB_GUARDRAIL_HITS, BB_REPORT, BB_RUN, PROMPT_STYLES,
                                       read)
from claimsleuth.models import ClaimRecord, FinalReport, to_markdown

APP_NAME = "claimsleuth"
EventSink = Callable[[dict], Awaitable[None]]


def initial_state(claim: dict, documents: list[dict], run: dict) -> dict:
    record = ClaimRecord.model_validate(claim)
    docs = [{"doc_id": d["id"], "kind": d["kind"], "filename": d["filename"]} for d in documents]
    return {BB_CLAIM: record.model_dump_json(), BB_DOCUMENTS: json.dumps(docs), BB_RUN: json.dumps(run),
            BB_GUARDRAIL_HITS: "[]"}


def event_summary(event) -> dict:
    """The fields the live view shows for one ADK event."""
    parts = event.content.parts if event.content and event.content.parts else []
    calls = [p.function_call for p in parts if p.function_call]
    results = [p.function_response for p in parts if p.function_response]
    text = " ".join(p.text for p in parts if p.text and not getattr(p, "thought", False)).strip()
    kind = "tool_call" if calls else "tool_result" if results else "error" if event.error_message else "text"
    tool = (calls or results)[0].name if (calls or results) else None
    return {"agent": event.author, "type": kind, "tool": tool, "text": (event.error_message or text)[:500],
            "ts": round(time.time(), 3)}


async def run_claim(claim_id: str, pattern: str = "hierarchical", prompt_style: str = "direct",
                    on_event: Optional[EventSink] = None, run_id: Optional[str] = None) -> dict:
    if not config.model_configured():
        raise RuntimeError("No model key configured; investigations are disabled")
    if pattern not in patterns.PATTERNS or prompt_style not in PROMPT_STYLES:
        raise ValueError(f"pattern must be one of {patterns.PATTERNS}, prompt_style one of {PROMPT_STYLES}")
    claim = db.get_claim(claim_id)
    if claim is None:
        raise KeyError(f"unknown claim {claim_id}")
    documents = db.get_documents(claim_id)
    run_id = run_id or uuid.uuid4().hex[:12]
    run = {"run_id": run_id, "pattern": pattern, "prompt_style": prompt_style, "model": config.GEMINI_MODEL}
    started_at, started = db.now(), time.perf_counter()
    with db.connect() as conn:
        conn.execute("INSERT INTO runs (id, claim_id, pattern, prompt_style, model, started_at, status)"
                     " VALUES (?, ?, ?, ?, ?, ?, 'running')", (run_id, claim_id, pattern, prompt_style,
                                                               config.GEMINI_MODEL, started_at))
    ledger = audit.AuditLogger(claim_id)
    root, toolsets = patterns.build(pattern, prompt_style)
    sessions = InMemorySessionService()
    runner = Runner(app_name=APP_NAME, agent=root, session_service=sessions)
    await sessions.create_session(app_name=APP_NAME, user_id="adjuster", session_id=run_id,
                                  state=initial_state(claim, documents, run))
    status, error, report = "failed", None, None
    recorder = metrics.start_run(run_id, on_event)
    tracing.setup_tracing()
    try:
        with tracing.investigation_span(claim_id, run_id, {"pattern": pattern, "prompt_style": prompt_style}) as trace_id:
            run["trace_id"] = trace_id
            ledger.log_step(0, "RUN_STARTED", f"{pattern}/{prompt_style} run {run_id}", reasoning=config.GEMINI_MODEL)
            message = types.Content(role="user", parts=[types.Part(text=f"Investigate claim {claim_id}")])
            async with asyncio.timeout(config.RUN_TIMEOUT_S):
                async for event in runner.run_async(user_id="adjuster", session_id=run_id, new_message=message,
                                                    run_config=RunConfig(max_llm_calls=config.MAX_LLM_CALLS)):
                    summary = event_summary(event)
                    if summary["type"] in ("text", "error") and summary["text"]:
                        await recorder.emit(summary["agent"], summary["type"], summary["text"])
        state = (await sessions.get_session(app_name=APP_NAME, user_id="adjuster", session_id=run_id)).state
        if state.get(BB_REPORT):
            report = FinalReport.model_validate(read(state, BB_REPORT))
            status = "completed"
        else:
            error = "the Claim Manager did not submit a report"
    except TimeoutError:
        error = f"run stopped after {config.RUN_TIMEOUT_S} s"
    except Exception as exc:                                  # e.g. the model-call cap, or a service that is down
        error = f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        for toolset in toolsets:
            await toolset.close()
        metrics.finish_run(run_id)
    seconds = round(time.perf_counter() - started, 2)
    totals = recorder.totals()
    with db.connect() as conn:
        conn.execute("UPDATE runs SET seconds = ?, status = ?, error = ?, prompt_tokens = ?, completion_tokens = ?,"
                     " cost_usd = ? WHERE id = ?", (seconds, status, error, totals["prompt_tokens"],
                                                    totals["completion_tokens"], totals["cost_usd"], run_id))
        conn.executemany("INSERT INTO agent_metrics (run_id, agent, seconds, prompt_tokens, completion_tokens,"
                         " llm_calls, tool_calls, errors, guardrail_hits) VALUES (:run_id, :agent, :seconds,"
                         " :prompt_tokens, :completion_tokens, :llm_calls, :tool_calls, :errors, :guardrail_hits)",
                         [{"run_id": run_id, **row} for row in recorder.as_rows()])
        if report:
            conn.execute("INSERT INTO reports (claim_id, run_id, report_json, report_md, created_at)"
                         " VALUES (?, ?, ?, ?, ?)", (claim_id, run_id, report.model_dump_json(),
                                                     to_markdown(report), db.now()))
            conn.execute("UPDATE claims SET status = ? WHERE id = ?", (f"decided:{report.decision}", claim_id))
    ledger.log_step(1, "RUN_FINISHED", f"{status} in {seconds} s", reasoning=error or (report.decision if report else None))
    return {"run_id": run_id, "status": status, "error": error, "seconds": seconds, "trace_id": run.get("trace_id"),
            "metrics": recorder.as_rows(), **totals,
            "report": report.model_dump(mode="json") if report else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one ClaimSleuth investigation")
    parser.add_argument("--claim", required=True)
    parser.add_argument("--pattern", default="hierarchical", choices=patterns.PATTERNS)
    parser.add_argument("--style", default="direct", choices=PROMPT_STYLES)
    parser.add_argument("--events", action="store_true", help="print each agent event")
    args = parser.parse_args()

    async def show(event):
        print(f"  [{event['agent']}] {event['type']} {event['tool'] or ''} {event['text'][:160]}")
    db.init()
    result = asyncio.run(run_claim(args.claim, args.pattern, args.style, on_event=show if args.events else None))
    print(json.dumps(result, indent=2))
