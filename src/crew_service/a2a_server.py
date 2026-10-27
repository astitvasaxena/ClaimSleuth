"""
CrewAI forensics service exposed over the A2A protocol (a2a-sdk 1.3.0).

Run in the separate CrewAI environment:
    .venv-crewai/Scripts/python -m src.crew_service.a2a_server --port 8101
    Agent Card:  http://127.0.0.1:8101/.well-known/agent-card.json
    JSON-RPC:    http://127.0.0.1:8101/

A request is an A2A message whose text contains the claim JSON (or at least a
"claim_id"). The service loads its OWN copy of the claim from the dossier, runs the
CrewAI ForensicsFlow, and returns a ForensicsReport artifact. It never returns a
payout or an approval: the schema forbids such fields.

Optional narrative LLM (CLAIMSLEUTH_CREWAI_LLM or --llm):
    none (default)         Flow only, no LLM, no network
    fake                   test double (tests/fixtures/fake_crewai_llm.py)
    <crewai model string>  e.g. "gemini/gemini-2.0-flash"; needs an API key; NOT live-tested
No authentication: binds to localhost by default; do not expose publicly.
"""
import argparse
import asyncio
import contextvars
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.crew_service.forensics_flow import CLAIM_ID_PATTERN, run_forensics  # sets telemetry opt-outs first

from a2a.helpers import new_data_part, new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from starlette.applications import Starlette

from src.core.forensics_schema import FORENSICS_JSON_PREFIX, ForensicsReport
from src.tools.evidence_store import EvidenceStore

SERVICE_NAME = "ClaimSleuthCrewAIForensics"
# Trace id of the request being served, taken from the caller's W3C traceparent header
REQUEST_TRACE_ID: contextvars.ContextVar = contextvars.ContextVar("request_trace_id", default=None)


class TraceparentMiddleware:
    """ASGI middleware: read the W3C traceparent header so logs and reports join the caller's trace."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            parts = headers.get(b"traceparent", b"").decode("latin-1").split("-")
            token = REQUEST_TRACE_ID.set(parts[1] if len(parts) == 4 and len(parts[1]) == 32 else None)
            try:
                await self.app(scope, receive, send)
            finally:
                REQUEST_TRACE_ID.reset(token)
            return
        await self.app(scope, receive, send)


def _log_call(entry: Dict[str, Any]) -> None:
    """Append each handled request (with this process's pid) to CLAIMSLEUTH_A2A_CALL_LOG, if set."""
    path = os.environ.get("CLAIMSLEUTH_A2A_CALL_LOG")
    if path:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                 "trace_id": REQUEST_TRACE_ID.get(), **entry}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def extract_claim_id(text: str) -> Optional[str]:
    """Find the claim id in the request text (claim JSON, or any '"claim_id": "..."')."""
    match = re.search(r'"claim_id"\s*:\s*"([^"]+)"', text or "")
    return match.group(1) if match else None


def load_claim(claim_id: str) -> Optional[Dict[str, Any]]:
    """The service trusts only its own dossier copy of the claim, never the caller's fields."""
    folder = EvidenceStore().get_claim_folder(claim_id)
    if folder is None or not (folder / "claim.json").exists():
        return None
    claim = json.loads((folder / "claim.json").read_text(encoding="utf-8"))
    return claim if claim.get("claim_id") == claim_id else None


def resolve_llm(spec: str):
    if not spec or spec == "none":
        return None
    if spec == "fake":
        from tests.fixtures.fake_crewai_llm import FakeReviewerLLM
        return FakeReviewerLLM()
    from crewai import LLM
    return LLM(model=spec)


class CrewForensicsExecutor(AgentExecutor):
    """Bridges an A2A request to the CrewAI ForensicsFlow."""

    def __init__(self, llm_spec: str = "none"):
        self.llm_spec = llm_spec

    def produce_report_json(self, claim: Dict[str, Any]) -> str:
        """Run CrewAI and return the report as JSON (tests override this to inject faults)."""
        return run_forensics(claim, llm=resolve_llm(self.llm_spec)).model_dump_json()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # a2a-sdk 1.x: the Task must be enqueued before any status update for it
        if context.current_task is None and context.message is not None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
            task_id, context_id = task.id, task.context_id
        else:
            task_id, context_id = context.task_id, context.context_id
        updater = TaskUpdater(event_queue, task_id, context_id)
        text = context.get_user_input() if context.message else ""
        claim_id = extract_claim_id(text)

        if not claim_id or not CLAIM_ID_PATTERN.match(claim_id):
            _log_call({"claim_id": claim_id, "outcome": "rejected_no_claim_id"})
            await updater.reject(new_text_message("Request must contain a valid \"claim_id\".",
                                                  context_id=context_id, task_id=task_id))
            return
        claim = load_claim(claim_id)
        if claim is None:
            _log_call({"claim_id": claim_id, "outcome": "rejected_unknown_claim"})
            await updater.reject(new_text_message(f"Unknown claim_id {claim_id!r}.",
                                                  context_id=context_id, task_id=task_id))
            return

        await updater.start_work()
        try:
            report_json = await asyncio.to_thread(self.produce_report_json, claim)
            report_data = json.loads(report_json)   # never let a bad report crash the task mid-flight
            if isinstance(report_data, dict) and report_data.get("trace_id") is None and "findings" in report_data:
                report_data["trace_id"] = REQUEST_TRACE_ID.get()
                report_json = json.dumps(report_data)
        except Exception as exc:
            _log_call({"claim_id": claim_id, "outcome": f"failed: {type(exc).__name__}"})
            await updater.failed(new_text_message(f"Forensic review failed: {type(exc).__name__}",
                                                  context_id=context_id, task_id=task_id))
            return

        _log_call({"claim_id": claim_id, "outcome": "completed"})
        await updater.add_artifact([new_text_part(FORENSICS_JSON_PREFIX + report_json),
                                    new_data_part(report_data)],
                                   name="forensics_report")
        await updater.complete(new_text_message(FORENSICS_JSON_PREFIX + report_json,
                                                context_id=context_id, task_id=task_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


def build_agent_card(url: str) -> AgentCard:
    return AgentCard(
        name=SERVICE_NAME,
        description=("CrewAI forensic review of a warranty claim dossier (serial consistency, purchase/incident "
                     "chronology, exclusion scan). Reports findings only; never approves or pays."),
        version="1.0.0",
        supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC")],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        skills=[AgentSkill(
            id="claim_forensics", name="Claim forensic review",
            description="Runs the CrewAI ForensicsFlow on a claim dossier and returns a ForensicsReport.",
            tags=["warranty", "forensics", "crewai"],
            examples=['{"claim_id": "CLM-2024-003"}'])],
    )


def create_app(url: str, executor: Optional[AgentExecutor] = None) -> Starlette:
    card = build_agent_card(url)
    handler = DefaultRequestHandler(agent_executor=executor or CrewForensicsExecutor(),
                                    task_store=InMemoryTaskStore(), agent_card=card)
    app = Starlette(routes=[*create_agent_card_routes(card, card_url="/.well-known/agent-card.json"),
                            *create_jsonrpc_routes(handler, "/", enable_v0_3_compat=True)])
    return TraceparentMiddleware(app)


def main(argv=None, executor_factory=None):
    parser = argparse.ArgumentParser(description="ClaimSleuth CrewAI forensics A2A service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--llm", default=os.environ.get("CLAIMSLEUTH_CREWAI_LLM", "none"))
    args, _ = parser.parse_known_args(argv)
    import uvicorn
    url = f"http://{args.host}:{args.port}/"
    executor = executor_factory(args) if executor_factory else CrewForensicsExecutor(llm_spec=args.llm)
    uvicorn.run(create_app(url, executor), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
