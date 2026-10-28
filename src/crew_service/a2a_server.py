"""
CrewAI forensics service exposed over the A2A protocol (a2a-sdk 1.3.0).

Run in the separate CrewAI environment:
    .venv-crewai/Scripts/python -m src.crew_service.a2a_server --port 8101
    Agent Card:  http://127.0.0.1:8101/.well-known/agent-card.json
    JSON-RPC:    http://127.0.0.1:8101/

A request is an A2A message whose text contains the claim id: either ADK's request
(FORENSICS_REQUEST_PREFIX + {"claim_id", "policy_evidence"}) or any text with a
"claim_id". The service loads its OWN copy of the claim from the dossier, runs the
CrewAI ForensicsFlow (deterministic checks, then the LLM-backed CrewAI Crew), and
returns a ForensicsReport artifact. It never returns a payout or an approval: the
schema forbids such fields.

LLM for the CrewAI agent (--llm or CLAIMSLEUTH_CREWAI_LLM; see llm_review.resolve_llm_config):
    auto (default)   Gemini through CrewAI's OpenAI-compatible provider if GOOGLE_API_KEY or
                     GEMINI_API_KEY is in this process's environment; otherwise LLM_NOT_CONFIGURED
    none             no LLM stage (reported as LLM_NOT_CONFIGURED)
    fake[:<mode>]    TEST DOUBLE (tests/fixtures/fake_crewai_llm.py)
    openrouter/<model id>   explicit opt-in: OpenRouter, needs OPENROUTER_API_KEY
Configuration comes from this process's environment only (never from the caller or the
browser). Beware: crewai calls load_dotenv() on import, so a service started from the
project root already has the root .env loaded. --env-file PATH loads only the keys in
ENV_FILE_KEYS from another file (values are never printed or logged).
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
from typing import Any, Dict, List, Optional

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

from src.core.forensics_schema import FORENSICS_JSON_PREFIX, FORENSICS_REQUEST_PREFIX, ForensicsReport
from src.crew_service.llm_review import resolve_llm_config
from src.tools.evidence_store import EvidenceStore

SERVICE_NAME = "ClaimSleuthCrewAIForensics"
ENV_FILE_KEYS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL", "OPENROUTER_API_KEY", "CLAIMSLEUTH_CREWAI_MODEL",
                 "CLAIMSLEUTH_CREWAI_LLM", "CLAIMSLEUTH_CREWAI_LLM_TIMEOUT")
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


def extract_policy_evidence(text: str) -> List[Dict[str, Any]]:
    """ADK's RAG citations sent with the request (checked again against the corpus before use)."""
    if not (text or "").startswith(FORENSICS_REQUEST_PREFIX):
        return []
    try:
        payload = json.loads(text[len(FORENSICS_REQUEST_PREFIX):])
    except json.JSONDecodeError:
        return []
    evidence = payload.get("policy_evidence") if isinstance(payload, dict) else None
    return [e for e in evidence if isinstance(e, dict)][:10] if isinstance(evidence, list) else []


def load_env_file(path: str, already_set: Optional[List[str]] = None) -> List[str]:
    """
    Load only ENV_FILE_KEYS from a dotenv-style file into this process. Returns the key NAMES loaded;
    names that were already set (and so left unchanged) are appended to `already_set`.
    Note: crewai itself calls load_dotenv() on import (crewai/llm.py), which loads a .env found from
    the working directory; started from the project root, the root .env is therefore already loaded.
    """
    loaded = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if not match or line.lstrip().startswith("#") or match.group(1) not in ENV_FILE_KEYS:
            continue
        if match.group(1) not in os.environ:          # an explicitly exported variable wins
            os.environ[match.group(1)] = match.group(2).strip("'\"")
            loaded.append(match.group(1))
        elif already_set is not None:
            already_set.append(match.group(1))
    return loaded


def _llm_log_fields(report_data: Dict[str, Any]) -> Dict[str, Any]:
    review = report_data.get("llm_review") or {}
    return {"crewai_flow_executed": report_data.get("crewai_flow_executed"),
            "crewai_crew_executed": review.get("crewai_crew_executed"),
            "llm_status": review.get("status"), "llm_error_category": review.get("error_category"),
            "llm_provider": review.get("llm_provider"), "llm_model": review.get("llm_model"),
            "llm_call_attempts": review.get("llm_call_attempts"), "llm_calls_succeeded": review.get("llm_calls_succeeded"),
            "prompt_tokens": review.get("prompt_tokens"), "completion_tokens": review.get("completion_tokens"),
            "llm_findings": len(review.get("findings") or []), "policy_evidence_origin": review.get("policy_evidence_origin")}


class CrewForensicsExecutor(AgentExecutor):
    """Bridges an A2A request to the CrewAI ForensicsFlow."""

    def __init__(self, llm_spec: Optional[str] = None):
        self.llm_spec = llm_spec

    def produce_report_json(self, claim: Dict[str, Any], policy_evidence: Optional[List[Dict[str, Any]]] = None) -> str:
        """Run CrewAI and return the report as JSON (tests override this to inject faults)."""
        config = resolve_llm_config(self.llm_spec)    # per request: a fresh LLM object and fresh call counters
        return run_forensics(claim, llm_config=config, policy_evidence=policy_evidence).model_dump_json()

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
            report_json = await asyncio.to_thread(self.produce_report_json, claim, extract_policy_evidence(text))
            report_data = json.loads(report_json)   # never let a bad report crash the task mid-flight
            if isinstance(report_data, dict) and report_data.get("trace_id") is None and "findings" in report_data:
                report_data["trace_id"] = REQUEST_TRACE_ID.get()
                report_json = json.dumps(report_data)
        except Exception as exc:
            _log_call({"claim_id": claim_id, "outcome": f"failed: {type(exc).__name__}"})
            await updater.failed(new_text_message(f"Forensic review failed: {type(exc).__name__}",
                                                  context_id=context_id, task_id=task_id))
            return

        _log_call({"claim_id": claim_id, "outcome": "completed",
                   **(_llm_log_fields(report_data) if isinstance(report_data, dict) else {})})
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
        description=("CrewAI forensic review of a warranty claim dossier: deterministic checks (serial consistency, "
                     "purchase/incident chronology, exclusion scan) in a CrewAI Flow, then an LLM-backed CrewAI "
                     "agent that cites the evidence. Reports findings only; never approves or pays."),
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
    parser.add_argument("--llm", default=None, help="auto (default) | none | fake[:mode] | <provider/model>")
    parser.add_argument("--env-file", default=None, help="load only the LLM keys in ENV_FILE_KEYS from this file")
    args, _ = parser.parse_known_args(argv)
    if args.env_file:
        already: List[str] = []
        loaded = load_env_file(args.env_file, already)
        print(f"[crew-service] --env-file {args.env_file}: loaded {', '.join(loaded) or 'nothing'}; "
              f"already in the environment (unchanged): {', '.join(already) or 'none'}", file=sys.stderr)
    args.llm = args.llm or os.environ.get("CLAIMSLEUTH_CREWAI_LLM", "auto")
    config = resolve_llm_config(args.llm)
    print(f"[crew-service] CrewAI agent LLM: provider={config.provider} model={config.model} "
          f"configured={config.configured}" + (f" ({config.not_configured_reason})" if not config.configured else ""),
          file=sys.stderr)
    import uvicorn
    url = f"http://{args.host}:{args.port}/"
    executor = executor_factory(args) if executor_factory else CrewForensicsExecutor(llm_spec=args.llm)
    uvicorn.run(create_app(url, executor), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
