"""
A2A service around the CrewAI fraud crew. Run it in the CrewAI environment:

    .venv-crewai/Scripts/python -m claimsleuth.crew.a2a_server --port 8101

Every request except the agent card needs the X-ClaimSleuth-Secret header. The request text starts with
"Investigate claim <id>" followed by a JSON object (claim id, claimant, document ids, facts, statement).
The reply is JSON: {"fraud": <Fraud>, "metrics": {...}}.
"""
import argparse
import asyncio
import contextvars
import hmac
import json
import re
import sys
from datetime import timedelta

from a2a.helpers import new_task_from_user_message, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT
from starlette.applications import Starlette

from claimsleuth import config
from claimsleuth.crew.crew import SECRET_HEADER, run_crew

AGENT_CARD_PATH = "/.well-known/agent-card.json"
CLAIM_ID_RE = re.compile(r"Investigate claim ([A-Za-z0-9][A-Za-z0-9_-]{2,60})")
REQUEST_TRACEPARENT: contextvars.ContextVar = contextvars.ContextVar("request_traceparent", default="")


class GuardMiddleware:
    """Refuse requests without the shared secret (the agent card stays public); read the caller's traceparent."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        secret = headers.get(SECRET_HEADER.lower(), "")
        if scope["path"] not in (AGENT_CARD_PATH, "/healthz") and not (
                config.A2A_SHARED_SECRET and hmac.compare_digest(secret, config.A2A_SHARED_SECRET)):
            body = json.dumps({"detail": f"missing or wrong {SECRET_HEADER} header"}).encode()
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})
            return
        token = REQUEST_TRACEPARENT.set(headers.get("traceparent", ""))
        try:
            await self.app(scope, receive, send)
        finally:
            REQUEST_TRACEPARENT.reset(token)


def parse_request(text: str) -> dict:
    match = CLAIM_ID_RE.search(text or "")
    if not match:
        raise ValueError('request must start with "Investigate claim <id>"')
    body = text[match.end():].strip()
    request = json.loads(body) if body.startswith("{") else {}
    request["claim_id"] = match.group(1)
    return request


async def fetch_documents(doc_ids: list[str]) -> dict[str, dict[int, str]]:
    """Read the claim's documents through the MCP server once, for the task guardrail's quote check."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    documents = {}
    async with streamablehttp_client(config.MCP_URL, headers={SECRET_HEADER: config.A2A_SHARED_SECRET},
                                     timeout=timedelta(seconds=60)) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for doc_id in doc_ids:
                result = await session.call_tool("parse_document", {"doc_id": doc_id})
                if not result.isError and result.structuredContent:
                    documents[doc_id] = {l["n"]: l["text"] for l in result.structuredContent["lines"]}
    return documents


class CrewExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is None and context.message is not None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
            task_id, context_id = task.id, task.context_id
        else:
            task_id, context_id = context.task_id, context.context_id
        updater = TaskUpdater(event_queue, task_id, context_id)
        try:
            request = parse_request(context.get_user_input() if context.message else "")
        except (ValueError, json.JSONDecodeError) as exc:
            await updater.reject(new_text_message(f"Bad request: {exc}", context_id=context_id, task_id=task_id))
            return
        await updater.start_work()
        try:
            documents = await fetch_documents(request.get("documents", []))
            reply = await asyncio.to_thread(run_crew, request, documents, REQUEST_TRACEPARENT.get())
        except Exception as exc:                                     # reported to ADK, which records bad_output
            print(f"[crew] {request['claim_id']} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            await updater.failed(new_text_message(f"Fraud crew failed: {type(exc).__name__}: {str(exc)[:300]}",
                                                  context_id=context_id, task_id=task_id))
            return
        print(f"[crew] {request['claim_id']} flagged={reply['fraud']['flagged']} "
              f"findings={len(reply['fraud']['findings'])} seconds={reply['metrics']['seconds']}", file=sys.stderr)
        await updater.complete(new_text_message(json.dumps(reply), context_id=context_id, task_id=task_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def agent_card(url: str) -> AgentCard:
    return AgentCard(
        name="ClaimSleuthFraudCrew", version="2.0.0",
        description="CrewAI crew (Forensics analyst, History analyst) that checks a warranty claim for fraud signs "
                    "and returns cited findings. It never approves or pays.",
        supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC", protocol_version=PROTOCOL_VERSION_CURRENT)],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"], default_output_modes=["application/json"],
        skills=[AgentSkill(id="fraud_checks", name="Fraud checks", tags=["warranty", "fraud", "crewai"],
                           description="Cross-document and claim-history checks for one claim.",
                           examples=['Investigate claim CLM-2024-003 {"claim_id": "CLM-2024-003", ...}'])])


def create_app(url: str):
    card = agent_card(url)
    handler = DefaultRequestHandler(agent_executor=CrewExecutor(), task_store=InMemoryTaskStore(), agent_card=card)
    routes = [*create_agent_card_routes(card, card_url=AGENT_CARD_PATH),
              *create_jsonrpc_routes(handler, "/", enable_v0_3_compat=True)]

    async def healthz(request):
        from starlette.responses import JSONResponse
        return JSONResponse({"status": "ok", "model": config.GEMINI_MODEL})
    from starlette.routing import Route
    return GuardMiddleware(Starlette(routes=[Route("/healthz", healthz), *routes]))


def main():
    parser = argparse.ArgumentParser(description="ClaimSleuth CrewAI fraud crew over A2A")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--public-url", default=None, help="URL in the agent card (default http://host:port/)")
    args = parser.parse_args()
    if not config.GOOGLE_API_KEY or not config.A2A_SHARED_SECRET:
        sys.exit("GOOGLE_API_KEY and A2A_SHARED_SECRET must be set")
    print(f"[crew] model={config.GEMINI_MODEL} mcp={config.MCP_URL}", file=sys.stderr)
    import uvicorn
    uvicorn.run(create_app(args.public_url or f"http://{args.host}:{args.port}/"), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
