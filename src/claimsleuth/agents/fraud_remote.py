"""
ADK side of the A2A link: the CrewAI fraud crew appears to ADK as one RemoteA2aAgent named fraud_crew.

The request carries the claim id, the claimant, the document ids, the facts on the blackboard and the
customer statement. The after-agent callback parses the crew's reply into Fraud and writes bb_fraud; a reply
that does not parse is recorded as a bad_output guardrail hit, which forces a full human review.
"""
import json

import httpx
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.genai import types
from pydantic import ValidationError

from claimsleuth import config
from claimsleuth.agents.common import (BB_CLAIM, BB_CREW_METRICS, BB_DOCUMENTS, BB_FACTS, BB_FRAUD, SECRET_HEADER,
                                       add_guardrail_hit, read)
from claimsleuth.models import Fraud, GuardrailHit
from claimsleuth.tracing import append_span_records, traceparent_headers

NAME = "fraud_crew"
DESCRIPTION = ("CrewAI fraud crew (Forensics analyst and History analyst) over A2A: checks the documents against each "
               "other and against past claims, and records fraud findings. Needs the facts.")


def request_text(state) -> str:
    claim = read(state, BB_CLAIM)
    body = {"claim_id": claim["id"], "claimant": claim["claimant"], "statement": claim["statement"],
            "documents": [d["doc_id"] for d in read(state, BB_DOCUMENTS)], "facts": read(state, BB_FACTS)}
    return f"Investigate claim {claim['id']}\n{json.dumps(body)}"


def _build_request(ctx, agent_name, part_converter):
    return [part_converter(types.Part(text=request_text(ctx.session.state)))], None


def _capture_reply(callback_context):
    events = [e for e in callback_context._invocation_context.session.events if e.author == NAME]
    last = events[-1] if events else None
    text = next((p.text for p in (last.content.parts if last and last.content else []) if p.text), None)
    state = callback_context.state
    try:
        reply = json.loads(text or "")
        fraud = Fraud.model_validate(reply["fraud"])
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
        detail = (text or (last.error_message if last and last.error_message else "no reply"))[:300]
        add_guardrail_hit(state, GuardrailHit(kind="bad_output", where=NAME, detail=f"unusable crew reply: {detail}"))
        return None
    state[BB_FRAUD] = fraud.model_dump_json()
    append_span_records(reply.get("spans") or [])
    state[BB_CREW_METRICS] = json.dumps(reply.get("metrics") or {})
    return None


class _TraceHeaders(httpx.Auth):
    """Adds the shared secret and the current W3C traceparent to every request to the crew."""

    def auth_flow(self, request):
        request.headers[SECRET_HEADER] = config.A2A_SHARED_SECRET
        request.headers.update(traceparent_headers())
        yield request


def build() -> RemoteA2aAgent:
    client = httpx.AsyncClient(auth=_TraceHeaders(), timeout=httpx.Timeout(600.0, connect=10.0))
    return RemoteA2aAgent(name=NAME, description=DESCRIPTION, agent_card=f"{config.CREW_URL}/.well-known/agent-card.json",
                          httpx_client=client, timeout=600.0, context_builder=_build_request,
                          after_agent_callback=_capture_reply)
