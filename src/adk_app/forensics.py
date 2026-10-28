"""
ADK side of the ADK <-> CrewAI interoperability over A2A.

build_forensics_agent() returns ADK's real RemoteA2aAgent pointed at the CrewAI
forensics service's Agent Card. Its context_builder sends one compact request: the
claim id plus the RAG policy citations already on the Blackboard (bb:rag), so the
CrewAI agent analyses the same policy evidence ADK retrieved. The after-agent
callback copies the reply, or the error, into the Blackboard; the deterministic
forensics_review_agent (steps.py) then validates it.

RemoteA2aAgent reports transport failures (service down, timeout, protocol error)
as an error event instead of raising, and a FAILED remote task still arrives as a
text message. So a reply only counts as a report if it carries the report prefix;
bb:a2a_call["ok"] is final only after ForensicsReviewAgent has validated the report.
"""
import json
import os

from google.adk.a2a.agent._remote_a2a_agent import RemoteA2aAgent
from google.genai import types as genai_types

from src.core.forensics_schema import FORENSICS_JSON_PREFIX, FORENSICS_REQUEST_PREFIX

DEFAULT_CREWAI_A2A_URL = "http://127.0.0.1:8101"
AGENT_NAME = "CrewAIForensics"
CITATION_FIELDS = ("topic", "chunk_id", "source_file", "section", "quote", "score")


def default_a2a_url() -> str:
    return os.environ.get("CLAIMSLEUTH_CREWAI_A2A_URL", DEFAULT_CREWAI_A2A_URL).rstrip("/")


def forensics_request(state) -> str:
    """The A2A request text: claim id + RAG citations (no claimant details, no documents)."""
    claim = state.get("bb:claim") or {}
    citations = ((state.get("bb:rag") or {}).get("citations") or {}).values()
    payload = {"claim_id": claim.get("claim_id"),
               "policy_evidence": [{k: c.get(k) for k in CITATION_FIELDS} for c in citations]}
    return FORENSICS_REQUEST_PREFIX + json.dumps(payload)


def _build_request(ctx, agent_name, part_converter):
    """RemoteA2aAgent context_builder: (message parts, context id)."""
    part = part_converter(genai_types.Part(text=forensics_request(ctx.session.state)))
    return [part], None


def _capture_reply(callback_context):
    """after_agent_callback: put the remote reply (or the transport error) on the Blackboard."""
    events = [e for e in callback_context._invocation_context.session.events if e.author == AGENT_NAME]
    last = events[-1] if events else None
    texts = [p.text for p in (last.content.parts if last and last.content else []) if p.text]
    meta = (last.custom_metadata or {}) if last else {}
    text = texts[0] if texts else None
    is_report = bool(text and text.startswith(FORENSICS_JSON_PREFIX))
    if text is None:
        error = last.error_message if last and last.error_message else "No reply from CrewAI service"
    elif not is_report:   # e.g. a failed remote task's status message: "Forensic review failed: ..."
        error = f"CrewAI service returned no forensics report: {text[:200]}"
    else:
        error = None
    # (bb:agent_sequence is recorded by ForensicsReviewAgent: writing it here could race in a parallel branch)
    callback_context.state["bb:forensics_raw"] = text
    callback_context.state["bb:forensics_error"] = error
    callback_context.state["bb:a2a_call"] = {
        "agent": AGENT_NAME,
        "task_id": meta.get("a2a:task_id"),
        "context_id": meta.get("a2a:context_id"),
        "reply": "report" if is_report else ("non_report_text" if text else "none"),
        "ok": False,                  # set by ForensicsReviewAgent once the report has passed validation
        "report_valid": None,
        "error": error,
    }


def build_forensics_agent(url: str, timeout: float = 150.0, httpx_client=None) -> RemoteA2aAgent:
    return RemoteA2aAgent(name=AGENT_NAME, agent_card=f"{url.rstrip('/')}/.well-known/agent-card.json",
                          description="CrewAI forensic review service reached over A2A",
                          timeout=timeout, httpx_client=httpx_client, after_agent_callback=_capture_reply,
                          context_builder=_build_request)
