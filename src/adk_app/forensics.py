"""
ADK side of the ADK <-> CrewAI interoperability over A2A.

build_forensics_agent() returns ADK's real RemoteA2aAgent pointed at the CrewAI
forensics service's Agent Card. It sends the conversation (whose first message is
the claim JSON) as an A2A JSON-RPC request over HTTP and turns the reply into an ADK
event. The after-agent callback copies that reply, or the error, into the
Blackboard; the deterministic forensics_review_agent (steps.py) then validates it.

RemoteA2aAgent reports transport failures (service down, timeout, protocol error)
as an error event instead of raising, so the pipeline always continues to the
verifier, which fails closed when the forensic review is missing or invalid.
"""
import os

from google.adk.a2a.agent._remote_a2a_agent import RemoteA2aAgent

DEFAULT_CREWAI_A2A_URL = "http://127.0.0.1:8101"
AGENT_NAME = "CrewAIForensics"


def default_a2a_url() -> str:
    return os.environ.get("CLAIMSLEUTH_CREWAI_A2A_URL", DEFAULT_CREWAI_A2A_URL).rstrip("/")


def _capture_reply(callback_context):
    """after_agent_callback: put the remote reply (or the transport error) on the Blackboard."""
    events = [e for e in callback_context._invocation_context.session.events if e.author == AGENT_NAME]
    last = events[-1] if events else None
    texts = [p.text for p in (last.content.parts if last and last.content else []) if p.text]
    meta = (last.custom_metadata or {}) if last else {}
    # (bb:agent_sequence is recorded by ForensicsReviewAgent: writing it here could race in a parallel branch)
    callback_context.state["bb:forensics_raw"] = texts[0] if texts else None
    callback_context.state["bb:forensics_error"] = (
        None if texts else (last.error_message if last and last.error_message else "No reply from CrewAI service"))
    callback_context.state["bb:a2a_call"] = {
        "agent": AGENT_NAME,
        "task_id": meta.get("a2a:task_id"),
        "context_id": meta.get("a2a:context_id"),
        "ok": bool(texts),
        "error": None if texts else callback_context.state["bb:forensics_error"],
    }


def build_forensics_agent(url: str, timeout: float = 120.0, httpx_client=None) -> RemoteA2aAgent:
    return RemoteA2aAgent(name=AGENT_NAME, agent_card=f"{url.rstrip('/')}/.well-known/agent-card.json",
                          description="CrewAI forensic review service reached over A2A",
                          timeout=timeout, httpx_client=httpx_client, after_agent_callback=_capture_reply)
