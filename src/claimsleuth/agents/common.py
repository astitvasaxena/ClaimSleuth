"""What every ADK agent shares: the model, MCP tools, prompt files, blackboard keys and the recording tools."""
import json
from decimal import Decimal
from pathlib import Path

from google.adk.models.google_llm import Gemini
from google.adk.planners import PlanReActPlanner
from google.adk.tools import ToolContext
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.genai import types

from claimsleuth import config
from claimsleuth.models import (Coverage, Facts, FinalReport, Fraud, GuardrailHit, ManagerSubmission, Payout,
                                review_level)

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_STYLES = ("direct", "react")
SECRET_HEADER = "X-ClaimSleuth-Secret"

# Blackboard keys (ADK session state). Values are JSON strings of the models in claimsleuth.models,
# so instructions can include them verbatim through {bb_...?} placeholders.
BB_CLAIM, BB_DOCUMENTS, BB_RUN = "bb_claim", "bb_documents", "bb_run"
BB_FACTS, BB_COVERAGE, BB_PAYOUT, BB_FRAUD = "bb_facts", "bb_coverage", "bb_payout", "bb_fraud"
BB_GUARDRAIL_HITS, BB_REPORT, BB_CREW_METRICS = "bb_guardrail_hits", "bb_report", "bb_crew_metrics"


def model() -> Gemini:
    return Gemini(model=config.GEMINI_MODEL, retry_options=types.HttpRetryOptions(
        attempts=config.MODEL_RETRY_ATTEMPTS, initial_delay=2.0, max_delay=30.0, exp_base=2.0,
        http_status_codes=[429, 500, 502, 503, 504]))


def mcp_tools(*names: str) -> McpToolset:
    """MCP tools from the app's /mcp endpoint, limited to the names an agent is allowed to use."""
    return McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=config.MCP_URL, headers={SECRET_HEADER: config.A2A_SHARED_SECRET}, timeout=60, sse_read_timeout=120),
        tool_filter=list(names))


def planner(style: str):
    """The react prompt style also uses ADK's PlanReActPlanner (plan, act, observe, replan)."""
    return PlanReActPlanner() if style == "react" else None


def prompt(style: str, name: str) -> str:
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {style!r}")
    return (PROMPTS_DIR / style / f"{name}.md").read_text(encoding="utf-8")


def read(state, key: str):
    raw = state.get(key)
    return json.loads(raw) if isinstance(raw, str) and raw else raw


def add_guardrail_hit(state, hit: GuardrailHit) -> None:
    hits = read(state, BB_GUARDRAIL_HITS) or []
    hits.append(hit.model_dump(mode="json"))
    state[BB_GUARDRAIL_HITS] = json.dumps(hits)


# ─── Recording tools: they only store a validated result on the blackboard ────────────────

def record_facts(facts: Facts, tool_context: ToolContext) -> dict:
    """Save the facts you extracted, each with its exact quote, and the list of missing items, to the blackboard."""
    facts = Facts.model_validate(facts)
    tool_context.state[BB_FACTS] = facts.model_dump_json()
    return {"saved": "bb_facts", "facts": len(facts.facts), "missing": facts.missing}


def record_coverage(coverage: Coverage, tool_context: ToolContext) -> dict:
    """Save your coverage decision, the quoted clause and the policy terms to the blackboard."""
    coverage = Coverage.model_validate(coverage)
    tool_context.state[BB_COVERAGE] = coverage.model_dump_json()
    return {"saved": "bb_coverage", "covered": coverage.covered}


def record_payout(payout: Payout, tool_context: ToolContext) -> dict:
    """Save the payout exactly as compute_payout returned it (amount and breakdown) to the blackboard."""
    payout = Payout.model_validate(payout)
    tool_context.state[BB_PAYOUT] = payout.model_dump_json()
    return {"saved": "bb_payout", "amount": str(payout.amount)}


def submit_report(submission: ManagerSubmission, tool_context: ToolContext) -> dict:
    """Submit the final decision with its reasons and supporting quotes. Call this exactly once, at the end."""
    sub = ManagerSubmission.model_validate(submission)
    state = tool_context.state
    payout = Payout.model_validate(read(state, BB_PAYOUT)) if state.get(BB_PAYOUT) else None
    coverage = Coverage.model_validate(read(state, BB_COVERAGE)) if state.get(BB_COVERAGE) else None
    fraud = Fraud.model_validate(read(state, BB_FRAUD)) if state.get(BB_FRAUD) else None
    if sub.decision == "approve" and payout is None:
        return {"error": "An approval needs a payout computed by compute_payout and recorded on the blackboard; "
                         "none is recorded. Get the settlement first, or choose another decision."}
    if sub.decision != "approve":
        payout = None
    elif sub.payout_amount is not None and Decimal(sub.payout_amount) != payout.amount:
        add_guardrail_hit(state, GuardrailHit(kind="bad_output", where="submit_report",
                                              detail=f"stated payout {sub.payout_amount} replaced by computed {payout.amount}"))
    run = read(state, BB_RUN) or {}
    hits = [GuardrailHit.model_validate(h) for h in (read(state, BB_GUARDRAIL_HITS) or [])]
    review = review_level(sub.decision, payout, fraud)
    if any(h.kind == "bad_output" and h.where == "fraud_crew" for h in hits):
        review = "full_review"                      # the fraud checks did not produce a usable result
    report = FinalReport(claim_id=read(state, BB_CLAIM)["id"], decision=sub.decision, payout=payout,
                         reasons=sub.reasons, evidence=sub.evidence, coverage=coverage, fraud=fraud,
                         guardrail_hits=hits, review=review,
                         pattern=run.get("pattern", ""), prompt_style=run.get("prompt_style", ""))
    state[BB_REPORT] = report.model_dump_json()
    tool_context.actions.skip_summarization = True
    return {"submitted": True, "decision": report.decision, "review": report.review}
