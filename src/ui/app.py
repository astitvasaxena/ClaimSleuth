import os
import re
import sys
import json
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Literal

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.state import ClaimInvestigationState
from src.core.audit import AuditLogger, verify_log_file
from src.core.adjudication import payout_view
from src.adk_app.pipeline import ADK_EXECUTION_STATUS, resolve_model, run_adk_investigation
from src.adk_app.tool_gateway import McpConfig, MCPUnavailableError
from src.adk_app.forensics import default_a2a_url
from src.core.tracing import read_trace
from src.adk_app.report import FinalReport, write_report_files
from src.agents.investigator import LeadInvestigatorAgent
from src.tools.evidence_store import EvidenceStore
from configs.settings import (
    ADJUDICATION_TOKEN, CORS_ORIGINS, USE_LLM, GEMINI_MODEL, LEGACY_AUDIT_LOG_DIR, REPORTS_DIR
)

app = FastAPI(title="ClaimSleuth: Agentic AI Warranty Investigation System")

# Only configured origins may call the API cross-origin (CLAIMSLEUTH_CORS_ORIGINS).
# No cookies are used, so credentials are not allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

evidence_store = EvidenceStore()
investigator = LeadInvestigatorAgent(evidence_store=evidence_store)

# In-memory storage for demo sessions
active_states: Dict[str, ClaimInvestigationState] = {}
# Which engine produced the claim's current result: {"source": "legacy"|"adk", "run_id", "report"}
active_runs: Dict[str, Dict[str, Any]] = {}

SCENARIO_MAP = {
    "scenario_1_valid": {
        "id": "scenario_1_valid",
        "title": "Scenario 1: Legitimate iPhone Repair",
        "badge": "Happy Path (Auto-Verified)",
        "badge_color": "emerald",
        "summary": "Valid iPhone 15 Pro cracked screen under active TechGuard warranty. Repair cost $280.",
        "expected": "Dual-Gate Verifier approves $251.00 settlement. Pending 1-click human sign-off."
    },
    "scenario_2_expired": {
        "id": "scenario_2_expired",
        "title": "Scenario 2: Expired Laptop Warranty",
        "badge": "Deterministic Rejection",
        "badge_color": "rose",
        "summary": "OmniLaptop power IC failure claimed 3 months after 12-month warranty expired.",
        "expected": "Deterministic date engine halts approval. Zero LLM hallucination override."
    },
    "scenario_3_conflict": {
        "id": "scenario_3_conflict",
        "title": "Scenario 3: Serial Number Conflict",
        "badge": "Escalated Conflict / Potential Fraud",
        "badge_color": "amber",
        "summary": "Refrigerator compressor repair bill contains a serial number differing from purchase invoice.",
        "expected": "Anomaly detector & Investigator flag conflict, abort automation, route to human."
    }
}

class AdjudicationRequest(BaseModel):
    decision: Literal["APPROVED", "REJECTED", "INFO_REQUESTED"]
    notes: Optional[str] = ""

# Human decisions each status may receive. Anything else (including every CLOSED_*
# status) is refused with 409. APPROVED is only possible after the verifier passed.
ALLOWED_DECISIONS: Dict[str, set] = {
    "VERIFIED_PENDING_APPROVAL": {"APPROVED", "REJECTED", "INFO_REQUESTED"},
    "REJECTED_DETERMINISTICALLY": {"REJECTED", "INFO_REQUESTED"},
    "CONFLICT_ESCALATED": {"REJECTED", "INFO_REQUESTED"},
    "NEEDS_EVIDENCE": {"REJECTED", "INFO_REQUESTED"},
    "BLOCKED_BY_GUARDRAIL": {"REJECTED", "INFO_REQUESTED"},
}

DECISION_TO_STATUS = {
    "APPROVED": "CLOSED_SETTLED",
    "REJECTED": "CLOSED_REJECTED",
    "INFO_REQUESTED": "NEEDS_EVIDENCE",
}


def permitted_decisions(state: ClaimInvestigationState) -> list:
    """
    The human decisions /api/adjudicate would accept for this state right now (the same rules as
    its guard). Read-only and informational: the dashboard uses it to show only valid controls,
    and the endpoint still checks every decision itself.
    """
    allowed = ALLOWED_DECISIONS.get(state.status, set())
    approvable = bool(state.verifier_report and state.verifier_report.is_approved_by_verifier
                      and state.deterministic_settlement is not None)
    return [d for d in ("APPROVED", "REJECTED", "INFO_REQUESTED")
            if d in allowed and (d != "APPROVED" or approvable)]

AUDIT_RESPONSE_LIMIT = 50  # entries returned with an adjudication response


def _adk_version() -> Optional[str]:
    try:
        import google.adk
        return google.adk.__version__
    except Exception:
        return None


def adk_dashboard_view(result) -> Dict[str, Any]:
    """
    Present an ADK run in the shapes the dashboard already renders:
    - state.steps_history: one entry per agent that actually ran, with the tools it called;
    - blackboard.facts: the bb:* session-state keys; blackboard.messages: the ADK
      state-change events (labelled STATE_DELTA; these are not A2A messages);
    - monitoring: counts only. ADK step latency is not measured, so it is null.
    """
    from src.core.state import InvestigationStep

    report, bb = result.report, result.state
    calls_by_agent: Dict[str, list] = {}
    for c in report.tool_calls:
        calls_by_agent.setdefault(c.get("agent", "?"), []).append(c["tool"])
    for c in report.model_tool_calls:
        calls_by_agent.setdefault("InvestigatorLLM", []).append(c["tool"])
    phase_for = {"VerifierAgent": "ASSESS", "ReportAgent": "STOP"}
    steps_history = [
        InvestigationStep(
            step_num=i, phase=phase_for.get(agent, "EXECUTE"),
            description=f"{agent} (Google ADK {report.pipeline_backend}, tools: {report.tool_backend})",
            tool_called=", ".join(dict.fromkeys(calls_by_agent.get(agent, []))) or None,
            reasoning=None).model_dump()
        for i, agent in enumerate(report.agent_sequence, start=1)
    ]
    state = result.claim_state.model_dump()
    state["steps_history"] = steps_history

    facts = {k: {"value": v, "source_agent": "ADK session state"}
             for k, v in bb.items() if k.startswith("bb:") and k not in ("bb:final_report", "bb:report_files")}
    messages = [{"sender": e["author"], "receiver": "session.state", "intent": "STATE_DELTA",
                 "payload": {"node": e["node_path"], "keys": e["state_keys"]}} for e in result.events]
    return {
        "state": state,
        "blackboard": {"case_id": report.claim_id, "fact_count": len(facts), "facts": facts,
                       "message_count": len(messages), "messages": messages, "replan_count": 0},
        "monitoring": {"case_id": report.claim_id, "steps_executed": len(report.agent_sequence),
                       "tools_called": len(report.tool_calls) + len(report.model_tool_calls),
                       "replans_triggered": 0, "error_count": len(report.tool_errors),
                       "avg_step_latency_ms": None},
    }


def runtime_status() -> Dict[str, Any]:
    """What is actually configured and running — never hard-coded 'Active' labels."""
    try:
        gemini_sdk = importlib.util.find_spec("google.generativeai") is not None
    except ModuleNotFoundError:
        gemini_sdk = False
    gemini_active = investigator.llm_client is not None
    return {
        "planner_mode": "gemini" if gemini_active else "deterministic_fallback",
        "extractor_mode": "gemini_with_regex_fallback" if gemini_active else "regex",
        "gemini": {
            "requested": USE_LLM,
            "sdk_installed": gemini_sdk,
            "api_key_configured": bool(os.environ.get("GOOGLE_API_KEY")),
            "active": gemini_active,
            "model": GEMINI_MODEL if gemini_active else None,
        },
        "mcp": {
            "pipelines": "in_process: /api/investigate calls the legacy "
                         "src/mcp/tool_server.py object directly (no transport)",
            "remote_server": "available: python -m src.mcp.remote_server --transport stdio|http "
                             "(6 policy-grounded tools; tested over both transports)",
            "remote_server_used_by_pipelines": "adk_only: /api/adk/investigate/{scenario}?tools=mcp_stdio|mcp_http "
                                               "(ADK McpToolset); the original loop still uses the in-process server",
            "remote_server_auth": "none: localhost-only; not safe to expose publicly",
        },
        "google_adk": {
            "installed_version": _adk_version(),
            "main_pipeline": "not_used: /api/investigate runs the original LeadInvestigatorAgent loop",
            "adk_pipeline_endpoint": "/api/adk/investigate/{scenario_id}",
            "executed_in_this_process": ADK_EXECUTION_STATUS["runs_completed"] > 0,
            **ADK_EXECUTION_STATUS,
        },
        # Configured != live-tested: a live Gemini call is only proven by tests/live.
        "gemini_configured_for_adk": resolve_model(None, None) is not None,
        "crewai": {
            "forensics_service": "real crewai in a separate environment (.venv-crewai): a Flow of deterministic "
                                 "checks, then an LLM-backed Agent/Task/Crew (python -m src.crew_service.a2a_server "
                                 "--env-file .env); reached from ADK over A2A",
            "llm_stage": "reported per run in report.forensics.llm_review (status, provider/model, Crew executed, "
                         "LLM calls, tokens); LLM_NOT_CONFIGURED when the service has no API key",
            "used_by": "/api/adk/investigate/{scenario}?forensics=crewai_a2a",
        },
        "a2a": {
            "implementation": "ADK RemoteA2aAgent (a2a-sdk 1.3.0) -> CrewAI service Agent Card + JSON-RPC over HTTP",
            "service_url": default_a2a_url(),
            "calls_ok_in_this_process": ADK_EXECUTION_STATUS["crewai_a2a_calls_ok"],
            "calls_failed_in_this_process": ADK_EXECUTION_STATUS["crewai_a2a_calls_failed"],
            "authentication": "none: localhost-only; not safe to expose publicly",
            "blackboard_messages": "the in-memory A2AMessage log in src/core/blackboard.py is NOT the A2A protocol",
        },
        "adjudication_auth": "token_required" if ADJUDICATION_TOKEN else "disabled_demo_mode",
    }


@app.get("/health")
def health():
    return {"status": "ok", "planner_mode": runtime_status()["planner_mode"]}

@app.get("/api/runtime-status")
def get_runtime_status():
    return runtime_status()

@app.get("/api/scenarios")
def get_scenarios():
    return list(SCENARIO_MAP.values())

@app.get("/api/scenario/{scenario_id}/docs")
def get_scenario_docs(scenario_id: str):
    folder = PROJECT_ROOT / "data" / "sample_claims" / scenario_id
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Scenario not found")
        
    docs = {}
    claim_meta = {}
    for f in folder.iterdir():
        if f.is_file():
            if f.name == "claim.json":
                with open(f, "r", encoding="utf-8") as jf:
                    claim_meta = json.load(jf)
            else:
                with open(f, "r", encoding="utf-8", errors="replace") as tf:
                    docs[f.name] = tf.read()
    return {"claim_meta": claim_meta, "documents": docs}

@app.post("/api/investigate/{scenario_id}")
def run_investigation(scenario_id: str):
    folder = PROJECT_ROOT / "data" / "sample_claims" / scenario_id
    claim_json = folder / "claim.json"
    if not claim_json.exists():
        raise HTTPException(status_code=404, detail="Claim definition not found")
        
    with open(claim_json, "r", encoding="utf-8") as f:
        claim_data = json.load(f)
        
    initial_state = ClaimInvestigationState(**claim_data)
    # Remember where this run starts so the response carries only its own entries
    entries_before = AuditLogger(initial_state.claim_id).count_entries()
    result_state = investigator.run_investigation(initial_state)
    active_states[result_state.claim_id] = result_state
    active_runs[result_state.claim_id] = {"source": "legacy", "run_id": None, "report": None}

    audit_logger = AuditLogger(result_state.claim_id)
    audit_trail = audit_logger.get_trail()[entries_before:]

    bb_summary = investigator.get_last_blackboard_summary(result_state.claim_id)
    monitor_summary = investigator.get_last_monitoring_summary(result_state.claim_id)

    return {
        "state": result_state.model_dump(),
        "payout": payout_view(result_state),
        "audit_trail": audit_trail,
        "blackboard": bb_summary,
        "monitoring": monitor_summary,
        "mcp_tools": ["check_coverage_window", "compute_settlement", "flag_serial_anomalies"],
        "runtime": runtime_status(),
        "allowed_decisions": permitted_decisions(result_state),
    }

@app.post("/api/adk/investigate/{scenario_id}")
def run_adk_pipeline(scenario_id: str, backend: Literal["workflow", "sequential"] = "workflow",
                     tools: Literal["local", "mcp_stdio", "mcp_http"] = "local",
                     forensics: Literal["none", "crewai_a2a"] = "none",
                     pattern: Literal["sequential", "parallel", "hierarchical"] = "sequential",
                     messaging: Literal["blackboard", "direct", "broadcast"] = "blackboard"):
    """
    Run the claim through the Google ADK pipeline (real Runner + session state).
    tools=mcp_stdio spawns the ClaimSleuth MCP server for the run; tools=mcp_http uses
    the server at CLAIMSLEUTH_MCP_URL (default http://127.0.0.1:8102/mcp). If the MCP
    server is unavailable the run is aborted with 503 and nothing is produced.
    forensics=crewai_a2a calls the CrewAI forensics service (CLAIMSLEUTH_CREWAI_A2A_URL,
    default http://127.0.0.1:8101) over A2A; if it is down or misbehaves the claim fails
    closed (NEEDS_EVIDENCE / escalated), it is never approved on CrewAI's word.
    """
    claim_json = PROJECT_ROOT / "data" / "sample_claims" / scenario_id / "claim.json"
    if not claim_json.exists():
        raise HTTPException(status_code=404, detail="Claim definition not found")
    with open(claim_json, "r", encoding="utf-8") as f:
        claim_data = json.load(f)

    entries_before = AuditLogger(claim_data["claim_id"]).count_entries()
    try:
        if tools == "local":
            result = run_adk_investigation(claim_data, backend=backend, forensics=forensics,
                                           pattern=pattern, messaging=messaging)
        else:
            transport = "stdio" if tools == "mcp_stdio" else "http"
            result = run_adk_investigation(claim_data, backend=backend, tools="mcp", forensics=forensics,
                                           pattern=pattern, messaging=messaging,
                                           mcp=McpConfig(transport=transport))
    except MCPUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"MCP tool server unavailable; no investigation result: {exc}")
    except ValueError as exc:   # e.g. pattern not supported by the chosen backend
        raise HTTPException(status_code=422, detail=str(exc))

    # Register the run so the SAME validated /api/adjudicate logic applies to it.
    claim_state = result.claim_state
    active_states[claim_state.claim_id] = claim_state
    active_runs[claim_state.claim_id] = {"source": "adk", "run_id": result.report.run_id, "report": result.report}

    view = adk_dashboard_view(result)
    return {
        # Same top-level fields as /api/investigate, so the dashboard can render either engine
        "state": view["state"],
        "payout": payout_view(claim_state),
        "audit_trail": AuditLogger(claim_state.claim_id).get_trail()[entries_before:],
        "blackboard": view["blackboard"],
        "monitoring": view["monitoring"],
        "runtime": runtime_status(),
        "allowed_decisions": permitted_decisions(claim_state),
        "investigation_source": "adk",
        # ADK-specific detail
        "report": result.report.model_dump(mode="json"),
        "report_files": result.files,
        "trace": result.events,
        "model_error": result.model_error,
        "blackboard_keys": sorted(k for k in result.state if k.startswith("bb:")),
    }

@app.get("/api/blackboard/{claim_id}")
def get_blackboard_data(claim_id: str):
    """Retrieve Blackboard shared state and A2A message bus for a claim."""
    summary = investigator.get_last_blackboard_summary(claim_id)
    if not summary:
        raise HTTPException(status_code=404, detail="No active blackboard found for claim")
    return summary

@app.get("/api/monitoring/{claim_id}")
def get_monitoring_data(claim_id: str):
    """Retrieve step-by-step latency and tool execution metrics."""
    summary = investigator.get_last_monitoring_summary(claim_id)
    if not summary:
        raise HTTPException(status_code=404, detail="No monitoring metrics found for claim")
    return summary

def _check_adjudication_auth(request: Request):
    """Enforce Bearer token when CLAIMSLEUTH_ADJUDICATION_TOKEN is configured."""
    if not ADJUDICATION_TOKEN:
        # Demo mode: no token configured, allow but warn
        return
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header (Bearer token required)")
    token = auth_header[len("Bearer "):]
    if token != ADJUDICATION_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid adjudication token")

@app.post("/api/adjudicate/{claim_id}")
def adjudicate_claim(claim_id: str, req: AdjudicationRequest, request: Request):
    _check_adjudication_auth(request)

    if claim_id not in active_states:
        raise HTTPException(status_code=404, detail="Claim not found in active session")
        
    state = active_states[claim_id]
    audit = AuditLogger(claim_id)

    # State-transition guard: refuse decisions the current status does not permit
    allowed = ALLOWED_DECISIONS.get(state.status, set())
    block_reason = None
    if req.decision not in allowed:
        block_reason = f"Decision {req.decision} is not permitted when claim status is {state.status}"
    elif req.decision == "APPROVED" and not (
        state.verifier_report and state.verifier_report.is_approved_by_verifier
        and state.deterministic_settlement is not None
    ):
        block_reason = "Approval requires a verifier-approved investigation with a deterministic settlement"
    if block_reason:
        audit.log_human_decision_blocked(req.decision, state.status, block_reason)
        raise HTTPException(status_code=409, detail=block_reason)

    state.human_decision = req.decision
    state.human_notes = req.notes
    state.status = DECISION_TO_STATUS[req.decision]
    audit.log_human_decision(req.decision, req.notes)

    run = active_runs.get(claim_id, {"source": "legacy", "run_id": None, "report": None})
    adjudicated_files = None
    if run["source"] == "adk":
        # The investigation report stays as written; the decision gets its own record.
        decided = FinalReport(**{
            **run["report"].model_dump(),
            "status": state.status,
            "payout": payout_view(state),
            "human_decision": req.decision,
            "human_notes": req.notes,
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "audit_chain_valid": audit.verify_chain()["valid"],
        })
        adjudicated_files = write_report_files(decided, REPORTS_DIR, suffix="_adjudicated")

    return {
        "status": state.status,
        "human_decision": state.human_decision,
        "payout": payout_view(state),
        "allowed_decisions": permitted_decisions(state),
        "audit_trail": audit.get_trail()[-AUDIT_RESPONSE_LIMIT:],
        "investigation_source": run["source"],
        "run_id": run["run_id"],
        "adjudicated_report_files": adjudicated_files,
    }

@app.get("/api/audit/{claim_id}")
def get_audit_trail(claim_id: str, limit: int = Query(200, ge=1, le=5000)):
    """Most recent `limit` ledger entries (oldest first). Full history stays on disk."""
    audit = AuditLogger(claim_id)
    return audit.get_trail()[-limit:]

@app.get("/api/audit/{claim_id}/verify")
def verify_audit_trail(claim_id: str):
    """Recompute the SHA-256 hash chain for the claim's ledger."""
    return AuditLogger(claim_id).verify_chain()

@app.get("/api/trace/{trace_id}")
def get_trace(trace_id: str, limit: int = Query(500, ge=1, le=5000)):
    """Spans recorded for one investigation (ADK nodes, MCP calls, the root span), by W3C trace id."""
    if not re.fullmatch(r"[0-9a-f]{32}", trace_id):
        raise HTTPException(status_code=422, detail="trace_id must be 32 lowercase hex characters")
    spans = read_trace(trace_id, limit)
    if not spans:
        raise HTTPException(status_code=404, detail="No spans recorded for this trace id")
    return {"trace_id": trace_id, "span_count": len(spans), "spans": spans}

@app.get("/api/audit-legacy")
def legacy_audit_status():
    """
    Read-only report on the ledgers committed before the D-3 fix (data/audit_logs/).
    The classification comes from actually re-running verification, never a fixed label.
    """
    results = []
    for log_file in sorted(LEGACY_AUDIT_LOG_DIR.glob("*_audit_trail.jsonl")):
        check = verify_log_file(log_file)
        results.append({
            "file": log_file.name,
            "classification": "LEGACY_VALID" if check["valid"] else "LEGACY_INVALID",
            "verify_chain": check,
        })
    return {
        "note": "Pre-fix ledgers preserved unchanged as evidence; the app does not write to them. "
                "See data/audit_logs/LEGACY_INVALID_README.md.",
        "logs": results,
    }

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_path = PROJECT_ROOT / "src" / "ui" / "templates" / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ClaimSleuth Dashboard UI loading...</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
