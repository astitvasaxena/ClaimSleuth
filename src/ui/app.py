import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.state import ClaimInvestigationState
from src.core.audit import AuditLogger
from src.agents.investigator import LeadInvestigatorAgent
from src.tools.evidence_store import EvidenceStore
from configs.settings import ADJUDICATION_TOKEN

app = FastAPI(title="ClaimSleuth: Agentic AI Warranty Investigation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

evidence_store = EvidenceStore()
investigator = LeadInvestigatorAgent(evidence_store=evidence_store)

# In-memory storage for demo sessions
active_states: Dict[str, ClaimInvestigationState] = {}

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
    decision: str
    notes: Optional[str] = ""

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
    result_state = investigator.run_investigation(initial_state)
    active_states[result_state.claim_id] = result_state
    
    audit_logger = AuditLogger(result_state.claim_id)
    audit_trail = audit_logger.get_trail()
    
    bb_summary = investigator.get_last_blackboard_summary(result_state.claim_id)
    monitor_summary = investigator.get_last_monitoring_summary(result_state.claim_id)

    return {
        "state": result_state.model_dump(),
        "audit_trail": audit_trail,
        "blackboard": bb_summary,
        "monitoring": monitor_summary,
        "mcp_tools": ["check_coverage_window", "compute_settlement", "flag_serial_anomalies"],
        "frameworks": {
            "google_adk": "Active",
            "crewai": "Active"
        }
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

@app.post("/api/investigate-crew/{scenario_id}")
def run_crewai_investigation(scenario_id: str):
    """Run collaborative multi-agent investigation using the CrewAI role adapter."""
    folder = PROJECT_ROOT / "data" / "sample_claims" / scenario_id
    claim_json = folder / "claim.json"
    if not claim_json.exists():
        raise HTTPException(status_code=404, detail="Claim definition not found")
        
    with open(claim_json, "r", encoding="utf-8") as f:
        claim_data = json.load(f)
        
    from src.core.blackboard import Blackboard
    from src.frameworks.crewai_adapter import ClaimInvestigationCrew

    claim_id = claim_data.get("claim_id", scenario_id)
    bb = Blackboard(claim_id)
    crew = ClaimInvestigationCrew(claim_id, bb)
    result = crew.kickoff(claim_data)
    return {
        "crew_result": result,
        "blackboard": bb.get_summary()
    }

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
    state.human_decision = req.decision
    state.human_notes = req.notes
    
    audit = AuditLogger(claim_id)
    if req.decision == "APPROVED":
        state.status = "CLOSED_SETTLED"
        audit.log_human_decision("APPROVED", req.notes)
    elif req.decision == "REJECTED":
        state.status = "CLOSED_REJECTED"
        audit.log_human_decision("REJECTED", req.notes)
    elif req.decision == "INFO_REQUESTED":
        state.status = "NEEDS_EVIDENCE"
        audit.log_human_decision("INFO_REQUESTED", req.notes)

    return {
        "status": state.status,
        "human_decision": state.human_decision,
        "audit_trail": audit.get_trail()
    }

@app.get("/api/audit/{claim_id}")
def get_audit_trail(claim_id: str):
    audit = AuditLogger(claim_id)
    return audit.get_trail()

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
