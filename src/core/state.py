from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class EvidenceItem(BaseModel):
    """Structured evidence item extracted from raw documents with provenance."""
    evidence_id: str
    source_document: str        # e.g., 'purchase_invoice.txt'
    field_name: str             # e.g., 'purchase_date', 'serial_number'
    extracted_value: Any        # Canonicalized value
    raw_snippet: str            # Verbatim snippet from the document
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: Literal["VERIFIED", "UNVERIFIED", "CONTRADICTED"] = "UNVERIFIED"
    verification_notes: Optional[str] = None

class PolicyTerms(BaseModel):
    """Warranty contract terms and limits."""
    policy_id: str
    name: str
    product_category: str
    duration_months: int
    max_coverage_limit: float
    deductible: float
    depreciation_rate: float = 0.0
    requires_original_receipt: bool = True
    requires_diagnostic_report: bool = True
    exclusions: List[str] = Field(default_factory=list)

class SettlementCalc(BaseModel):
    """Deterministic mathematical calculation of claim payout."""
    repair_cost: float
    deductible: float
    depreciation_applied: float
    max_coverage_limit: float
    net_eligible_amount: float
    approved_payout: float
    formula_explanation: str
    is_arithmetically_valid: bool = True

class AgentAction(BaseModel):
    """Structured action selected by the Lead Investigator (via Gemini or fallback)."""
    tool_name: Literal[
        "list_dossier_documents",
        "extract_document_facts",
        "check_coverage_window",
        "compute_settlement",
        "check_exclusions",
        "flag_serial_anomalies",
        "STOP"
    ]
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""  # Agent's explanation of why this action was chosen
    replan_reason: Optional[str] = None  # If replanning, why the previous plan was insufficient

class InvestigationStep(BaseModel):
    """Record of a single cycle in the PLAN-EXECUTE-OBSERVE-ASSESS-REPLAN loop."""
    step_num: int
    phase: Literal["PLAN", "EXECUTE", "OBSERVE", "ASSESS", "REPLAN", "STOP"]
    description: str
    tool_called: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    replan_reason: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class VerifierReport(BaseModel):
    """Dual-Gate Independent Verifier outcome."""
    is_approved_by_verifier: bool
    gate_1_evidence_substantiated: bool
    gate_2_deterministic_rules_passed: bool
    gate_2_arithmetic_matches: bool
    discrepancies: List[str] = Field(default_factory=list)
    verified_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ClaimInvestigationState(BaseModel):
    """Full investigation dossier state for a warranty claim."""
    claim_id: str
    policy_id: str
    claimant_id: str
    device_model: str
    incident_date: str
    reported_issue: str
    current_step: int = 0
    max_steps: int = 5
    llm_calls_used: int = 0
    max_llm_calls: int = 8
    
    # Evidence & Analysis
    evidence: Dict[str, EvidenceItem] = Field(default_factory=dict)
    conflicts: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    steps_history: List[InvestigationStep] = Field(default_factory=list)
    
    # Adjudication
    deterministic_settlement: Optional[SettlementCalc] = None
    verifier_report: Optional[VerifierReport] = None
    requires_human_approval: bool = True
    human_decision: Optional[Literal["APPROVED", "REJECTED", "INFO_REQUESTED"]] = None
    human_notes: Optional[str] = None
    
    status: Literal[
        "INITIALIZED",
        "IN_PROGRESS",
        "NEEDS_EVIDENCE",
        "CONFLICT_ESCALATED",
        "VERIFIED_PENDING_APPROVAL",
        "REJECTED_DETERMINISTICALLY",
        "CLOSED_SETTLED",
        "CLOSED_REJECTED"
    ] = "INITIALIZED"
