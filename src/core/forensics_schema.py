"""
ForensicsReport: what the CrewAI forensics service returns over A2A.

Imported by BOTH environments (the main ADK app and the separate CrewAI venv), so
it depends on pydantic only. extra="forbid" makes any additional field (for
example an attempt to return "approved_payout" or "authorize") a validation
error: the CrewAI service can report findings but cannot carry a payout decision.

Two stages, reported separately:
- findings:   deterministic checks run as steps of a CrewAI Flow (authoritative);
- llm_review: a CrewAI Agent/Task/Crew backed by an LLM that analyses the same
              evidence (advisory). Its execution fields are set from what actually
              happened in the request (Crew.kickoff() returned, LLM calls counted on
              the LLM object, token usage reported by the provider).
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

FORENSICS_JSON_PREFIX = "CLAIMSLEUTH_FORENSICS_JSON:"
FORENSICS_REQUEST_PREFIX = "CLAIMSLEUTH_FORENSICS_REQUEST:"

LLMStatus = Literal["COMPLETED", "LLM_NOT_CONFIGURED", "LLM_ERROR", "LLM_TIMEOUT", "INVALID_OUTPUT"]


class ForensicFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    check: Literal["serial_consistency", "purchase_before_incident", "exclusion_scan", "documents_present"]
    result: Literal["PASS", "FLAG", "NOT_EVALUATED"]
    detail: str
    evidence: List[str] = Field(default_factory=list)   # verbatim snippets the finding rests on


class LLMFinding(BaseModel):
    """One finding written by the CrewAI forensic agent. Advisory: it can only add caution."""
    model_config = ConfigDict(extra="forbid")
    finding_type: Literal["serial_mismatch", "date_inconsistency", "possible_exclusion", "missing_evidence",
                          "document_inconsistency", "suspicious_instruction", "insufficient_evidence",
                          "no_issue_found"]
    severity: Literal["none", "low", "medium", "high"]
    explanation: str = Field(max_length=800)
    source_refs: List[str] = Field(default_factory=list, max_length=12)  # ids of evidence items sent to the agent
    quotes: List[str] = Field(default_factory=list, max_length=6)        # verbatim text from those items
    uncertainty: Literal["low", "medium", "high"]


class LLMAssessment(BaseModel):
    """The JSON object the agent must return (validated before use; extra keys are rejected)."""
    model_config = ConfigDict(extra="forbid")
    findings: List[LLMFinding] = Field(max_length=10)
    recommendation: Literal["NO_ADDITIONAL_CONCERNS", "RECOMMEND_HUMAN_REVIEW", "INSUFFICIENT_EVIDENCE"]
    summary: str = Field(max_length=600)


class LLMReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: LLMStatus
    error_category: Optional[str] = None      # safe category only, never the raw error or any secret
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    crew_kickoff_called: bool = False         # Crew.kickoff() was invoked
    crewai_crew_executed: bool = False        # Crew.kickoff() returned a result
    llm_call_performed: bool = False          # at least one LLM call returned
    llm_call_attempts: int = 0
    llm_calls_succeeded: int = 0
    prompt_tokens: Optional[int] = None       # as reported by the provider
    completion_tokens: Optional[int] = None
    duration_ms: Optional[float] = None
    evidence_sent: List[str] = Field(default_factory=list)   # ids of the evidence items sent (not their text)
    policy_evidence_origin: Optional[str] = None   # "adk_rag_citations" or "service_corpus_lookup"
    findings: List[LLMFinding] = Field(default_factory=list)          # validated findings only
    rejected_findings: List[Dict[str, str]] = Field(default_factory=list)   # {"finding_type", "reason"}
    contradictions: List[str] = Field(default_factory=list)   # LLM vs deterministic disagreements
    recommendation: Optional[Literal["NO_ADDITIONAL_CONCERNS", "RECOMMEND_HUMAN_REVIEW",
                                     "INSUFFICIENT_EVIDENCE"]] = None
    summary: Optional[str] = None


class ForensicsReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    findings: List[ForensicFinding]
    recommendation: Literal["NO_FORENSIC_ISSUES", "ESCALATE_FOR_REVIEW"]
    narrative: Optional[str] = None          # the CrewAI agent's summary; never authoritative
    narrative_source: str = "not_generated"  # "crewai_crew:<provider>/<model>" or "not_generated"
    flow_steps: List[str] = Field(default_factory=list)   # CrewAI Flow methods that executed
    crewai_flow_executed: bool = False       # Flow.kickoff() completed every step
    llm_review: Optional[LLMReview] = None   # CrewAI Agent/Task/Crew stage (None = not attempted)
    crewai_version: str
    service_pid: int
    trace_id: Optional[str] = None   # W3C trace id received from the caller's traceparent header

    def flagged(self) -> List[ForensicFinding]:
        return [f for f in self.findings if f.result == "FLAG"]
