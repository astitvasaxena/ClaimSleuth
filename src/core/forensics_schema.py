"""
ForensicsReport: what the CrewAI forensics service returns over A2A.

Imported by BOTH environments (the main ADK app and the separate CrewAI venv), so
it depends on pydantic only. extra="forbid" makes any additional field (for
example an attempt to return "approved_payout" or "authorize") a validation
error: the CrewAI service can report findings but cannot carry a payout decision.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

FORENSICS_JSON_PREFIX = "CLAIMSLEUTH_FORENSICS_JSON:"


class ForensicFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    check: Literal["serial_consistency", "purchase_before_incident", "exclusion_scan", "documents_present"]
    result: Literal["PASS", "FLAG", "NOT_EVALUATED"]
    detail: str
    evidence: List[str] = Field(default_factory=list)   # verbatim snippets the finding rests on


class ForensicsReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    findings: List[ForensicFinding]
    recommendation: Literal["NO_FORENSIC_ISSUES", "ESCALATE_FOR_REVIEW"]
    narrative: Optional[str] = None          # free text from the CrewAI Crew; never authoritative
    narrative_source: str = "not_generated"  # e.g. "crewai_crew:fake-reviewer" or "not_generated"
    flow_steps: List[str] = Field(default_factory=list)   # CrewAI Flow methods that executed
    crewai_version: str
    service_pid: int
    trace_id: Optional[str] = None   # W3C trace id received from the caller's traceparent header

    def flagged(self) -> List[ForensicFinding]:
        return [f for f in self.findings if f.result == "FLAG"]
