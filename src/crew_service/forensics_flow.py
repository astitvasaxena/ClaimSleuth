"""
CrewAI forensics: a real crewai Flow whose steps run the deterministic checks, then
a real crewai Agent/Task/Crew backed by an LLM that analyses the same request's
evidence (src/crew_service/llm_review.py).

Runs ONLY in the separate CrewAI environment (.venv-crewai): crewai pins mcp<2 and
pydantic<2.13, which conflict with the main app. It reuses the project's
deterministic code (extractor, anomaly detector, policy rules), which stays
authoritative, and never computes, proposes or authorizes a payout.
"""
import os
import re
from typing import Any, Dict, List, Optional

# Keep the service offline and private: no CrewAI/OpenTelemetry telemetry.
for _var, _val in (("CREWAI_DISABLE_TELEMETRY", "true"), ("CREWAI_DISABLE_TRACKING", "true"),
                   ("OTEL_SDK_DISABLED", "true"), ("CREWAI_TRACING_ENABLED", "false")):
    os.environ.setdefault(_var, _val)

import crewai
from crewai.flow.flow import Flow, listen, start
from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, Field

from src.agents.extractor import EvidenceExtractor
from src.core.forensics_schema import ForensicFinding, ForensicsReport, LLMReview
from src.crew_service.llm_review import LLMConfig, config_for_instance, resolve_llm_config, run_llm_review
from src.tools.anomaly_detector import flag_serial_anomalies
from src.tools.evidence_store import EvidenceStore
from src.tools.policy_rules import check_exclusions, load_policy

CLAIM_ID_PATTERN = re.compile(r"^[A-Z]{3}-[A-Z0-9-]{3,40}$")
FLOW_STEPS = ["load_dossier", "extract_facts", "check_serials", "check_chronology", "scan_exclusions",
              "llm_forensic_review"]


def is_invoice(name: str) -> bool:
    return "invoice" in name.lower() or "receipt" in name.lower()


class ForensicsState(BaseModel):
    claim: Dict[str, Any] = Field(default_factory=dict)
    documents: Dict[str, str] = Field(default_factory=dict)
    serials: Dict[str, Dict[str, str]] = Field(default_factory=dict)       # doc -> {value, snippet}
    purchase_date: Optional[Dict[str, str]] = None                        # {value, snippet, doc}
    incident_cause: str = ""
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    supplied_policy_evidence: List[Dict[str, Any]] = Field(default_factory=list)   # ADK's RAG citations
    llm_review: Optional[Dict[str, Any]] = None


class ForensicsFlow(Flow[ForensicsState]):
    """Forensic review of one claim dossier, orchestrated by CrewAI Flow: deterministic checks, then the LLM Crew."""

    def __init__(self, llm_config: Optional[LLMConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm_config = llm_config or resolve_llm_config("none")

    @start()
    def load_dossier(self):
        self.state.steps.append("load_dossier")
        store = EvidenceStore()
        self.state.documents = store.get_all_claim_documents(self.state.claim["claim_id"])
        present = bool(self.state.documents)
        self.state.findings.append({"check": "documents_present", "result": "PASS" if present else "FLAG",
                                    "detail": f"{len(self.state.documents)} dossier document(s)"})

    @listen(load_dossier)
    def extract_facts(self):
        self.state.steps.append("extract_facts")
        extractor = EvidenceExtractor(llm_client=None)
        for name, text in self.state.documents.items():
            fields = ["purchase_date", "serial_number"] if is_invoice(name) else ["serial_number", "incident_cause"]
            facts = extractor.extract_from_document(name, text, fields, use_llm=False)
            if "serial_number" in facts and facts["serial_number"].status == "VERIFIED":
                self.state.serials[name] = {"value": str(facts["serial_number"].extracted_value),
                                            "snippet": facts["serial_number"].raw_snippet}
            if "purchase_date" in facts and facts["purchase_date"].status == "VERIFIED" and not self.state.purchase_date:
                self.state.purchase_date = {"value": str(facts["purchase_date"].extracted_value),
                                            "snippet": facts["purchase_date"].raw_snippet, "doc": name}
            if "incident_cause" in facts:
                self.state.incident_cause += " " + str(facts["incident_cause"].extracted_value)

    @listen(extract_facts)
    def check_serials(self):
        self.state.steps.append("check_serials")
        inv = next((v for d, v in self.state.serials.items() if is_invoice(d)), None)
        diag = next((v for d, v in self.state.serials.items() if not is_invoice(d)), None)
        if not inv or not diag:
            self.state.findings.append({"check": "serial_consistency", "result": "NOT_EVALUATED",
                                        "detail": "Serial number not found on both invoice and repair document"})
            return
        res = flag_serial_anomalies(inv["value"], inv["value"], diag["value"])
        self.state.findings.append({
            "check": "serial_consistency", "result": "FLAG" if res["has_anomalies"] else "PASS",
            "detail": "; ".join(res["anomalies"]) or "Invoice and repair document serials match",
            "evidence": [inv["snippet"], diag["snippet"]]})

    @listen(check_serials)
    def check_chronology(self):
        self.state.steps.append("check_chronology")
        pd = self.state.purchase_date
        incident = self.state.claim.get("incident_date", "")
        if not pd or not incident:
            self.state.findings.append({"check": "purchase_before_incident", "result": "NOT_EVALUATED",
                                        "detail": "Purchase or incident date missing"})
            return
        ok = pd["value"] <= incident   # ISO dates compare correctly as strings
        self.state.findings.append({
            "check": "purchase_before_incident", "result": "PASS" if ok else "FLAG",
            "detail": f"Purchase {pd['value']} {'precedes' if ok else 'is AFTER'} incident {incident}",
            "evidence": [pd["snippet"]]})

    @listen(check_chronology)
    def scan_exclusions(self):
        self.state.steps.append("scan_exclusions")
        policy = load_policy(self.state.claim.get("policy_id", ""))
        if policy is None:
            self.state.findings.append({"check": "exclusion_scan", "result": "NOT_EVALUATED",
                                        "detail": "Unknown policy"})
            return
        cause = f"{self.state.claim.get('reported_issue', '')} {self.state.incident_cause}"
        excluded, matched = check_exclusions(cause, policy.exclusions)
        self.state.findings.append({"check": "exclusion_scan", "result": "FLAG" if excluded else "PASS",
                                    "detail": f"Matches exclusion '{matched}'" if excluded else "No exclusion matched"})

    @listen(scan_exclusions)
    def llm_forensic_review(self):
        """CrewAI Agent/Task/Crew + LLM on this request's evidence. Never raises: failures become a status."""
        self.state.steps.append("llm_forensic_review")
        review = run_llm_review(self.state.claim, self.state.documents, self.state.findings,
                                self.state.supplied_policy_evidence, self._llm_config)
        self.state.llm_review = review.model_dump()


def run_forensics(claim: Dict[str, Any], llm: Optional[BaseLLM] = None, llm_config: Optional[LLMConfig] = None,
                  policy_evidence: Optional[List[Dict[str, Any]]] = None) -> ForensicsReport:
    """Run the CrewAI Flow (and its LLM Crew stage) for one claim and return a validated ForensicsReport."""
    config = llm_config or (config_for_instance(llm) if llm is not None else None)
    flow = ForensicsFlow(llm_config=config)
    flow.kickoff(inputs={"claim": claim, "supplied_policy_evidence": list(policy_evidence or [])})
    findings = [ForensicFinding(**f) for f in flow.state.findings]
    flagged = any(f.result == "FLAG" for f in findings)
    review = LLMReview(**flow.state.llm_review) if flow.state.llm_review else None
    completed = review is not None and review.status == "COMPLETED"
    return ForensicsReport(
        claim_id=claim["claim_id"], findings=findings,
        recommendation="ESCALATE_FOR_REVIEW" if flagged else "NO_FORENSIC_ISSUES",
        narrative=review.summary if completed else None,
        narrative_source=f"crewai_crew:{review.llm_provider}/{review.llm_model}" if completed else "not_generated",
        flow_steps=list(flow.state.steps), crewai_flow_executed=list(flow.state.steps) == FLOW_STEPS,
        llm_review=review, crewai_version=crewai.__version__, service_pid=os.getpid())
