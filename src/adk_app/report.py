"""
FinalReport: the structured (Pydantic) output of one ADK investigation run,
written to disk as JSON and Markdown.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.core.state import SettlementCalc, VerifierReport


class EvidenceRef(BaseModel):
    field_name: str
    value: Any
    source_document: str
    raw_snippet: str
    confidence: float
    status: str


class Eligibility(BaseModel):
    covered: Optional[bool] = None
    coverage_reason: Optional[str] = None
    expiration_date: Optional[str] = None
    excluded: bool = False
    matched_exclusion: Optional[str] = None


class PayoutView(BaseModel):
    payout_status: Literal["PROPOSED_PENDING_APPROVAL", "AUTHORIZED", "NOT_PAYABLE"]
    authorized_payout: Optional[float] = None
    proposed_payout: Optional[float] = None
    diagnostic_calculation: Optional[Dict[str, Any]] = None
    explanation: str


class FinalReport(BaseModel):
    run_id: str
    trace_id: Optional[str] = None                                         # W3C trace id (ADK, MCP and A2A logs)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    claim_id: str
    policy_id: str
    pipeline_backend: Literal["workflow", "sequential_agent"]
    planner: str                               # "deterministic" or "llm:<model name>"
    tool_backend: str = "local"                # "local", "mcp_stdio" or "mcp_http"
    pattern: str = "sequential"                # "sequential", "parallel" or "hierarchical"
    messaging: str = "blackboard"              # "blackboard", "direct" or "broadcast"
    status: str
    summary: str
    recommended_action: Literal["PROPOSE_PAYOUT_FOR_HUMAN_APPROVAL", "REJECT", "ESCALATE_TO_HUMAN", "REQUEST_EVIDENCE"]
    requires_human_approval: bool
    evidence: List[EvidenceRef] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    eligibility: Eligibility
    settlement_proposal: Optional[SettlementCalc] = None   # deterministic tool output only
    model_payout_claim: Optional[float] = None             # what a model stated; never authoritative
    verifier: VerifierReport
    payout: PayoutView
    risk_flags: List[str] = Field(default_factory=list)
    model_tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)   # deterministic agents' tool calls
    tool_errors: List[str] = Field(default_factory=list)
    forensics_mode: str = "none"                       # "none" or "crewai_a2a@<url>"
    forensics: Optional[Dict[str, Any]] = None         # validated CrewAI ForensicsReport (advisory only)
    a2a_call: Optional[Dict[str, Any]] = None          # A2A task/context ids of the CrewAI call
    agent_timings: List[Dict[str, Any]] = Field(default_factory=list)     # measured per-agent start/end (ms)
    manager_decisions: List[Dict[str, Any]] = Field(default_factory=list) # hierarchical pattern only
    messaging_summary: Dict[str, Any] = Field(default_factory=dict)
    agent_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # per agent: latency, tools, errors, tokens
    errors: List[Dict[str, Any]] = Field(default_factory=list)              # structured: agent, kind, message
    guardrail: Optional[Dict[str, Any]] = None
    # RAG (src/rag/): retrieved passages are untrusted evidence with real source references
    rag: Optional[Dict[str, Any]] = None                                  # retriever, warnings, errors, stated terms
    policy_citations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)   # topic -> file, section, quote, score
    retrieved_evidence: List[Dict[str, Any]] = Field(default_factory=list)      # every passage used (or quarantined)
    agent_sequence: List[str] = Field(default_factory=list)
    audit_chain_valid: bool
    # Filled only after a human decision via /api/adjudicate (written as a separate *_adjudicated file)
    human_decision: Optional[Literal["APPROVED", "REJECTED", "INFO_REQUESTED"]] = None
    human_notes: Optional[str] = None
    decided_at: Optional[str] = None


def to_markdown(report: FinalReport) -> str:
    """Human-readable rendering of the same report."""
    money = lambda v: "—" if v is None else f"${v:,.2f}"
    lines = [
        f"# ClaimSleuth investigation report: {report.claim_id}",
        "",
        f"- **Run ID:** {report.run_id}",
        f"- **Trace ID:** {report.trace_id or '—'}",
        f"- **Generated:** {report.generated_at}",
        f"- **Pipeline:** Google ADK `{report.pipeline_backend}`; pattern: {report.pattern}; messaging: "
        f"{report.messaging}; planner: {report.planner}; tools: {report.tool_backend}",
        f"- **Status:** {report.status}",
        f"- **Recommended action:** {report.recommended_action}",
        f"- **Human approval required:** {'yes' if report.requires_human_approval else 'no'}",
        f"- **Audit chain valid:** {'yes' if report.audit_chain_valid else 'NO'}",
        *([f"- **Human decision:** {report.human_decision} at {report.decided_at}"
           + (f" ({report.human_notes})" if report.human_notes else "")] if report.human_decision else []),
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "## Payout",
        "",
        "| Payout status | Proposed | Authorized | Diagnostic only |",
        "|---|---|---|---|",
        f"| {report.payout.payout_status} | {money(report.payout.proposed_payout)} | "
        f"{money(report.payout.authorized_payout)} | "
        f"{money((report.payout.diagnostic_calculation or {}).get('approved_payout'))} |",
        "",
        report.payout.explanation,
        "",
        "## Evidence",
        "",
        "| Field | Value | Source | Snippet | Status |",
        "|---|---|---|---|---|",
    ]
    for ev in report.evidence:
        snippet = ev.raw_snippet.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {ev.field_name} | {ev.value} | {ev.source_document} | `{snippet}` | {ev.status} |")
    lines += ["", "## Eligibility", "",
              f"- Covered: {report.eligibility.covered} ({report.eligibility.coverage_reason})",
              f"- Excluded: {report.eligibility.excluded} {report.eligibility.matched_exclusion or ''}".rstrip(),
              "", "## Conflicts and missing evidence", ""]
    lines += [f"- Conflict: {c}" for c in report.conflicts] or ["- No conflicts"]
    lines += [f"- Missing: {m}" for m in report.missing_evidence]
    lines += ["", "## Verifier", "",
              f"- Approved by verifier: {report.verifier.is_approved_by_verifier}",
              f"- Gate 1 (evidence): {report.verifier.gate_1_evidence_substantiated}",
              f"- Gate 2 (rules): {report.verifier.gate_2_deterministic_rules_passed}",
              f"- Gate 2 (arithmetic): {report.verifier.gate_2_arithmetic_matches}"]
    lines += [f"- Discrepancy: {d}" for d in report.verifier.discrepancies]
    if report.forensics_mode != "none":
        lines += ["", "## CrewAI forensics (via A2A, advisory)", "", f"- Service: {report.forensics_mode}"]
        if report.forensics:
            lines.append(f"- Recommendation: {report.forensics['recommendation']}")
            lines += [f"- {f['check']}: {f['result']} ({f['detail']})" for f in report.forensics["findings"]]
        else:
            lines.append("- No valid forensics report (see tool errors)")
    if report.rag is not None:
        lines += ["", "## Policy evidence retrieved (RAG)", "",
                  f"- Retriever: {report.rag.get('retriever') or '—'}"
                  + (f"; ERROR: {report.rag['error']}" if report.rag.get("error") else "")]
        if report.policy_citations:
            lines += ["", "| Topic | Source | Section | Quote | Score |", "|---|---|---|---|---|"]
            for topic, c in report.policy_citations.items():
                quote = c["quote"].replace("|", "\\|")[:160]
                lines.append(f"| {topic} | `{c['source_file']}` | {c['section']} | {quote} | {c['score']:.2f} |")
        if report.rag.get("missing_citations"):
            lines.append(f"- No wording found for: {', '.join(report.rag['missing_citations'])}")
        quarantined = [p for p in report.retrieved_evidence if p.get("quarantined")]
        if quarantined:
            lines.append(f"- Quarantined passages (instruction-like text, not used): "
                         + ", ".join(f"`{p['source_file']}` §{p['section']}" for p in quarantined))
    lines += ["", "## Agent sequence", "", " → ".join(report.agent_sequence), ""]
    if report.manager_decisions:
        lines += ["## Manager decisions (hierarchical)", ""]
        lines += [f"{d['n']}. {d['route']}: {d['reason']}" for d in report.manager_decisions]
        lines.append("")
    if report.agent_timings:
        lines += ["## Agent timings (ms since run start)", "", "| Agent | Start | End | Branch |", "|---|---|---|---|"]
        lines += [f"| {t['agent']} | {t['start_ms']} | {t['end_ms']} | {'yes' if t.get('branch') else ''} |"
                  for t in report.agent_timings]
        lines.append("")
    if report.model_tool_calls:
        lines += ["## Model tool calls", ""]
        lines += [f"- {c.get('tool')}({json.dumps(c.get('args'), default=str)})" for c in report.model_tool_calls]
        lines.append("")
    return "\n".join(lines)


def write_report_files(report: FinalReport, reports_dir: Path, suffix: str = "") -> Dict[str, str]:
    """Write <claim>_<run><suffix>.json and .md; return their paths."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.claim_id}_{report.run_id}{suffix}"
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
