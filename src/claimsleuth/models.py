"""Every shape that agents, tools and the API share. Agents write these to the blackboard as JSON."""
from datetime import date
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, Field

from claimsleuth.config import FULL_REVIEW_THRESHOLD


class Quote(BaseModel):
    """A verbatim piece of a document. `text` must appear at `line` of `doc_id`."""
    doc_id: str
    line: int = Field(ge=1)
    text: str = Field(min_length=1)


class Fact(BaseModel):
    name: Literal["device", "serial", "purchase_date", "incident_date", "repair_cost", "cause", "other"]
    value: str
    quote: Quote


class Facts(BaseModel):
    facts: list[Fact]
    missing: list[str] = []


class PolicyTerms(BaseModel):
    """The policy's money and period terms, as stated in its wording (each backed by a quote)."""
    warranty_months: int
    deductible: Decimal
    coverage_limit: Decimal
    depreciation_pct: Decimal = Decimal("0")
    quotes: list[Quote] = []


class Coverage(BaseModel):
    covered: bool
    within_window: bool
    exclusion: Optional[str] = None
    clause: Quote
    days_since_purchase: int
    terms: Optional[PolicyTerms] = None


class Payout(BaseModel):
    """Produced only from the compute_payout tool's result."""
    amount: Decimal
    currency: str = "USD"
    breakdown: dict[str, Decimal]


class FraudFinding(BaseModel):
    kind: str
    severity: Literal["low", "medium", "high"]
    detail: str
    quotes: list[Quote] = []


class Fraud(BaseModel):
    findings: list[FraudFinding]
    flagged: bool


class GuardrailHit(BaseModel):
    kind: Literal["injection", "abuse", "bad_tool_result", "bad_quote", "bad_output"]
    where: str
    detail: str


Decision = Literal["approve", "reject", "request_evidence"]


class ManagerSubmission(BaseModel):
    """What the Claim Manager hands to submit_report. Code adds everything else from the blackboard."""
    decision: Decision
    reasons: list[str] = Field(min_length=1)
    evidence: list[Quote] = []
    payout_amount: Optional[Decimal] = None


class FinalReport(BaseModel):
    claim_id: str
    decision: Decision
    payout: Optional[Payout] = None
    reasons: list[str]
    evidence: list[Quote]
    coverage: Optional[Coverage] = None
    fraud: Optional[Fraud] = None
    guardrail_hits: list[GuardrailHit] = []
    review: Literal["one_click", "full_review"]     # set by code, never by the model
    pattern: str
    prompt_style: str


def review_level(decision: str, payout: Optional[Payout], fraud: Optional[Fraud]) -> str:
    """full_review if the payout is over the threshold, the claim is rejected, or fraud is flagged."""
    if decision == "reject" or (fraud is not None and fraud.flagged):
        return "full_review"
    if payout is not None and payout.amount > Decimal(str(FULL_REVIEW_THRESHOLD)):
        return "full_review"
    return "one_click"


class ClaimRecord(BaseModel):
    """A claim as stored in the claims table and seeded into the blackboard as bb_claim."""
    id: str
    source: Literal["demo", "upload", "eval"]
    claimant: str
    device: str
    serial: Optional[str] = None
    purchase_date: Optional[date] = None
    incident_date: date
    statement: str
    policy_id: Optional[str] = None
    status: str = "new"


class DocumentRecord(BaseModel):
    id: str
    claim_id: str
    kind: Literal["invoice", "diagnostic", "policy", "other"]
    filename: str
    path: str
    sha256: str
    pages: int = 1
    has_text: bool = True


def to_markdown(report: FinalReport) -> str:
    """The human-readable version of a report."""
    lines = [f"# Claim {report.claim_id}", "",
             f"**Decision:** {report.decision}  ", f"**Review:** {report.review}  ",
             f"**Pattern:** {report.pattern} ({report.prompt_style} prompts)", ""]
    if report.payout:
        lines += [f"**Payout:** {report.payout.amount} {report.payout.currency}", "",
                  "| Step | Amount |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in report.payout.breakdown.items()]
        lines.append("")
    lines += ["## Reasons", ""] + [f"- {r}" for r in report.reasons] + [""]
    if report.coverage:
        c = report.coverage
        lines += ["## Coverage", "", f"- Covered: {c.covered}; within window: {c.within_window} "
                  f"({c.days_since_purchase} days since purchase)", f"- Exclusion: {c.exclusion or 'none'}",
                  f"- Clause: \"{c.clause.text}\" ({c.clause.doc_id}, line {c.clause.line})", ""]
    if report.fraud:
        lines += ["## Fraud checks", "", f"Flagged: {report.fraud.flagged}", ""]
        lines += [f"- [{f.severity}] {f.kind}: {f.detail}" for f in report.fraud.findings] + [""]
    if report.evidence:
        lines += ["## Evidence", "", "| Document | Line | Quote |", "| --- | --- | --- |"]
        lines += [f"| {q.doc_id} | {q.line} | {q.text.replace('|', '/')} |" for q in report.evidence] + [""]
    if report.guardrail_hits:
        lines += ["## Guardrail hits", ""] + [f"- {h.kind} in {h.where}: {h.detail}" for h in report.guardrail_hits]
    return "\n".join(lines).rstrip() + "\n"
