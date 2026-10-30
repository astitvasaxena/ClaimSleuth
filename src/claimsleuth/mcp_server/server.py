"""
The ClaimSleuth MCP server: five tools, served over streamable HTTP at /mcp inside the FastAPI app.

The tools read documents and do the exact work (dates, money, database lookups). They never decide
anything: the agents call them and interpret the results. Invalid input or a missing record raises
ToolError, which MCP returns to the caller as an error result, never as made-up data.
"""
import functools
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Optional

from dateutil.relativedelta import relativedelta
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

from claimsleuth import config, db, tracing
from claimsleuth.mcp_server.pdf import NoTextLayer, read_lines
from claimsleuth.rag.retriever import retrieve

TOOL_NAMES = ("parse_document", "search_policy", "check_coverage", "compute_payout", "claim_history")
POLICY_DOC_PREFIX = "policy:"
CENTS = Decimal("0.01")

server = MCPServer("ClaimSleuthTools", instructions="Warranty-claim tools: documents, policy search, dates, money, claim history.")


class Line(BaseModel):
    n: int
    text: str


class ParsedDocument(BaseModel):
    doc_id: str
    kind: str
    lines: list[Line]
    has_text: bool


class PolicyPassage(BaseModel):
    doc_id: str
    section: str
    score: float
    lines: list[Line]


class PolicySearchResult(BaseModel):
    passages: list[PolicyPassage]
    warnings: list[str] = []


class CoverageCheck(BaseModel):
    within_window: bool
    days_since_purchase: int
    days_left: int
    warranty_end: date
    matched_exclusion: Optional[str]


class PayoutResult(BaseModel):
    amount: Decimal
    breakdown: dict[str, Decimal]


class PastClaim(BaseModel):
    serial: Optional[str]
    claimant: str
    claim_date: str
    outcome: str
    amount: Optional[float]
    source: str


class HistoryResult(BaseModel):
    past_claims: list[PastClaim]
    same_serial_count: int
    last_claim_days_ago: Optional[int]


# ─── Shared helpers (also used by the quote guardrail) ─────────────────────────────────────

def document_lines(doc_id: str) -> tuple[str, list[str]]:
    """(kind, lines) for a claim document id or a 'policy:<POLICY-ID>' wording document."""
    if doc_id.startswith(POLICY_DOC_PREFIX):
        path = Path(config.RAG_CORPUS_DIR) / f"{doc_id[len(POLICY_DOC_PREFIX):]}.md"
        if not path.is_file():
            raise ToolError(f"unknown policy document '{doc_id}'")
        return "policy", path.read_text(encoding="utf-8").splitlines()
    with db.connect() as conn:
        row = conn.execute("SELECT kind, path FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        raise ToolError(f"unknown document '{doc_id}'")
    path = db.resolve_document_path(row["path"])
    if not path.is_file():
        raise ToolError(f"document file for '{doc_id}' is missing")
    try:
        lines, _ = read_lines(path)
    except NoTextLayer as exc:
        raise ToolError(str(exc)) from exc
    return row["kind"], lines


def numbered(lines: list[str], start: int = 1, end: Optional[int] = None) -> list[Line]:
    stop = len(lines) if end is None else min(end, len(lines))
    return [Line(n=i, text=lines[i - 1]) for i in range(start, stop + 1) if lines[i - 1].strip()]


def _section_lines(lines: list[str], section: str) -> list[Line]:
    """The numbered lines of a '## <section>' block in a Markdown file."""
    start = next((i + 1 for i, line in enumerate(lines) if line.strip() == f"## {section}"), None)
    if start is None:
        return []
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    return numbered(lines, start, end)


def traced(fn):
    """Run the tool inside an 'mcp.<tool>' span whose parent is the caller's traceparent header."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with tracing.remote_span(f"mcp.{fn.__name__}", {"mcp.tool": fn.__name__}):
            return fn(*args, **kwargs)
    return wrapper


# ─── Tools ─────────────────────────────────────────────────────────────────────────────────

@server.tool()
@traced
def parse_document(doc_id: str) -> ParsedDocument:
    """Read one claim document (PDF, TXT or MD) and return its non-empty lines with line numbers.

    Use the line numbers when quoting. A scanned PDF without a text layer returns an error.
    """
    kind, lines = document_lines(doc_id)
    return ParsedDocument(doc_id=doc_id, kind=kind, lines=numbered(lines), has_text=True)


@server.tool()
@traced
def search_policy(claim_id: str, query: str, k: int = 4) -> PolicySearchResult:
    """Search the wording of this claim's warranty policy (and any policy file uploaded with the claim).

    Returns the best-matching sections as numbered lines. Quote the clause with its doc_id and line.
    """
    claim = db.get_claim(claim_id)
    if claim is None:
        raise ToolError(f"unknown claim '{claim_id}'")
    if not 1 <= k <= 8:
        raise ToolError("k must be between 1 and 8")
    uploaded = {}
    for doc in db.get_documents(claim_id):
        if doc["kind"] == "policy":
            uploaded[doc["filename"]] = "\n".join(document_lines(doc["id"])[1])   # chunk source = "<claim>/<file>" = doc id
    if not claim["policy_id"] and not uploaded:
        raise ToolError("this claim has no policy_id and no uploaded policy document")
    result = retrieve(query, policy_id=claim["policy_id"], claim_id=claim_id,
                      claim_documents=uploaded or None, k=k, include_guidelines=True)
    passages = []
    for p in result.passages:
        if p.doc_type == "claim_document":                     # an uploaded policy: section is "lines a-b"
            doc_id = p.source_file
            a, b = (int(x) for x in p.section.removeprefix("lines ").split("-"))
            lines = numbered(document_lines(doc_id)[1], a, b)
        else:
            doc_id = POLICY_DOC_PREFIX + Path(p.source_file).stem
            lines = _section_lines((config.BASE_DIR / p.source_file).read_text(encoding="utf-8").splitlines(), p.section)
        passages.append(PolicyPassage(doc_id=doc_id, section=p.section, score=round(p.score, 4), lines=lines))
    warnings = result.warnings + [f"quarantined (instruction-like text): {q.section}" for q in result.quarantined]
    return PolicySearchResult(passages=passages, warnings=warnings)


@server.tool()
@traced
def check_coverage(purchase_date: date, incident_date: date, warranty_months: int,
                   exclusions: list[str], cause: str) -> CoverageCheck:
    """Date maths for the warranty window, and exact phrase matching of the cause against the exclusions.

    within_window is true when the incident is on or after the purchase date and on or before the
    warranty end date (purchase date + warranty_months). matched_exclusion is the first exclusion
    phrase that appears word for word in `cause`, ignoring case; deciding whether a differently
    worded cause is excluded is up to you.
    """
    if not 1 <= warranty_months <= 240:
        raise ToolError("warranty_months must be between 1 and 240")
    end = purchase_date + relativedelta(months=warranty_months)
    lowered = cause.lower()
    matched = next((e for e in exclusions if e.strip() and e.strip().lower() in lowered), None)
    return CoverageCheck(within_window=purchase_date <= incident_date <= end,
                         days_since_purchase=(incident_date - purchase_date).days,
                         days_left=(end - incident_date).days, warranty_end=end, matched_exclusion=matched)


@server.tool()
@traced
def compute_payout(repair_cost: Decimal, deductible: Decimal, coverage_limit: Decimal,
                   depreciation_pct: Decimal = Decimal("0")) -> PayoutResult:
    """The only source of money figures. payout = min(repair_cost - deductible, coverage_limit) x (1 - depreciation_pct/100), rounded to cents."""
    if repair_cost < 0 or deductible < 0 or coverage_limit < 0:
        raise ToolError("amounts must not be negative")
    if not 0 <= depreciation_pct < 100:
        raise ToolError("depreciation_pct must be from 0 to below 100")
    eligible = max(repair_cost - deductible, Decimal("0"))
    capped = min(eligible, coverage_limit)
    depreciation = (capped * depreciation_pct / Decimal(100)).quantize(CENTS, ROUND_HALF_UP)
    amount = (capped - depreciation).quantize(CENTS, ROUND_HALF_UP)
    return PayoutResult(amount=amount, breakdown={
        "repair_cost": repair_cost.quantize(CENTS), "deductible": deductible.quantize(CENTS),
        "eligible_after_deductible": eligible.quantize(CENTS), "coverage_limit": coverage_limit.quantize(CENTS),
        "capped_amount": capped.quantize(CENTS), "depreciation": depreciation, "payout": amount})


@server.tool()
@traced
def claim_history(serial: str, claimant: str, claim_id: Optional[str] = None) -> HistoryResult:
    """Earlier claims for this serial number or this claimant: the claim-history records and earlier claims
    in ClaimSleuth, excluding `claim_id` (the claim under investigation). last_claim_days_ago is measured
    from this claim's incident date (today if no claim_id is given)."""
    if not serial.strip() and not claimant.strip():
        raise ToolError("give a serial number or a claimant name")
    reference = date.today()
    found: list[PastClaim] = []
    with db.connect() as conn:
        if claim_id:
            row = conn.execute("SELECT incident_date FROM claims WHERE id = ?", (claim_id,)).fetchone()
            if row is None:
                raise ToolError(f"unknown claim '{claim_id}'")
            reference = date.fromisoformat(row["incident_date"])
        for r in conn.execute("SELECT * FROM past_claims WHERE upper(serial) = upper(?) OR lower(claimant) = lower(?)",
                              (serial.strip(), claimant.strip())):
            found.append(PastClaim(serial=r["serial"], claimant=r["claimant"], claim_date=r["claim_date"],
                                   outcome=r["outcome"], amount=r["amount"], source="claim_history"))
        # Evaluation claims are test fixtures, not history: the five persona variants of one packet share a
        # claimant and serial, and must not count as each other's earlier claims.
        for r in conn.execute("SELECT * FROM claims WHERE id != ? AND source != 'eval'"
                              " AND (upper(serial) = upper(?) OR lower(claimant) = lower(?))",
                              (claim_id or "", serial.strip(), claimant.strip())):
            found.append(PastClaim(serial=r["serial"], claimant=r["claimant"], claim_date=r["incident_date"],
                                   outcome=r["status"], amount=None, source="claimsleuth"))
    same_serial = sum(1 for c in found if c.serial and serial.strip() and c.serial.upper() == serial.strip().upper())
    days = [(reference - date.fromisoformat(c.claim_date)).days for c in found]
    earlier = [d for d in days if d >= 0]
    return HistoryResult(past_claims=found, same_serial_count=same_serial,
                         last_claim_days_ago=min(earlier) if earlier else None)


def http_app():
    """The Starlette app serving /mcp. Stateless JSON-RPC over HTTP; the shared-secret check sits in front of it
    (in the FastAPI app), so the SDK's localhost-only host check is turned off."""
    return server.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
