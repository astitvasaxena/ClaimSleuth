"""
Input guardrails (C-8): validation, prompt-injection detection and toxicity handling.

Deterministic and dependency-free, so it behaves identically in every pipeline and
in tests. Applied to the claim's free text AND to every dossier document, because
injected instructions usually arrive inside documents, not in the claim form.

Actions
  block   invalid input, prompt injection (claim text or documents), threats of harm.
          The claim gets status BLOCKED_BY_GUARDRAIL: not payable, cannot be approved.
  redact  insults / profanity. The words are replaced with [REDACTED] and the claim is
          investigated normally on its evidence (rudeness is not grounds for denial).
  allow   nothing found.

Limits (stated, not hidden): the toxicity check is a keyword list, not a trained
classifier; injection detection is pattern-based and can be evaded by paraphrase.
Both only ever make the system MORE cautious; neither can approve anything.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Tuple

FREE_TEXT_FIELDS = ("reported_issue", "claimant_statement")
MAX_TEXT_CHARS = 2000
CLAIM_ID_RE = re.compile(r"^[A-Z]{3}-[A-Z0-9-]{3,40}$")
POLICY_ID_RE = re.compile(r"^POL-[A-Z0-9-]{3,40}$")
ZERO_WIDTH_RE = re.compile(r"[​-‏⁠﻿]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)",
    r"disregard\s+(all\s+|the\s+|any\s+)?(previous|prior|above)?\s*(instructions|policy|policies|rules|limits)",
    r"forget\s+(all\s+|your\s+)?(previous\s+)?instructions",
    r"system\s*(prompt|override|message)\s*[:\-]",
    r"\[\s*(system|admin|developer)\s*\]",
    r"you\s+are\s+now\s+(in\s+)?(developer|admin|god)\s*mode",
    r"(approve|authori[sz]e|pay)\s+(this\s+|the\s+)?(claim|payout|payment)\s+(immediately|now|without\s+review)",
    r"set\s+(the\s+)?(deductible|payout|approved_payout|limit)\s+to",
    r"admin\s+override",
    r"(bypass|skip)\s+(the\s+)?(verifier|verification|human\s+review|approval)",
    r"<\s*/?\s*script\s*>",
    r"\bdrop\s+table\b",
]
INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# (term, severity). "threat" blocks; "insult"/"profanity" are redacted.
TOXIC_TERMS: List[Tuple[str, str]] = [
    (r"\bi\s*(will|'ll)\s+(kill|hurt|find)\s+you\b", "threat"),
    (r"\b(burn|blow)\s+(down|up)\s+(your|the)\s+(store|office|shop)\b", "threat"),
    (r"\byou(\s+will|'ll)\s+regret\s+this\b", "threat"),
    (r"\bidiots?\b", "insult"), (r"\bmorons?\b", "insult"), (r"\bstupid\b", "insult"),
    (r"\bimbeciles?\b", "insult"), (r"\bclowns?\b", "insult"), (r"\bincompetent\b", "insult"),
    (r"\bscammers?\b", "insult"), (r"\bthieves\b", "insult"), (r"\bpathetic\b", "insult"),
    (r"\bdamn(ed)?\b", "profanity"), (r"\bcrap(py)?\b", "profanity"), (r"\bf[\*u]ck(ing)?\b", "profanity"),
    (r"\bsh[\*i]t(ty)?\b", "profanity"), (r"\bbullsh[\*i]t\b", "profanity"),
]
TOXIC_RE = [(re.compile(p, re.IGNORECASE), sev) for p, sev in TOXIC_TERMS]


def normalise(text: str) -> str:
    """NFKC, strip zero-width characters, collapse whitespace (defeats spacing/homoglyph tricks)."""
    text = unicodedata.normalize("NFKC", text or "")
    text = ZERO_WIDTH_RE.sub("", text)
    return re.sub(r"\s+", " ", text)


def scan_injection(text: str) -> List[str]:
    norm = normalise(text)
    return [m.group(0) for rx in INJECTION_RE for m in [rx.search(norm)] if m]


def scan_toxicity(text: str) -> List[Tuple[str, str]]:
    norm = normalise(text)
    return [(m.group(0), sev) for rx, sev in TOXIC_RE for m in rx.finditer(norm)]


def redact_toxicity(text: str) -> str:
    out = normalise(text)
    for rx, sev in TOXIC_RE:
        if sev != "threat":
            out = rx.sub("[REDACTED]", out)
    return out


def _valid_iso_date(value: Any) -> bool:
    try:
        date.fromisoformat(str(value))
        return True
    except ValueError:
        return False


def validate_claim(claim: Dict[str, Any]) -> List[Dict[str, str]]:
    """Schema/format validation of the claim form. Any violation blocks the claim."""
    problems = []
    for key in ("claim_id", "policy_id", "claimant_id", "device_model", "incident_date", "reported_issue"):
        if not str(claim.get(key) or "").strip():
            problems.append({"kind": "invalid_input", "field": key, "detail": "missing"})
    if claim.get("claim_id") and not CLAIM_ID_RE.match(str(claim["claim_id"])):
        problems.append({"kind": "invalid_input", "field": "claim_id", "detail": "bad format"})
    if claim.get("policy_id") and not POLICY_ID_RE.match(str(claim["policy_id"])):
        problems.append({"kind": "invalid_input", "field": "policy_id", "detail": "bad format"})
    if claim.get("incident_date") and not _valid_iso_date(claim["incident_date"]):
        problems.append({"kind": "invalid_input", "field": "incident_date",
                         "detail": f"not a valid YYYY-MM-DD date: {str(claim['incident_date'])[:20]!r}"})
    for key in FREE_TEXT_FIELDS:
        value = str(claim.get(key) or "")
        if len(value) > MAX_TEXT_CHARS:
            problems.append({"kind": "invalid_input", "field": key, "detail": f"longer than {MAX_TEXT_CHARS} characters"})
        if CONTROL_RE.search(value):
            problems.append({"kind": "invalid_input", "field": key, "detail": "contains control characters"})
    return problems


@dataclass
class GuardrailResult:
    action: str                                   # "allow" | "redact" | "block"
    violations: List[Dict[str, str]] = field(default_factory=list)
    sanitized_claim: Dict[str, Any] = field(default_factory=dict)

    @property
    def flags(self) -> List[str]:
        """Compact flags stored on the claim state; BLOCK:* flags make the claim non-payable."""
        prefix = {"block": "BLOCK", "redact": "REDACT"}
        return [f"{prefix.get(v['action'], 'NOTE')}:{v['kind']}({v['field']})" for v in self.violations]

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "violations": self.violations, "flags": self.flags}


def evaluate_claim(claim: Dict[str, Any], documents: Dict[str, str]) -> GuardrailResult:
    """Run every check on the claim form and its dossier documents."""
    violations = [{**v, "action": "block"} for v in validate_claim(claim)]
    sanitized = dict(claim)

    for key in FREE_TEXT_FIELDS:
        text = str(claim.get(key) or "")
        for hit in scan_injection(text):
            violations.append({"kind": "prompt_injection", "field": key, "detail": hit[:120], "action": "block"})
        toxic = scan_toxicity(text)
        for term, sev in toxic:
            violations.append({"kind": "threat" if sev == "threat" else "toxic_language", "field": key,
                               "detail": term[:60], "action": "block" if sev == "threat" else "redact"})
        if toxic:
            sanitized[key] = redact_toxicity(text)

    for name, text in documents.items():
        for hit in scan_injection(text):
            violations.append({"kind": "document_injection", "field": name, "detail": hit[:120], "action": "block"})

    action = ("block" if any(v["action"] == "block" for v in violations)
              else "redact" if violations else "allow")
    return GuardrailResult(action=action, violations=violations, sanitized_claim=sanitized)
