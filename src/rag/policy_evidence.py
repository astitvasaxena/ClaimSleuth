"""
Turns RAG retrieval into evidence the investigation can use and the verifier can check.

For one claim, PolicyRetrievalAgent asks a fixed set of investigation questions against the
claim's policy wording (plus the claims-handling guidelines and the claim's own documents),
and for each policy topic keeps a CITATION: source file, section, chunk id, score and a
verbatim quote. For the four numeric terms it also parses what the wording STATES
(coverage months, deductible, limit, depreciation).

The VerifierAgent then compares those stated terms with the structured terms the
deterministic settlement code used (check_policy_citations). RAG therefore can only add
caution: missing wording -> no settlement may be proposed; contradicting wording or
instruction-like text in a passage -> the claim is escalated. It never sets or approves money.
"""
import re
from typing import Any, Dict, List, Optional

from src.rag.retriever import RetrievalResult, retrieve

TERM_TOPICS = {
    "coverage_period": ("How long is the warranty coverage period from the purchase date?", "Coverage period",
                        r"[^.]*\bfor (\d+) months[^.]*\.", "duration_months", int),
    "deductible": ("What deductible applies to a repair claim?", "Deductible",
                   r"[^.]*deductible of \$([\d,]+\.\d{2})[^.]*\.", "deductible", float),
    "max_coverage_limit": ("What is the maximum amount payable for a single claim?", "Maximum coverage limit",
                           r"[^.]*maximum amount payable for a single claim is \$([\d,]+\.\d{2})", "max_coverage_limit", float),
    "depreciation": ("What depreciation rate is applied to the payout?", "Depreciation",
                     r"[^.]*depreciation rate of ([\d.]+)%[^.]*\.", "depreciation_rate", lambda v: round(float(v) / 100, 6)),
}
OTHER_TOPICS = {
    "required_documents": ("Which documents are required to support a claim?", "Required documents"),
}


def _section_name(section: str) -> str:
    return re.sub(r"^\d+\.\s*", "", section).strip().lower()


def _citation(passage, quote: str) -> Dict[str, Any]:
    sentence = quote.strip().split("\n")[-1].strip()       # the sentence only, without the heading line
    return {"chunk_id": passage.chunk_id, "source_file": passage.source_file, "section": passage.section,
            "score": passage.score, "rank": passage.rank, "quote": sentence, "retriever": passage.retriever}


def _record(result: RetrievalResult, topic: str, store: List[Dict[str, Any]], quarantined: List[Dict[str, Any]]):
    for p in result.passages:
        if not any(x["chunk_id"] == p.chunk_id for x in store):
            store.append({**p.model_dump(exclude={"text"}), "topic": topic, "question": result.question,
                          "excerpt": p.text[:400]})
    for p in result.quarantined:
        quarantined.append({**p.model_dump(exclude={"text"}), "topic": topic, "excerpt": p.text[:200]})


def gather_policy_evidence(claim: Dict[str, Any], incident_cause: str,
                           claim_documents: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Run the investigation questions and return citations, stated terms and all passages."""
    policy_id = claim.get("policy_id")
    passages: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    citations: Dict[str, Dict[str, Any]] = {}
    stated: Dict[str, Any] = {}
    warnings: List[str] = []
    retriever_name = None

    for topic, (question, section, pattern, term, cast) in TERM_TOPICS.items():
        res = retrieve(question, policy_id=policy_id, include_guidelines=False, k=3)
        retriever_name = res.retriever
        warnings += res.warnings
        _record(res, topic, passages, quarantined)
        match_passage = next((p for p in res.passages if _section_name(p.section) == section.lower()), None)
        if match_passage is None:
            continue
        m = re.search(pattern, match_passage.text)
        if m:
            citations[topic] = _citation(match_passage, m.group(0))
            stated[term] = cast(m.group(1).replace(",", ""))

    cause = (incident_cause or claim.get("reported_issue") or "").strip()
    excl = retrieve(f"Is this damage excluded from the warranty: {cause}?", policy_id=policy_id,
                    include_guidelines=False, k=3)
    _record(excl, "exclusions", passages, quarantined)
    excl_passage = next((p for p in excl.passages if _section_name(p.section) == "exclusions"), None)
    if excl_passage is not None:
        listed = [l[2:].strip() for l in excl_passage.text.splitlines() if l.startswith("- ")]
        citations["exclusions"] = _citation(excl_passage, "; ".join(listed))
        stated["exclusions"] = listed

    for topic, (question, section) in OTHER_TOPICS.items():
        res = retrieve(question, policy_id=policy_id, include_guidelines=False, k=3)
        _record(res, topic, passages, quarantined)
        hit = next((p for p in res.passages if _section_name(p.section) == section.lower()), None)
        if hit is not None:
            citations[topic] = _citation(hit, hit.text.split("\n", 1)[-1][:300])

    if claim_documents is not None:
        support = retrieve(f"What caused the problem and what repair is needed: {claim.get('reported_issue', '')}",
                           claim_id=claim.get("claim_id"), claim_documents=claim_documents, k=2)
        _record(support, "claim_support", passages, quarantined)
        warnings += [w for w in support.warnings if w.startswith("skipped")]

    missing = [t for t in TERM_TOPICS if t not in citations]
    return {"retriever": retriever_name, "policy_id": policy_id, "citations": citations, "stated_terms": stated,
            "missing_citations": missing, "passages": passages, "quarantined": quarantined,
            "warnings": sorted(set(warnings))}


def check_policy_citations(rag: Dict[str, Any], policy) -> Dict[str, List[str]]:
    """
    Compare what the retrieved wording states with the structured terms used for the money.
    Returns {"conflicts": [...], "missing": [...]}; the VerifierAgent decides what they mean.
    """
    conflicts, missing = [], list(rag.get("missing_citations", []))
    stated = rag.get("stated_terms", {})
    for topic, (_, section, _, term, _) in TERM_TOPICS.items():
        if term not in stated:
            continue
        expected = getattr(policy, term)
        if abs(float(stated[term]) - float(expected)) > 1e-6:
            cite = rag["citations"][topic]
            conflicts.append(f"Policy wording ({cite['source_file']} §{cite['section']}) states {term}={stated[term]} "
                             f"but the policy terms used for calculation say {expected}")
    for q in rag.get("quarantined", []):
        conflicts.append(f"Retrieved document contains instruction-like text and was quarantined: "
                         f"{q['source_file']} §{q['section']}")
    return {"conflicts": conflicts, "missing": missing}
