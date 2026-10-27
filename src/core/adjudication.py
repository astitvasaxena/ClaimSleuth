"""
Shared adjudication rules used by every execution path (the original
LeadInvestigatorAgent loop and the Google ADK pipeline), so both reach the
same status and describe payouts the same way.
"""
from typing import Any, Dict, Tuple

from configs.settings import HIGH_VALUE_THRESHOLD
from src.core.state import ClaimInvestigationState, VerifierReport


def determine_status(state: ClaimInvestigationState, report: VerifierReport) -> Tuple[str, bool]:
    """
    Map evidence conflicts and the verifier report to (status, requires_human_approval).
    Fail closed: anything the verifier did not approve needs a human.
    """
    # Guardrail blocks win over everything: such a claim is never payable or approvable.
    if any(flag.startswith("BLOCK:") for flag in state.guardrail_flags):
        return "BLOCKED_BY_GUARDRAIL", True
    if state.conflicts:
        return "CONFLICT_ESCALATED", True
    if not report.is_approved_by_verifier:
        # A deterministic rule violation (outside the coverage window, or an excluded peril) is a
        # rejection even when other things are also missing: e.g. an excluded claim never gets a
        # settlement proposal, so "missing settlement" must not turn it into NEEDS_EVIDENCE.
        if any("expired" in d.lower() or "excluded peril" in d.lower() for d in report.discrepancies):
            return "REJECTED_DETERMINISTICALLY", True
        if any("missing" in d.lower() for d in report.discrepancies):
            return "NEEDS_EVIDENCE", True
        return "REJECTED_DETERMINISTICALLY", True
    payout = state.deterministic_settlement.approved_payout if state.deterministic_settlement else 0.0
    return "VERIFIED_PENDING_APPROVAL", payout >= HIGH_VALUE_THRESHOLD


def payout_view(state: ClaimInvestigationState) -> Dict[str, Any]:
    """
    The single source of truth for how a payout may be described.
    - AUTHORIZED: a human approved a verifier-approved settlement.
    - PROPOSED_PENDING_APPROVAL: verifier-approved, but nothing is paid until a human approves.
    - NOT_PAYABLE: conflicted, rejected or incomplete. Any figure is a diagnostic
      calculation only (diagnostic_calculation) and is never presented as a payout.
    """
    settlement = state.deterministic_settlement
    report = state.verifier_report
    verified = bool(report and report.is_approved_by_verifier and settlement is not None)

    if state.status == "CLOSED_SETTLED" and verified:
        return {
            "payout_status": "AUTHORIZED",
            "authorized_payout": settlement.approved_payout,
            "proposed_payout": settlement.approved_payout,
            "diagnostic_calculation": None,
            "explanation": "Settlement approved by a human reviewer.",
        }
    if state.status == "VERIFIED_PENDING_APPROVAL" and verified:
        return {
            "payout_status": "PROPOSED_PENDING_APPROVAL",
            "authorized_payout": None,
            "proposed_payout": settlement.approved_payout,
            "diagnostic_calculation": None,
            "explanation": "Verifier-approved proposal. Nothing is paid until a human approves it.",
        }

    diagnostic = settlement or (report.recalculated_settlement if report else None)
    reasons = {
        "CONFLICT_ESCALATED": "Unresolved evidence conflict: escalated to human review.",
        "REJECTED_DETERMINISTICALLY": "Claim failed deterministic eligibility rules.",
        "CLOSED_REJECTED": "Claim rejected by a human reviewer.",
        "NEEDS_EVIDENCE": "More evidence is required before any settlement can be proposed.",
        "BLOCKED_BY_GUARDRAIL": "Blocked by an input guardrail (invalid input, prompt injection or threat): "
                                "a human must review it; it cannot be approved.",
    }
    explanation = reasons.get(state.status, f"No payable settlement in status {state.status}.")
    if diagnostic is not None:
        explanation += " The figure shown is a diagnostic calculation only, not an approved or proposed payout."
    return {
        "payout_status": "NOT_PAYABLE",
        "authorized_payout": None,
        "proposed_payout": None,
        "diagnostic_calculation": diagnostic.model_dump() if diagnostic is not None else None,
        "explanation": explanation,
    }
