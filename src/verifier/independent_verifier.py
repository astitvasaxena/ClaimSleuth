from typing import Dict, Any, List
from src.core.state import ClaimInvestigationState, VerifierReport, PolicyTerms
from src.tools.policy_rules import calc_coverage_window, check_exclusions
from src.tools.math_engine import compute_settlement

class DualGateVerifier:
    """
    Independent Verification Engine.
    Executes zero-trust checks against the Lead Investigator's findings.
    Automated decisions are forbidden if any assertion fails.
    """

    def verify_investigation(
        self,
        state: ClaimInvestigationState,
        policy: PolicyTerms
    ) -> VerifierReport:
        discrepancies: List[str] = []
        
        # ----------------------------------------------------
        # GATE 1: EVIDENCE SUBSTANTIATION & CONFLICT CHECK
        # ----------------------------------------------------
        gate_1_passed = True
        
        # 1. Unresolved conflicts check
        if state.conflicts:
            discrepancies.append(f"Unresolved evidence conflicts present: {'; '.join(state.conflicts)}")
            gate_1_passed = False
            
        # 2. Required evidence presence
        required_fields = ["purchase_date", "serial_number", "repair_cost"]
        for rf in required_fields:
            if rf not in state.evidence:
                discrepancies.append(f"Missing required evidence item: '{rf}'")
                gate_1_passed = False
            else:
                ev = state.evidence[rf]
                if ev.status != "VERIFIED":
                    discrepancies.append(f"Evidence '{rf}' status is {ev.status}, not VERIFIED")
                    gate_1_passed = False
                if ev.confidence < 0.85:
                    discrepancies.append(f"Evidence '{rf}' confidence ({ev.confidence}) below required threshold 0.85")
                    gate_1_passed = False

        # ----------------------------------------------------
        # GATE 2: DETERMINISTIC RULES & ARITHMETIC VALIDATION
        # ----------------------------------------------------
        gate_2_rules_passed = True
        gate_2_math_passed = True
        
        # 1. Coverage Window check
        if "purchase_date" in state.evidence:
            p_date_str = str(state.evidence["purchase_date"].extracted_value)
            cov_result = calc_coverage_window(p_date_str, state.incident_date, policy.duration_months)
            
            if not cov_result.get("is_valid"):
                discrepancies.append(f"Date check error: {cov_result.get('error')}")
                gate_2_rules_passed = False
            elif not cov_result.get("is_covered"):
                discrepancies.append(f"Policy coverage expired: {cov_result.get('reason')}")
                gate_2_rules_passed = False
                
        # 2. Policy Exclusion check
        incident_cause = state.reported_issue
        if "incident_cause" in state.evidence:
            incident_cause += f" {state.evidence['incident_cause'].extracted_value}"
            
        is_excluded, matched_exclusion = check_exclusions(incident_cause, policy.exclusions)
        if is_excluded:
            discrepancies.append(f"Claim matches excluded peril: '{matched_exclusion}'")
            gate_2_rules_passed = False

        # 3. Arithmetic & Financial Check
        if "repair_cost" in state.evidence:
            extracted_cost = float(state.evidence["repair_cost"].extracted_value)
            recalculated = compute_settlement(
                repair_cost=extracted_cost,
                deductible=policy.deductible,
                max_coverage_limit=policy.max_coverage_limit,
                depreciation_rate=policy.depreciation_rate
            )
            
            if state.deterministic_settlement:
                claimed_payout = state.deterministic_settlement.approved_payout
                if abs(claimed_payout - recalculated.approved_payout) > 0.009:
                    discrepancies.append(
                        f"Arithmetic mismatch: Agent proposed payout ${claimed_payout:.2f} "
                        f"but recalculated is ${recalculated.approved_payout:.2f}"
                    )
                    gate_2_math_passed = False
            else:
                # If agent forgot to attach settlement, verifier attaches the recalculation
                state.deterministic_settlement = recalculated

        overall_approved = gate_1_passed and gate_2_rules_passed and gate_2_math_passed

        return VerifierReport(
            is_approved_by_verifier=overall_approved,
            gate_1_evidence_substantiated=gate_1_passed,
            gate_2_deterministic_rules_passed=gate_2_rules_passed,
            gate_2_arithmetic_matches=gate_2_math_passed,
            discrepancies=discrepancies
        )
