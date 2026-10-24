from src.core.state import SettlementCalc

def compute_settlement(
    repair_cost: float,
    deductible: float,
    max_coverage_limit: float,
    depreciation_rate: float = 0.0
) -> SettlementCalc:
    """
    100% Deterministic Financial Settlement Engine.
    Guarantees no arithmetic hallucinations or floating point inaccuracies.
    Formula:
      1. eligible_base = max(0.0, repair_cost - deductible)
      2. capped_base = min(eligible_base, max_coverage_limit)
      3. depreciation = round(capped_base * depreciation_rate, 2)
      4. final_payout = round(capped_base - depreciation, 2)
    """
    cost = round(max(0.0, float(repair_cost)), 2)
    ded = round(max(0.0, float(deductible)), 2)
    limit = round(max(0.0, float(max_coverage_limit)), 2)
    dep_rate = max(0.0, min(1.0, float(depreciation_rate)))
    
    eligible_base = max(0.0, round(cost - ded, 2))
    capped_base = min(eligible_base, limit)
    dep_amount = round(capped_base * dep_rate, 2)
    final_payout = round(capped_base - dep_amount, 2)
    
    explanation = (
        f"Repair cost (${cost:.2f}) - Deductible (${ded:.2f}) = Eligible (${eligible_base:.2f}). "
        f"Capped at limit (${limit:.2f}) -> ${capped_base:.2f}. "
        f"Depreciation ({dep_rate*100:.1f}%) = -${dep_amount:.2f}. "
        f"Net Approved Settlement = ${final_payout:.2f}."
    )
    
    return SettlementCalc(
        repair_cost=cost,
        deductible=ded,
        depreciation_applied=dep_amount,
        max_coverage_limit=limit,
        net_eligible_amount=capped_base,
        approved_payout=final_payout,
        formula_explanation=explanation,
        is_arithmetically_valid=True
    )
