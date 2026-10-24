import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, Tuple, Optional
from configs.settings import POLICIES_FILE
from src.core.state import PolicyTerms

def load_policy(policy_id: str) -> Optional[PolicyTerms]:
    """Deterministically loads policy terms from configs/policies.json."""
    if not POLICIES_FILE.exists():
        raise FileNotFoundError(f"Policy file not found at {POLICIES_FILE}")
        
    with open(POLICIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if policy_id not in data:
        return None
    return PolicyTerms(**data[policy_id])

def calc_coverage_window(purchase_date_str: str, incident_date_str: str, duration_months: int) -> Dict[str, Any]:
    """
    Pure Python date calculations. Zero generative hallucination.
    Evaluates whether incident_date falls strictly between purchase_date and warranty expiration.
    """
    try:
        p_date = datetime.strptime(purchase_date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return {"is_valid": False, "error": f"Invalid purchase date format: {purchase_date_str}. Expected YYYY-MM-DD"}
        
    try:
        i_date = datetime.strptime(incident_date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return {"is_valid": False, "error": f"Invalid incident date format: {incident_date_str}. Expected YYYY-MM-DD"}
        
    if i_date < p_date:
        return {
            "is_valid": True,
            "is_covered": False,
            "reason": f"Incident date ({i_date}) precedes purchase date ({p_date})",
            "days_diff": (p_date - i_date).days
        }
        
    expiration_date = p_date + relativedelta(months=duration_months)
    is_covered = i_date <= expiration_date
    delta_days = (expiration_date - i_date).days
    
    return {
        "is_valid": True,
        "is_covered": is_covered,
        "purchase_date": p_date.isoformat(),
        "incident_date": i_date.isoformat(),
        "expiration_date": expiration_date.isoformat(),
        "days_remaining_or_overdue": delta_days,
        "reason": "Incident is within active warranty period" if is_covered else f"Warranty expired {abs(delta_days)} days before incident date"
    }

def check_exclusions(reported_cause: str, exclusions: list) -> Tuple[bool, Optional[str]]:
    """
    Deterministic substring & keyword check for common explicit warranty exclusions.
    """
    reported_lower = reported_cause.lower()
    for exc in exclusions:
        if exc.lower() in reported_lower:
            return True, exc
    return False, None
