from typing import Dict, Any, List

def flag_serial_anomalies(
    claim_serial: str,
    invoice_serial: str,
    diagnostic_serial: str,
    known_prior_claims: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Deterministic rule-based anomaly & fraud detector.
    Detects mismatched serial numbers across untrusted submission documents
    and repeat claims within cooling-off periods.
    """
    anomalies = []
    
    clean_claim = (claim_serial or "").strip().upper()
    clean_inv = (invoice_serial or "").strip().upper()
    clean_diag = (diagnostic_serial or "").strip().upper()
    
    # 1. Cross-document serial consistency check
    if clean_inv and clean_claim and clean_inv != clean_claim:
        anomalies.append(
            f"Serial mismatch: Claim specifies '{clean_claim}' but purchase invoice states '{clean_inv}'"
        )
        
    if clean_diag and clean_claim and clean_diag != clean_claim:
        anomalies.append(
            f"Serial mismatch: Claim specifies '{clean_claim}' but diagnostic repair report states '{clean_diag}'"
        )
        
    if clean_inv and clean_diag and clean_inv != clean_diag:
        anomalies.append(
            f"Inter-document conflict: Purchase invoice serial '{clean_inv}' differs from repair report serial '{clean_diag}'"
        )
        
    # 2. Prior claim history duplication check
    if known_prior_claims and clean_claim:
        for prior in known_prior_claims:
            if prior.get("serial_number", "").upper() == clean_claim:
                anomalies.append(
                    f"Duplicate claim history: Serial '{clean_claim}' was already filed under Claim ID {prior.get('claim_id')}"
                )
                
    return {
        "has_anomalies": len(anomalies) > 0,
        "anomalies": anomalies,
        "clean_serial": clean_claim
    }
