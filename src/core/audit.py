import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from configs.settings import AUDIT_LOG_DIR

class AuditLogger:
    """Immutable append-only audit trail logger for claim investigations."""
    
    def __init__(self, claim_id: str):
        self.claim_id = claim_id
        self.log_file = AUDIT_LOG_DIR / f"{claim_id}_audit_trail.jsonl"
        self._last_hash = "GENESIS"
        
        # Initialize if not exists
        if not self.log_file.exists():
            self._write_entry("AUDIT_TRAIL_INITIALIZED", {"claim_id": claim_id})

    def _compute_hash(self, prev_hash: str, payload_str: str) -> str:
        return hashlib.sha256(f"{prev_hash}|{payload_str}".encode("utf-8")).hexdigest()[:16]

    def _write_entry(self, event_type: str, details: Dict[str, Any]):
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp": timestamp,
            "claim_id": self.claim_id,
            "event_type": event_type,
            "details": details,
            "prev_hash": self._last_hash
        }
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        entry_hash = self._compute_hash(self._last_hash, payload_str)
        payload["entry_hash"] = entry_hash
        self._last_hash = entry_hash
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def log_step(self, step_num: int, phase: str, description: str, tool: str = None, reasoning: str = None):
        self._write_entry("INVESTIGATION_STEP", {
            "step_num": step_num,
            "phase": phase,
            "description": description,
            "tool": tool,
            "reasoning": reasoning
        })

    def log_tool_execution(self, tool_name: str, inputs: Dict[str, Any], output: Any):
        self._write_entry("TOOL_CALL", {
            "tool_name": tool_name,
            "inputs": inputs,
            "output": output
        })

    def log_evidence_extracted(self, evidence_id: str, field: str, value: Any, confidence: float, source: str):
        self._write_entry("EVIDENCE_CAPTURED", {
            "evidence_id": evidence_id,
            "field": field,
            "value": value,
            "confidence": confidence,
            "source": source
        })

    def log_conflict(self, conflict_desc: str):
        self._write_entry("CONFLICT_DETECTED", {"conflict": conflict_desc})

    def log_verifier_result(self, is_approved: bool, discrepancies: list):
        self._write_entry("VERIFIER_ADJUDICATION", {
            "is_approved": is_approved,
            "discrepancies": discrepancies
        })

    def log_human_decision(self, decision: str, notes: str = None):
        self._write_entry("HUMAN_DECISION", {
            "decision": decision,
            "notes": notes
        })

    def get_trail(self) -> list:
        if not self.log_file.exists():
            return []
        entries = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        return entries

    def verify_chain(self) -> dict:
        """
        Validates the integrity of the entire audit chain.
        Returns {"valid": True/False, "entries_checked": N, "error": str|None}.
        """
        entries = self.get_trail()
        if not entries:
            return {"valid": True, "entries_checked": 0, "error": None}

        prev_hash = "GENESIS"
        for idx, entry in enumerate(entries):
            stored_hash = entry.pop("entry_hash", None)
            if stored_hash is None:
                return {
                    "valid": False,
                    "entries_checked": idx,
                    "error": f"Entry {idx} missing entry_hash"
                }
            # Recompute: the payload was hashed *without* entry_hash in it
            payload_str = json.dumps(entry, sort_keys=True, default=str)
            expected_hash = self._compute_hash(prev_hash, payload_str)
            if stored_hash != expected_hash:
                return {
                    "valid": False,
                    "entries_checked": idx,
                    "error": f"Hash mismatch at entry {idx}: expected {expected_hash}, found {stored_hash}"
                }
            prev_hash = stored_hash

        return {"valid": True, "entries_checked": len(entries), "error": None}
