import json
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from claimsleuth.config import AUDIT_LOG_DIR

# One lock per log file, shared by every AuditLogger instance in this process,
# so concurrent writers (e.g. two API requests) cannot fork the hash chain.
_FILE_LOCKS: Dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _current_trace_id():
    """The investigation's trace id, if a run is in progress (see src/core/tracing.py)."""
    try:
        from claimsleuth.tracing import CURRENT_TRACE_ID
        return CURRENT_TRACE_ID.get()
    except Exception:
        return None


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(key, threading.Lock())


class AuditLogger:
    """Immutable append-only audit trail logger for claim investigations."""

    def __init__(self, claim_id: str):
        # Read the directory at construction time so CLAIMSLEUTH_AUDIT_LOG_DIR
        # and test patches of this module attribute are both honoured.
        self.claim_id = claim_id
        self.log_file = Path(AUDIT_LOG_DIR) / f"{claim_id}_audit_trail.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        with _lock_for(self.log_file):
            if not self.log_file.exists():
                self._append_locked("AUDIT_TRAIL_INITIALIZED", {"claim_id": claim_id})

    def _compute_hash(self, prev_hash: str, payload_str: str) -> str:
        return hashlib.sha256(f"{prev_hash}|{payload_str}".encode("utf-8")).hexdigest()[:16]

    def _read_last_hash(self) -> str:
        """Return entry_hash of the last non-empty line, or GENESIS for a new/empty log."""
        if not self.log_file.exists():
            return "GENESIS"
        with open(self.log_file, "rb") as f:
            f.seek(0, 2)
            pos = f.tell()
            buf = b""
            # Read backwards in blocks until we hold one complete non-empty line
            while pos > 0:
                step = min(4096, pos)
                pos -= step
                f.seek(pos)
                buf = f.read(step) + buf
                lines = [ln for ln in buf.split(b"\n") if ln.strip()]
                if len(lines) >= 2 or (lines and pos == 0):
                    return json.loads(lines[-1].decode("utf-8")).get("entry_hash", "GENESIS")
        return "GENESIS"

    def _write_entry(self, event_type: str, details: Dict[str, Any]):
        with _lock_for(self.log_file):
            self._append_locked(event_type, details)

    def _append_locked(self, event_type: str, details: Dict[str, Any]):
        """
        Append one entry chained to the current on-disk head. Caller holds the file lock.
        Reading the head from disk (not from memory) is the D-3 fix: a fresh logger
        for an existing claim continues the chain instead of restarting at GENESIS.
        """
        prev_hash = self._read_last_hash()
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp": timestamp,
            "claim_id": self.claim_id,
            "event_type": event_type,
            "details": details,
            "prev_hash": prev_hash
        }
        trace_id = _current_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        payload["entry_hash"] = self._compute_hash(prev_hash, payload_str)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def count_entries(self) -> int:
        if not self.log_file.exists():
            return 0
        with open(self.log_file, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

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

    def log_human_decision_blocked(self, decision: str, from_status: str, reason: str):
        self._write_entry("HUMAN_DECISION_BLOCKED", {
            "decision": decision,
            "from_status": from_status,
            "reason": reason
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


def verify_log_file(log_file: Path) -> dict:
    """
    Read-only chain verification of any ledger file (used for the legacy logs).
    Unlike AuditLogger(...), this never creates or appends to the file.
    """
    reader = AuditLogger.__new__(AuditLogger)
    reader.claim_id = Path(log_file).name.replace("_audit_trail.jsonl", "")
    reader.log_file = Path(log_file)
    return reader.verify_chain()
