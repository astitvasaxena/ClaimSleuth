"""
ClaimSleuth — Execution Monitoring & Callback System
Implements Priority 3: Simple callback-based execution monitoring and error logging.
Reuses existing audit logs without introducing bloated external observability platforms.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from src.core.audit import AuditLogger


logger = logging.getLogger("ClaimSleuth.Monitor")


class StepMetric(BaseModel):
    step_num: int
    action_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    is_success: bool = True
    error_message: Optional[str] = None


class ExecutionMonitor:
    """
    Callback-based lightweight monitor tracking lifecycle events,
    tool execution latency, step bounds, and audit trail synchronization.
    """
    def __init__(self, case_id: str, audit_logger: Optional[AuditLogger] = None):
        self.case_id = case_id
        self.audit_logger = audit_logger
        self.step_metrics: List[StepMetric] = []
        self._current_step_metric: Optional[StepMetric] = None
        self.total_tools_called: int = 0
        self.total_replans: int = 0
        self.errors_logged: List[Dict[str, Any]] = []
        
        # User/agent custom callbacks
        self._on_step_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._on_error_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def register_step_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self._on_step_callbacks.append(cb)

    def register_error_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self._on_error_callbacks.append(cb)

    def on_step_start(self, step_num: int, action_name: str, reasoning: str = "") -> None:
        """Invoked when an investigation cycle starts."""
        metric = StepMetric(
            step_num=step_num,
            action_name=action_name,
            start_time=time.perf_counter()
        )
        self._current_step_metric = metric
        
        event_data = {
            "case_id": self.case_id,
            "step_num": step_num,
            "action": action_name,
            "reasoning": reasoning,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        for cb in self._on_step_callbacks:
            try:
                cb(event_data)
            except Exception as e:
                logger.warning(f"Error in step callback: {e}")

    def on_tool_executed(self, tool_name: str, inputs: Dict[str, Any], output: Dict[str, Any], is_success: bool = True, error: Optional[str] = None) -> None:
        """Invoked after an MCP or deterministic tool finishes."""
        self.total_tools_called += 1
        now = time.perf_counter()
        
        if self._current_step_metric:
            self._current_step_metric.end_time = now
            self._current_step_metric.duration_ms = round((now - self._current_step_metric.start_time) * 1000.0, 2)
            self._current_step_metric.is_success = is_success
            self._current_step_metric.error_message = error
            self.step_metrics.append(self._current_step_metric)
            self._current_step_metric = None

        if not is_success or error:
            err_data = {
                "case_id": self.case_id,
                "tool": tool_name,
                "error": error,
                "inputs": inputs,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.errors_logged.append(err_data)
            for cb in self._on_error_callbacks:
                try:
                    cb(err_data)
                except Exception:
                    pass

    def on_replan(self, step_num: int, reason: str) -> None:
        """Invoked when adaptive replanning triggers."""
        self.total_replans += 1
        logger.info(f"[{self.case_id}] Adaptive Replan #{self.total_replans} at step {step_num}: {reason}")

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Aggregate execution statistics."""
        total_duration = sum(m.duration_ms or 0.0 for m in self.step_metrics)
        avg_latency = round(total_duration / max(1, len(self.step_metrics)), 2)
        
        return {
            "case_id": self.case_id,
            "steps_executed": len(self.step_metrics),
            "tools_called": self.total_tools_called,
            "replans_triggered": self.total_replans,
            "error_count": len(self.errors_logged),
            "total_tool_duration_ms": round(total_duration, 2),
            "avg_step_latency_ms": avg_latency,
            "step_breakdown": [m.model_dump() for m in self.step_metrics]
        }
