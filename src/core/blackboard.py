"""
ClaimSleuth — Blackboard Architecture & Agent-to-Agent (A2A) Messaging
Implements Master Document Section 11 (Blackboard & Shared State Tiers)
and Section 9.4 ("Agents communicate structured objects, not unbounded transcripts").

State Tiers:
1. Case State: Durable claim dossier facts (claim_id, dates, policy limits).
2. Execution State: Step counters, budget limits, loop bounds.
3. Working Memory (Blackboard): Dynamic, short-term scratchpad where specialist agents
   publish intermediate observations, extracted tokens, tool responses, and flags.
4. Durable Case Memory: Append-only audit trail and verified findings.
"""

from typing import List, Dict, Any, Optional, Literal, Callable
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid


class BlackboardTask(BaseModel):
    """A unit of work posted to the Blackboard for a specialist agent to solve."""
    task_id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    goal: str
    assigned_agent: Literal["LeadInvestigator", "WarrantyAgent", "EvidenceAgent", "VerifierAgent", "AnomalyAgent"]
    status: Literal["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "BLOCKED"] = "PENDING"
    dependencies: List[str] = Field(default_factory=list)
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None


class A2AMessage(BaseModel):
    """
    Structured Agent-to-Agent message.
    Adheres to Master Doc Section 9.4: structured communication schema, not conversational chat.
    """
    message_id: str = Field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:8]}")
    sender: str
    receiver: str
    intent: Literal[
        "REQUEST_ACTION",
        "EVIDENCE_DISCOVERED",
        "POLICY_EVALUATED",
        "DISCREPANCY_FLAGGED",
        "VERDICT_ISSUED",
        "REPLAN_TRIGGERED"
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BlackboardFact(BaseModel):
    """An asserted fact with source provenance placed in working memory."""
    key: str
    value: Any
    source_agent: str
    confidence: float = 1.0
    provenance_doc: Optional[str] = None
    raw_snippet: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BlackboardState(BaseModel):
    """Snapshot of the complete Blackboard working memory."""
    case_id: str
    working_facts: Dict[str, BlackboardFact] = Field(default_factory=dict)
    tasks: Dict[str, BlackboardTask] = Field(default_factory=dict)
    messages: List[A2AMessage] = Field(default_factory=list)
    replan_count: int = 0
    active_hypotheses: List[str] = Field(default_factory=list)


class Blackboard:
    """
    Blackboard coordination controller.
    Allows specialist agents to post tasks, retrieve intermediate findings,
    exchange structured A2A messages, and trigger reactive listeners.
    """
    def __init__(self, case_id: str):
        self.state = BlackboardState(case_id=case_id)
        self._listeners: List[Callable[[str, Any], None]] = []

    def add_listener(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for blackboard state changes."""
        self._listeners.append(callback)

    def _notify(self, event_type: str, data: Any) -> None:
        for listener in self._listeners:
            try:
                listener(event_type, data)
            except Exception:
                pass

    # --- Facts / Working Memory ---
    def set_fact(
        self,
        key: str,
        value: Any,
        source_agent: str,
        confidence: float = 1.0,
        provenance_doc: Optional[str] = None,
        raw_snippet: Optional[str] = None
    ) -> BlackboardFact:
        """Post an asserted fact to short-term working memory."""
        fact = BlackboardFact(
            key=key,
            value=value,
            source_agent=source_agent,
            confidence=confidence,
            provenance_doc=provenance_doc,
            raw_snippet=raw_snippet
        )
        self.state.working_facts[key] = fact
        self._notify("FACT_SET", fact)
        return fact

    def get_fact(self, key: str) -> Optional[Any]:
        """Retrieve a fact value by key."""
        fact = self.state.working_facts.get(key)
        return fact.value if fact else None

    def get_fact_object(self, key: str) -> Optional[BlackboardFact]:
        """Retrieve the full fact object with provenance."""
        return self.state.working_facts.get(key)

    def has_fact(self, key: str) -> bool:
        return key in self.state.working_facts

    # --- Task Management ---
    def post_task(
        self,
        goal: str,
        assigned_agent: Literal["LeadInvestigator", "WarrantyAgent", "EvidenceAgent", "VerifierAgent", "AnomalyAgent"],
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> BlackboardTask:
        """Lead Investigator posts a work unit for a specialist."""
        task = BlackboardTask(
            goal=goal,
            assigned_agent=assigned_agent,
            tool_name=tool_name,
            tool_args=tool_args or {},
            dependencies=dependencies or []
        )
        self.state.tasks[task.task_id] = task
        self._notify("TASK_POSTED", task)
        return task

    def update_task_status(
        self,
        task_id: str,
        status: Literal["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "BLOCKED"],
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Optional[BlackboardTask]:
        """Specialist agent marks task progress or completion."""
        task = self.state.tasks.get(task_id)
        if not task:
            return None
        task.status = status
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        if status in ("COMPLETED", "FAILED"):
            task.completed_at = datetime.now(timezone.utc).isoformat()
        self._notify("TASK_UPDATED", task)
        return task

    def get_pending_tasks(self, for_agent: Optional[str] = None) -> List[BlackboardTask]:
        """List tasks ready to execute."""
        tasks = [t for t in self.state.tasks.values() if t.status == "PENDING"]
        if for_agent:
            tasks = [t for t in tasks if t.assigned_agent == for_agent]
        return tasks

    # --- A2A Messaging ---
    def send_message(
        self,
        sender: str,
        receiver: str,
        intent: Literal[
            "REQUEST_ACTION",
            "EVIDENCE_DISCOVERED",
            "POLICY_EVALUATED",
            "DISCREPANCY_FLAGGED",
            "VERDICT_ISSUED",
            "REPLAN_TRIGGERED"
        ],
        payload: Dict[str, Any]
    ) -> A2AMessage:
        """Route a structured message between agents."""
        msg = A2AMessage(
            sender=sender,
            receiver=receiver,
            intent=intent,
            payload=payload
        )
        self.state.messages.append(msg)
        self._notify("MESSAGE_SENT", msg)
        return msg

    def get_messages(
        self,
        receiver: Optional[str] = None,
        intent: Optional[str] = None
    ) -> List[A2AMessage]:
        """Query messages by recipient or intent."""
        msgs = self.state.messages
        if receiver:
            msgs = [m for m in msgs if m.receiver in (receiver, "ALL", "*")]
        if intent:
            msgs = [m for m in msgs if m.intent == intent]
        return msgs

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured summary of blackboard state for serialization."""
        return {
            "case_id": self.state.case_id,
            "fact_count": len(self.state.working_facts),
            "facts": {k: f.model_dump() for k, f in self.state.working_facts.items()},
            "task_count": len(self.state.tasks),
            "tasks": [t.model_dump() for t in self.state.tasks.values()],
            "message_count": len(self.state.messages),
            "messages": [m.model_dump() for m in self.state.messages],
            "replan_count": self.state.replan_count
        }
