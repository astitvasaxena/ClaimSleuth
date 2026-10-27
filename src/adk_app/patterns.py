"""
Agentic design patterns and messaging architectures for the ADK pipeline.

PATTERNS (C-4), all built as real google.adk Workflow graphs:
  sequential    Intake -> Evidence -> Anomaly -> Warranty -> Settlement -> Verifier -> Report
  parallel      Intake -> Evidence -> (Anomaly || Warranty [|| CrewAI via A2A]) -> join -> MergeAgent -> ...
                The branches run concurrently. Each branch writes only bb:branch_out:<Agent>; the
                MergeAgent then combines them, so concurrent writes to shared lists cannot be lost.
  hierarchical  ClaimManager (deterministic manager) delegates to worker agents, which report back
                to it; it picks the next worker from the evolving case state (for example, it skips
                Warranty and Settlement once a conflict exists) and records every decision.

MESSAGING (C-1): what each agent is allowed to SEE when it runs.
  blackboard    the whole shared session state
  direct        only the messages explicitly routed to it (peer-to-peer, see DIRECT_ROUTES)
  broadcast     every message published by every agent so far
Every agent's output is still recorded in session state (the system of record for the
report and audit); the mode controls the context an agent receives. Bookkeeping keys
(INFRA_KEYS) are always visible. Model mode supports blackboard only, because the
InvestigatorLLM writes through ADK tools rather than through this message layer.

The deterministic verifier, settlement and adjudication rules are identical in every combination.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

from google.adk import Workflow
from google.adk.agents import BaseAgent
from google.adk.workflow import START, FunctionNode, JoinNode

from src.adk_app import steps
from src.core.audit import AuditLogger

Pattern = Literal["sequential", "parallel", "hierarchical"]
Messaging = Literal["blackboard", "direct", "broadcast"]

# Always visible: identifiers, bookkeeping and the outputs of ADK-native agents (model / A2A).
INFRA_KEYS = {"bb:claim", "bb:run", "bb:agent_sequence", "bb:tool_calls", "bb:tool_errors", "bb:agent_timings",
              "bb:forensics_raw", "bb:forensics_error", "bb:a2a_call", "bb:model_tool_calls",
              "bb:model_payout_claim", "bb:model_summary", "bb:manager_decisions"}
# Never part of an agent's context (large or purely for evaluation)
HIDDEN_KEYS = {"bb:messages", "bb:context_stats"}

MANAGER = "ClaimManager"
# Peer-to-peer routing table: producer -> agents that receive its messages.
DIRECT_ROUTES: Dict[str, List[str]] = {
    "IntakeAgent": ["EvidenceAgent", "PolicyRetrievalAgent", "AnomalyAgent", "WarrantyAgent", "SettlementAgent",
                    "ForensicsReviewAgent", "VerifierAgent", "ReportAgent", MANAGER],
    "EvidenceAgent": ["PolicyRetrievalAgent", "AnomalyAgent", "WarrantyAgent", "SettlementAgent",
                      "ForensicsReviewAgent", "VerifierAgent", "ReportAgent", MANAGER],
    "PolicyRetrievalAgent": ["VerifierAgent", "ReportAgent", MANAGER],
    "AnomalyAgent": ["ForensicsReviewAgent", "SettlementAgent", "VerifierAgent", "ReportAgent", MANAGER],
    "ForensicsReviewAgent": ["SettlementAgent", "VerifierAgent", "ReportAgent", MANAGER],
    "WarrantyAgent": ["SettlementAgent", "ReportAgent", MANAGER],
    "SettlementAgent": ["VerifierAgent", "ReportAgent", MANAGER],
    "VerifierAgent": ["ReportAgent", MANAGER],
    "ReportAgent": [],
}
MAX_MANAGER_DECISIONS = 12


@dataclass
class RunOptions:
    pattern: Pattern = "sequential"
    messaging: Messaging = "blackboard"
    drop_routes: Set[Tuple[str, str]] = field(default_factory=set)   # failure injection for "direct"
    rag: bool = True                                                  # run PolicyRetrievalAgent (RAG)


def agent_name(step: Callable) -> str:
    """intake_agent -> IntakeAgent"""
    return "".join(part.capitalize() for part in step.__name__.split("_"))


def _elapsed_ms(state: Dict[str, Any]) -> float:
    return round((time.perf_counter() - state["bb:run"]["t0"]) * 1000, 1)


# ─── Messaging ─────────────────────────────────────────────────────────────

def _delivered(msg: Dict[str, Any], agent: str, opts: RunOptions) -> bool:
    if opts.messaging == "broadcast":
        return True
    return agent in DIRECT_ROUTES.get(msg["from"], []) and (msg["from"], agent) not in opts.drop_routes


def build_view(state: Dict[str, Any], agent: str, opts: RunOptions) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The context `agent` receives under the messaging mode, plus statistics about it."""
    if opts.messaging == "blackboard":
        # Parallel branch outputs are private until the MergeAgent applies them
        view = {k: v for k, v in state.items() if k not in HIDDEN_KEYS and not k.startswith("bb:branch_out:")}
        received = 0
    else:
        view = {k: state[k] for k in INFRA_KEYS if k in state}
        received = 0
        for msg in state.get("bb:messages", []):
            if _delivered(msg, agent, opts):
                view.update(msg["payload"])
                received += 1
    if agent == "ReportAgent":
        view["bb:messages"] = state.get("bb:messages", [])
        view["bb:context_stats"] = state.get("bb:context_stats", [])
    context = {k: v for k, v in view.items() if k not in INFRA_KEYS and k not in HIDDEN_KEYS}
    stats = {"agent": agent, "messages_received": received, "keys_visible": len(context),
             "bytes_visible": len(json.dumps(context, default=str))}
    return view, stats


def _message(agent: str, delta: Dict[str, Any], opts: RunOptions) -> Optional[Dict[str, Any]]:
    if opts.messaging == "blackboard":
        return None
    payload = {k: v for k, v in delta.items() if k not in INFRA_KEYS and k not in HIDDEN_KEYS}
    to = "*" if opts.messaging == "broadcast" else [a for a in DIRECT_ROUTES.get(agent, [])
                                                    if (agent, a) not in opts.drop_routes]
    return {"from": agent, "to": to, "keys": sorted(payload), "payload": payload,
            "bytes": len(json.dumps(payload, default=str))}


async def execute_step(step: Callable, state: Dict[str, Any], gateway, opts: RunOptions,
                       branch: bool = False) -> Dict[str, Any]:
    """
    Run one deterministic agent with the context its messaging mode allows, then return the
    state delta: its own output plus bookkeeping (tool calls, timing, context stats, message).
    In a parallel branch everything goes to bb:branch_out:<Agent> for the MergeAgent.
    """
    name = agent_name(step)
    view, stats = build_view(state, name, opts)
    if name == "ReportAgent":
        # The report summarises context statistics, so it must include its own entry
        view["bb:context_stats"] = list(state.get("bb:context_stats", [])) + [stats]
    start = _elapsed_ms(state)
    delta = await step(view, gateway)
    end = _elapsed_ms(state)

    delta["bb:tool_calls"] = list(state.get("bb:tool_calls", [])) + [{**c, "agent": name} for c in gateway.calls]
    delta["bb:agent_timings"] = list(state.get("bb:agent_timings", [])) + [
        {"agent": name, "start_ms": start, "end_ms": end, "branch": branch}]
    delta["bb:context_stats"] = list(state.get("bb:context_stats", [])) + [stats]
    msg = _message(name, delta, opts)
    if msg is not None:
        delta["bb:messages"] = list(state.get("bb:messages", [])) + [msg]
    if branch:
        return {f"bb:branch_out:{name}": delta}
    return delta


# ─── Parallel: merge node ────────────────────────────────────────────────────

LIST_KEYS = {"bb:conflicts", "bb:tool_errors", "bb:agent_sequence", "bb:tool_calls", "bb:agent_timings",
             "bb:context_stats", "bb:messages"}


def merge_branch_outputs(state: Dict[str, Any], branch_agents: List[str]) -> Dict[str, Any]:
    """
    Combine the parallel branches' outputs. Shared lists are united in a fixed order (base
    first, then each branch's new items); other keys must not be written by two branches with
    different values: if they are, the claim fails closed with a recorded merge conflict.
    """
    merged: Dict[str, Any] = {}
    for key in LIST_KEYS:
        merged[key] = list(state.get(key, []))
    owner: Dict[str, str] = {}
    for agent in branch_agents:
        out = state.get(f"bb:branch_out:{agent}")
        if out is None:
            merged["bb:tool_errors"].append(f"MergeAgent: branch {agent} produced no output")
            continue
        for key, value in out.items():
            if key in LIST_KEYS:
                merged[key] += [item for item in value if item not in merged[key]]
            elif key in merged and merged[key] != value:
                merged["bb:tool_errors"].append(
                    f"MergeAgent: branches {owner[key]} and {agent} wrote different values for {key}")
            else:
                merged[key] = value
                owner[key] = agent
    merged["bb:agent_sequence"].append("MergeAgent")
    for agent in branch_agents:
        merged[f"bb:branch_out:{agent}"] = None   # applied; do not leave duplicates in shared state
    return merged


# ─── Hierarchical: manager ──────────────────────────────────────────────────

def manager_decision(view: Dict[str, Any], forensics: bool, decisions: List[Dict[str, Any]],
                     rag: bool = False) -> Tuple[str, str]:
    """The manager's delegation policy: (route, reason). Pure function of the case state."""
    done = set(view.get("bb:agent_sequence", []))
    conflicts = view.get("bb:conflicts", [])
    elig = view.get("bb:eligibility") or {}
    if len(decisions) >= MAX_MANAGER_DECISIONS:
        return "verify", "decision budget exhausted; verifying with what is known (fail closed)"
    if (view.get("bb:guardrail") or {}).get("action") == "block":
        return "verify", "blocked by input guardrail: no further investigation, escalate to a human"
    if "EvidenceAgent" not in done and "InvestigatorLLM" not in done:
        if not view.get("bb:documents"):
            return "verify", "no documents in the dossier; nothing to investigate"
        return "evidence", "evidence not gathered yet"
    if rag and "PolicyRetrievalAgent" not in done:
        return "retrieval", "policy wording not retrieved yet (RAG)"
    if "AnomalyAgent" not in done:
        return "anomaly", "cross-document checks not run yet"
    if forensics and "ForensicsReviewAgent" not in done:
        return "forensics", "independent CrewAI forensic review requested"
    if conflicts:
        return "verify", f"{len(conflicts)} unresolved conflict(s): skip eligibility and settlement, escalate"
    if "WarrantyAgent" not in done:
        return "warranty", "eligibility not evaluated yet"
    if elig.get("covered") is not True or elig.get("excluded"):
        return "verify", "claim not eligible: skip settlement"
    if "SettlementAgent" not in done:
        return "settlement", "eligible and conflict-free: request a settlement proposal"
    return "verify", "all specialist work done"


def _manager_node(opts: RunOptions, forensics: bool) -> FunctionNode:
    rag = opts.rag
    async def claim_manager(ctx):
        state = ctx.state.to_dict()
        view, stats = build_view(state, MANAGER, opts)
        decisions = list(state.get("bb:manager_decisions", []))
        route, reason = manager_decision(view, forensics, decisions, rag)
        decisions.append({"n": len(decisions) + 1, "route": route, "reason": reason, "at_ms": _elapsed_ms(state)})
        ctx.state["bb:manager_decisions"] = decisions
        ctx.state["bb:context_stats"] = list(state.get("bb:context_stats", [])) + [stats]
        AuditLogger(state["bb:claim"]["claim_id"]).log_step(len(decisions), "PLAN",
                                                             f"[{state['bb:run']['run_id']}] {MANAGER}: {route} ({reason})")
        ctx.route = route
        return route
    return FunctionNode(func=claim_manager, name=MANAGER)


# ─── Graph builders ─────────────────────────────────────────────────────────

GatewayFactory = Callable[[Any], Any]


def step_node(step: Callable, make_gateway: GatewayFactory, opts: RunOptions, branch: bool = False) -> FunctionNode:
    async def run_step(ctx):
        delta = await execute_step(step, ctx.state.to_dict(), make_gateway(ctx), opts, branch=branch)
        for key, value in delta.items():
            ctx.state[key] = value
        return agent_name(step)
    return FunctionNode(func=run_step, name=agent_name(step))


def _merge_node(branch_agents: List[str]) -> FunctionNode:
    async def merge_agent(ctx):
        state = ctx.state.to_dict()
        start = _elapsed_ms(state)
        merged = merge_branch_outputs(state, branch_agents)
        merged["bb:agent_timings"].append({"agent": "MergeAgent", "start_ms": start, "end_ms": _elapsed_ms(state),
                                           "branch": False})
        for key, value in merged.items():
            ctx.state[key] = value
        return "MergeAgent"
    return FunctionNode(func=merge_agent, name="MergeAgent")


def build_pattern_workflow(opts: RunOptions, make_gateway: GatewayFactory,
                           model_agent: Optional[BaseAgent] = None,
                           forensics_agent: Optional[BaseAgent] = None) -> Workflow:
    node = lambda s, branch=False: step_node(s, make_gateway, opts, branch)
    gather = model_agent if model_agent is not None else node(steps.evidence_agent)
    tail = [node(steps.settlement_agent), node(steps.verifier_agent), node(steps.report_agent)]

    retrieval = [node(steps.policy_retrieval_agent)] if opts.rag else []
    if opts.pattern == "sequential":
        chain = [node(steps.intake_agent), gather, *retrieval, node(steps.anomaly_agent)]
        if forensics_agent is not None:
            chain += [forensics_agent, node(steps.forensics_review_agent)]
        chain += [node(steps.warranty_agent)] + tail
        return Workflow(name="ClaimInvestigationWorkflow", edges=[(START, *chain)])

    if opts.pattern == "parallel":
        branches = [node(steps.anomaly_agent, True), node(steps.warranty_agent, True)]
        if opts.rag:
            branches.append(node(steps.policy_retrieval_agent, True))
        if forensics_agent is not None:
            branches.append(forensics_agent)
        merge = _merge_node(["AnomalyAgent", "WarrantyAgent"] + (["PolicyRetrievalAgent"] if opts.rag else []))
        chain = [node(steps.intake_agent), gather, tuple(branches), JoinNode(name="ParallelJoin"), merge]
        if forensics_agent is not None:
            chain.append(node(steps.forensics_review_agent))
        return Workflow(name="ClaimInvestigationParallel", edges=[(START, *chain + tail)])

    if opts.pattern == "hierarchical":
        manager = _manager_node(opts, forensics_agent is not None)
        workers = {"evidence": gather, "anomaly": node(steps.anomaly_agent),
                   "warranty": node(steps.warranty_agent), "settlement": node(steps.settlement_agent)}
        if opts.rag:
            workers["retrieval"] = node(steps.policy_retrieval_agent)
        verifier, report = node(steps.verifier_agent), node(steps.report_agent)
        routes: Dict[str, Any] = {**workers, "verify": verifier}
        edges = [(START, node(steps.intake_agent), manager)]
        if forensics_agent is not None:
            review = node(steps.forensics_review_agent)
            routes["forensics"] = forensics_agent
            edges += [(forensics_agent, review), (review, manager)]
        edges.append((manager, routes))
        edges += [(w, manager) for w in workers.values()]
        edges.append((verifier, report))
        return Workflow(name="ClaimInvestigationHierarchical", edges=edges)

    raise ValueError(f"Unknown pattern '{opts.pattern}'")
