"""
Google ADK execution path for ClaimSleuth.

run_adk_investigation() runs the claim through a real google.adk Runner with an
in-memory session. The session state is the Blackboard (see steps.py).

Two interchangeable backends run the SAME step functions:
  - "workflow":   google.adk.Workflow graph of FunctionNodes (preferred; the
                  graph API ADK 2.x recommends).
  - "sequential": google.adk.agents.SequentialAgent of BaseAgents. Deprecated in
                  google-adk 2.12.0 ("deprecated in favor of Workflow"); kept as a
                  fallback and for parity testing, to be removed after migration.

Tool backends (how agents reach their tools, see tool_gateway.py):
  - "local": direct calls into the deterministic code (default; no network).
  - "mcp":   the ClaimSleuth MCP server via ADK's McpToolset, over stdio (the server
             is spawned per run) or streamable HTTP (server already running). If the
             server cannot be reached the run is aborted (MCPUnavailableError); it
             never silently falls back. The VerifierAgent always recomputes locally.

Forensics (optional): forensics="crewai_a2a" adds ADK's RemoteA2aAgent, which calls the
CrewAI forensics service (separate process and environment) over the A2A protocol, followed
by the deterministic ForensicsReviewAgent. CrewAI findings can escalate a claim, never approve it.

Planner modes:
  - deterministic (default, no API key): EvidenceAgent extracts facts.
  - model: an LlmAgent ("InvestigatorLLM") chooses which documents/fields to
    investigate. Used when a model object is passed (tests use a fake BaseLlm) or
    when USE_LLM=true and GOOGLE_API_KEY is set (Gemini). The model never computes
    or authorizes a payout: it has no settlement tool in either tool backend.
"""
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from google.adk import Context, Runner, Workflow
from google.adk.agents import BaseAgent, LlmAgent, RunConfig, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models import BaseLlm
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, FunctionNode
from google.genai import types
from pydantic import PrivateAttr

from configs.settings import GEMINI_MODEL, MAX_MODEL_CALLS, USE_LLM
from src.adk_app import steps
from src.adk_app.forensics import build_forensics_agent, default_a2a_url
from src.core.tracing import CURRENT_TRACE_ID, investigation_span, traceparent_headers
import httpx


async def _add_traceparent(request):
    """httpx request hook: propagate the current W3C trace context to the A2A service."""
    for key, value in traceparent_headers().items():
        request.headers[key] = value
from src.adk_app.patterns import RunOptions, agent_name, build_pattern_workflow, execute_step
from src.adk_app.report import FinalReport
from src.adk_app.tool_gateway import (
    LocalTools, McpConfig, McpTools, MCPUnavailableError, ToolCallError, build_mcp_toolset, unwrap_mcp_result,
)
from src.adk_app.tools import MODEL_TOOLS, instruction_provider, investigator_instruction, record_model_payout_claim
from google.adk.models.llm_response import LlmResponse
from src.core.guardrails import evaluate_claim, redact_toxicity, scan_injection, scan_toxicity
from src.tools.evidence_store import EvidenceStore
from src.core.audit import AuditLogger
from src.core.state import ClaimInvestigationState, EvidenceItem, VerifierReport

Backend = Literal["workflow", "sequential"]
ToolBackend = Literal["local", "mcp"]
APP_NAME = "claimsleuth"



def pipeline_chain(model_agent: Optional[LlmAgent] = None, forensics_agent: Optional[BaseAgent] = None) -> list:
    """
    The ordered pipeline. Items are step functions (deterministic agents) or ADK agents.
    - model mode: the InvestigatorLLM replaces EvidenceAgent;
    - forensics mode: the CrewAI service (RemoteA2aAgent over A2A) and the deterministic
      ForensicsReviewAgent run after AnomalyAgent.
    Every check after evidence gathering stays deterministic.
    """
    chain = [steps.intake_agent, model_agent if model_agent is not None else steps.evidence_agent,
             steps.anomaly_agent]
    if forensics_agent is not None:
        chain += [forensics_agent, steps.forensics_review_agent]
    return chain + [steps.warranty_agent, steps.settlement_agent, steps.verifier_agent, steps.report_agent]
# The only MCP tools the model may use: investigative, no money or eligibility decisions.
MODEL_MCP_TOOLS = ["list_claim_documents", "extract_claim_document_facts", "flag_serial_anomalies"]

# What has actually executed in this process (reported by /api/runtime-status).
ADK_EXECUTION_STATUS: Dict[str, Any] = {"runs_completed": 0, "last_backend": None, "last_planner": None,
                                        "last_tool_backend": None, "last_completed_at": None,
                                        "crewai_a2a_calls_ok": 0, "crewai_a2a_calls_failed": 0}

GatewayFactory = Callable[[Any], Any]


# ─── Backend 1: ADK Workflow -> see patterns.py (sequential / parallel / hierarchical) ───

# ─── Backend 2: ADK SequentialAgent (deprecated in 2.12.0; sequential pattern only) ───

class StepAgent(BaseAgent):
    """Wraps one step function as an ADK agent that emits its state delta as an event."""
    _step: Callable = PrivateAttr()
    _make_gateway: GatewayFactory = PrivateAttr()
    _opts: RunOptions = PrivateAttr()

    def __init__(self, step: Callable, make_gateway: GatewayFactory, opts: RunOptions):
        super().__init__(name=agent_name(step))
        self._step = step
        self._make_gateway = make_gateway
        self._opts = opts

    async def _run_async_impl(self, ctx: InvocationContext):
        gateway = self._make_gateway(Context(ctx))
        delta = await execute_step(self._step, dict(ctx.session.state), gateway, self._opts)
        yield Event(author=self.name, invocation_id=ctx.invocation_id,
                    actions=EventActions(state_delta=delta))


def _build_sequential(chain: list, make_gateway: GatewayFactory, opts: RunOptions) -> SequentialAgent:
    sub_agents = [item if isinstance(item, BaseAgent) else StepAgent(item, make_gateway, opts) for item in chain]
    return SequentialAgent(name="ClaimInvestigationSequence", sub_agents=sub_agents)


# ─── Model agent ─────────────────────────────────────────────────────────

def _mark_model_agent_started(callback_context):
    """ADK before_agent_callback: record the model agent in the agent sequence."""
    sequence = list(callback_context.state.get("bb:agent_sequence", []))
    callback_context.state["bb:agent_sequence"] = sequence + ["InvestigatorLLM"]


def _record_model_call(tool_context, tool: str, args: Dict[str, Any], ok: bool, via: str, error: str = None):
    calls = list(tool_context.state.get("bb:model_tool_calls", []))
    calls.append({"tool": tool, "args": args, "ok": ok, "via": via, **({"error": error} if error else {})})
    tool_context.state["bb:model_tool_calls"] = calls


def _guard_mcp_tool(tool, args, tool_context):
    """
    ADK before_tool_callback (MCP model mode): the model may only investigate the
    claim under review. Returning a dict skips the real call.
    """
    if tool.name not in MODEL_MCP_TOOLS:
        return None
    claim_id = tool_context.state.get("bb:claim", {}).get("claim_id")
    if "claim_id" in args and args["claim_id"] != claim_id:
        _record_model_call(tool_context, tool.name, args, False, "blocked",
                           f"claim_id {args['claim_id']!r} is not the claim under review")
        return {"error": f"Blocked: you may only investigate claim {claim_id}."}
    return None


def _merge_mcp_result(tool, args, tool_context, tool_response):
    """
    ADK after_tool_callback (MCP model mode): record the call and put extracted
    facts on the Blackboard, using the same merge/conflict rule as EvidenceAgent.
    """
    if tool.name not in MODEL_MCP_TOOLS:
        return None
    if isinstance(tool_response, dict) and "error" in tool_response and "isError" not in tool_response:
        return None  # blocked by _guard_mcp_tool, already recorded
    try:
        result = unwrap_mcp_result(tool.name, tool_response)
    except ToolCallError as exc:
        _record_model_call(tool_context, tool.name, args, False, "mcp", str(exc))
        return None
    _record_model_call(tool_context, tool.name, args, True, "mcp")
    if tool.name == "extract_claim_document_facts":
        items = {k: EvidenceItem(**v) for k, v in result.get("facts", {}).items()}
        for key, value in steps.merge_evidence(tool_context.state.to_dict(), items).items():
            tool_context.state[key] = value
    return None


def _guard_model_request(callback_context, llm_request):
    """
    ADK before_model_callback (C-8): inspect the exact text about to be sent to the model.
    Prompt injection -> the call is BLOCKED (a refusal is returned; the model never runs).
    Toxic language   -> redacted in place before the model sees it.
    """
    texts = [p for c in (llm_request.contents or []) for p in (c.parts or []) if p.text]
    config = getattr(llm_request, "config", None)
    system = getattr(config, "system_instruction", None) if config else None
    system_text = system if isinstance(system, str) else " ".join(
        p.text for p in (getattr(system, "parts", None) or []) if getattr(p, "text", None))
    hits = [h for p in texts for h in scan_injection(p.text)] + scan_injection(system_text or "")
    blocks = list(callback_context.state.get("bb:model_guardrail", []))
    if hits:
        blocks.append({"action": "block", "kind": "prompt_injection", "detail": hits[0][:120]})
        callback_context.state["bb:model_guardrail"] = blocks
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(
            text="Request blocked by input guardrail (possible prompt injection). No tools were used.")]))
    for part in texts:
        if scan_toxicity(part.text):
            part.text = redact_toxicity(part.text)
            blocks.append({"action": "redact", "kind": "toxic_language", "detail": "model request redacted"})
    if isinstance(system, str) and scan_toxicity(system):
        config.system_instruction = redact_toxicity(system)
        blocks.append({"action": "redact", "kind": "toxic_language", "detail": "system instruction redacted"})
    if blocks:
        callback_context.state["bb:model_guardrail"] = blocks
    return None


def _record_model_usage(callback_context, llm_response):
    """
    ADK after_model_callback (C-5): record token usage ONLY when the model reports it.
    Test doubles report nothing, so no token or cost figures are ever invented.
    """
    usage = getattr(llm_response, "usage_metadata", None)
    if usage is None:
        return None
    entries = list(callback_context.state.get("bb:llm_usage", []))
    entries.append({"agent": callback_context.agent_name,
                    "input_tokens": getattr(usage, "prompt_token_count", None),
                    "output_tokens": getattr(usage, "candidates_token_count", None)})
    callback_context.state["bb:llm_usage"] = entries
    return None


def _type_check_tool_result(tool, args, tool_context, tool_response):
    """ADK after_tool_callback (C-8): a local tool must return a JSON object; anything else becomes an error."""
    if not isinstance(tool_response, dict):
        return {"error": f"tool {tool.name} returned {type(tool_response).__name__}, expected an object"}
    return None


def _build_model_agent(model: Union[str, BaseLlm], model_toolset=None, prompt_variant: str = "baseline") -> LlmAgent:
    investigator_instruction(prompt_variant)   # validates the variant name early
    common = dict(name="InvestigatorLLM", model=model, output_key="bb:model_summary",
                  before_agent_callback=_mark_model_agent_started,
                  before_model_callback=_guard_model_request, after_model_callback=_record_model_usage)
    if model_toolset is None:
        return LlmAgent(instruction=instruction_provider(prompt_variant), tools=MODEL_TOOLS,
                        after_tool_callback=_type_check_tool_result, **common)
    return LlmAgent(instruction=instruction_provider(prompt_variant, mcp=True),
                    tools=[model_toolset, record_model_payout_claim],
                    before_tool_callback=_guard_mcp_tool, after_tool_callback=_merge_mcp_result, **common)


def resolve_model(model: Optional[Union[str, BaseLlm]], use_llm: Optional[bool]) -> Optional[Union[str, BaseLlm]]:
    """An explicit model wins; otherwise Gemini only when enabled AND a key is configured."""
    if model is not None:
        return model
    enabled = USE_LLM if use_llm is None else use_llm
    if enabled and os.environ.get("GOOGLE_API_KEY"):
        return GEMINI_MODEL
    return None


def planner_label(model: Optional[Union[str, BaseLlm]]) -> str:
    if model is None:
        return "deterministic"
    return f"llm:{model if isinstance(model, str) else model.model}"


# ─── Runner ──────────────────────────────────────────────────────────────

@dataclass
class AdkRunResult:
    report: FinalReport
    state: Dict[str, Any]
    claim_state: ClaimInvestigationState
    events: List[Dict[str, Any]] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)
    model_error: Optional[str] = None
    wall_ms: Optional[float] = None


def claim_state_from_blackboard(state: Dict[str, Any]) -> ClaimInvestigationState:
    """The canonical ClaimInvestigationState for a finished ADK run (used for human adjudication)."""
    claim_state = steps.build_claim_state(state)
    claim_state.verifier_report = VerifierReport(**state["bb:verifier"])
    claim_state.status = state["bb:status"]
    claim_state.requires_human_approval = state["bb:requires_human_approval"]
    claim_state.missing_fields = list(state.get("bb:missing", []))
    return claim_state


def _find_exception(exc: BaseException, cls):
    """Find `cls` in an exception, its causes, or (nested) exception groups."""
    seen = set()
    stack = [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, cls):
            return e
        stack.extend(getattr(e, "exceptions", []) or [])
        stack.extend([e.__cause__, e.__context__])
    return None


async def _run_async(claim: Dict[str, Any], backend: Backend, model: Optional[Union[str, BaseLlm]],
                     run_id: str, planner: str, tools: ToolBackend, mcp: McpConfig,
                     forensics: str = "none", a2a_url: str = "", a2a_timeout: float = 120.0,
                     opts: Optional[RunOptions] = None, prompt_variant: str = "baseline") -> AdkRunResult:
    opts = opts or RunOptions()
    toolsets, closers = [], []
    try:
        if tools == "mcp":
            toolset = build_mcp_toolset(mcp)
            toolsets.append(toolset)
            make_gateway = lambda ctx: McpTools(toolset, ctx, mcp.transport)
            tool_label = f"mcp_{mcp.transport}"
        else:
            make_gateway = lambda ctx: LocalTools()
            tool_label = "local"

        model_agent = None
        if model is not None:
            model_toolset = None
            if tools == "mcp":
                model_toolset = build_mcp_toolset(mcp, tool_filter=MODEL_MCP_TOOLS)
                toolsets.append(model_toolset)
            model_agent = _build_model_agent(model, model_toolset, prompt_variant)

        forensics_agent = None
        if forensics == "crewai_a2a":
            # Every A2A request carries the run's W3C traceparent header
            a2a_http = httpx.AsyncClient(timeout=a2a_timeout, event_hooks={"request": [_add_traceparent]})
            closers.append(a2a_http.aclose)
            forensics_agent = build_forensics_agent(a2a_url, a2a_timeout, httpx_client=a2a_http)
        if backend == "workflow":
            root = build_pattern_workflow(opts, make_gateway, model_agent, forensics_agent)
        else:
            root = _build_sequential(pipeline_chain(model_agent, forensics_agent), make_gateway, opts)
        backend_label = "workflow" if backend == "workflow" else "sequential_agent"

        session_service = InMemorySessionService()
        runner_kwargs = {"node": root} if backend == "workflow" else {"agent": root}
        runner = Runner(app_name=APP_NAME, session_service=session_service, **runner_kwargs)
        session = await session_service.create_session(
            app_name=APP_NAME, user_id="claimsleuth",
            state={"bb:claim": claim, "bb:tool_calls": [], "bb:tool_errors": [], "bb:agent_timings": [],
                   "bb:context_stats": [], "bb:messages": [], "bb:manager_decisions": [],
                   "bb:run": {"run_id": run_id, "trace_id": CURRENT_TRACE_ID.get(),
                              "backend": backend_label, "planner": planner,
                              "tool_backend": tool_label, "t0": time.perf_counter(),
                              "pattern": opts.pattern, "messaging": opts.messaging,
                              "forensics": f"crewai_a2a@{a2a_url}" if forensics == "crewai_a2a" else "none"}})

        # The model only ever sees the guardrail-sanitised claim (toxic words redacted)
        safe_claim = evaluate_claim(claim, EvidenceStore().get_all_claim_documents(
            claim.get("claim_id", ""))).sanitized_claim
        message = types.Content(role="user", parts=[types.Part(text=(
            "Investigate this warranty claim:\n" + json.dumps(safe_claim, indent=2)))])
        events = []
        async for event in runner.run_async(user_id="claimsleuth", session_id=session.id, new_message=message,
                                            run_config=RunConfig(max_llm_calls=MAX_MODEL_CALLS)):
            node_info = getattr(event, "node_info", None)
            events.append({"author": event.author,
                           "node_path": getattr(node_info, "path", None) if node_info else None,
                           "state_keys": sorted((event.actions.state_delta or {}).keys()) if event.actions else []})

        final = await session_service.get_session(app_name=APP_NAME, user_id="claimsleuth", session_id=session.id)
        state = dict(final.state)
        if "bb:final_report" not in state:
            raise RuntimeError("ADK run ended without a FinalReport (pipeline did not complete)")
        return AdkRunResult(report=FinalReport(**state["bb:final_report"]), state=state,
                            claim_state=claim_state_from_blackboard(state),
                            events=events, files=state.get("bb:report_files", {}))
    finally:
        for toolset in toolsets:
            await toolset.close()
        for close in closers:
            await close()


def run_adk_investigation(claim: Dict[str, Any], backend: Backend = "workflow",
                          model: Optional[Union[str, BaseLlm]] = None, use_llm: Optional[bool] = None,
                          fallback_on_model_error: bool = True, tools: ToolBackend = "local",
                          mcp: Optional[McpConfig] = None, forensics: str = "none",
                          a2a_url: Optional[str] = None, a2a_timeout: float = 120.0,
                          pattern: str = "sequential", messaging: str = "blackboard",
                          drop_routes=None, prompt_variant: str = "baseline") -> AdkRunResult:
    """
    Run one claim through the ADK pipeline and return the FinalReport plus trace.
    - If a model run fails (network, quota, bad output) and fallback is allowed, the
      claim is re-run with the deterministic planner and the report says so.
    - If the MCP server is unavailable, MCPUnavailableError is raised: no report and
      no payout are produced (fail closed). There is no silent switch to local tools.
    """
    if backend not in ("workflow", "sequential"):
        raise ValueError(f"Unknown backend '{backend}'")
    if tools not in ("local", "mcp"):
        raise ValueError(f"Unknown tool backend '{tools}'")
    if forensics not in ("none", "crewai_a2a"):
        raise ValueError(f"Unknown forensics mode '{forensics}'")
    a2a_url = (a2a_url or default_a2a_url()).rstrip("/")
    if pattern not in ("sequential", "parallel", "hierarchical"):
        raise ValueError(f"Unknown pattern '{pattern}'")
    if messaging not in ("blackboard", "direct", "broadcast"):
        raise ValueError(f"Unknown messaging mode '{messaging}'")
    if backend == "sequential" and pattern != "sequential":
        raise ValueError("The deprecated SequentialAgent backend supports only pattern='sequential'")
    opts = RunOptions(pattern=pattern, messaging=messaging, drop_routes=set(drop_routes or ()))
    mcp = mcp or McpConfig()
    resolved = resolve_model(model, use_llm)
    run_id = f"adk-{uuid.uuid4().hex[:10]}"
    model_error = None

    def run(model_, run_id_, planner_):
        try:
            if model_ is not None and opts.messaging != "blackboard":
                raise ValueError("Model mode supports messaging='blackboard' only")
            started = time.perf_counter()
            with investigation_span(claim.get("claim_id", "?"), run_id_,
                                    {"pattern": opts.pattern, "messaging": opts.messaging, "tools": tools}):
                result_ = asyncio.run(_run_async(claim, backend, model_, run_id_, planner_, tools, mcp,
                                                 forensics, a2a_url, a2a_timeout, opts, prompt_variant))
            result_.wall_ms = round((time.perf_counter() - started) * 1000, 1)
            return result_
        except BaseException as exc:
            unavailable = _find_exception(exc, MCPUnavailableError)
            if unavailable is not None:
                AuditLogger(claim["claim_id"]).log_conflict(
                    f"[{run_id_}] MCP server unavailable; investigation aborted without a result: {unavailable}")
                raise unavailable from exc
            raise

    try:
        result = run(resolved, run_id, planner_label(resolved))
    except MCPUnavailableError:
        raise
    except Exception as exc:
        if resolved is None or not fallback_on_model_error:
            raise
        model_error = f"{type(exc).__name__}: {exc}"
        planner = f"deterministic (fallback after model error in {planner_label(resolved)})"
        result = run(None, run_id + "-fb", planner)
    result.model_error = model_error

    ADK_EXECUTION_STATUS.update(runs_completed=ADK_EXECUTION_STATUS["runs_completed"] + 1,
                                last_backend=result.report.pipeline_backend,
                                last_planner=result.report.planner,
                                last_tool_backend=result.report.tool_backend,
                                crewai_a2a_calls_ok=ADK_EXECUTION_STATUS["crewai_a2a_calls_ok"]
                                + int(bool(result.report.a2a_call and result.report.a2a_call.get("ok"))),
                                crewai_a2a_calls_failed=ADK_EXECUTION_STATUS["crewai_a2a_calls_failed"]
                                + int(bool(result.report.a2a_call and not result.report.a2a_call.get("ok"))),
                                last_completed_at=datetime.now(timezone.utc).isoformat())
    return result
