import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from src.core.state import (
    ClaimInvestigationState,
    InvestigationStep,
    SettlementCalc,
    PolicyTerms,
    AgentAction,
    EvidenceItem
)
from src.core.audit import AuditLogger
from src.core.tool_registry import ToolRegistry, build_default_registry
from src.tools.policy_rules import load_policy, calc_coverage_window, check_exclusions
from src.tools.math_engine import compute_settlement
from src.tools.anomaly_detector import flag_serial_anomalies
from src.tools.evidence_store import EvidenceStore
from src.agents.extractor import EvidenceExtractor
from src.verifier.independent_verifier import DualGateVerifier
from configs.settings import (
    MAX_INVESTIGATION_STEPS,
    MAX_MODEL_CALLS,
    HIGH_VALUE_THRESHOLD,
    USE_LLM
)
from src.core.blackboard import Blackboard
from src.core.monitoring import ExecutionMonitor
from src.mcp.tool_server import MCPToolClient

logger = logging.getLogger(__name__)


class LeadInvestigatorAgent:
    """
    Lead Investigator orchestrating the agentic loop:
    PLAN -> EXECUTE -> OBSERVE -> ASSESS -> REPLAN or STOP.
    
    Features:
    - Genuine Gemini-powered adaptive tool selection (when enabled)
    - Deterministic rule and mathematical safety gates
    - Restricted ToolRegistry with Pydantic input validation
    - Provenance tracking with verbatim substring verification
    - Bounded execution (enforced MAX_INVESTIGATION_STEPS & MAX_MODEL_CALLS)
    - DualGate independent verification
    - Blackboard shared state with structured A2A messaging
    - MCP Tool Server gateway integration
    - Execution callback monitoring and error logging
    """

    def __init__(
        self,
        evidence_store: Optional[EvidenceStore] = None,
        llm_client: Optional[Any] = None,
        tool_registry: Optional[ToolRegistry] = None,
        use_llm: Optional[bool] = None
    ):
        self.evidence_store = evidence_store or EvidenceStore()

        if use_llm is None:
            use_llm = USE_LLM

        if llm_client is not None:
            self.llm_client = llm_client
        elif use_llm:
            from src.core.llm_client import try_create_client
            self.llm_client = try_create_client()
        else:
            self.llm_client = None

        self.extractor = EvidenceExtractor(llm_client=self.llm_client)
        self.verifier = DualGateVerifier()
        self.tool_registry = tool_registry or build_default_registry(
            evidence_store=self.evidence_store,
            extractor=self.extractor
        )
        self.mcp_client = MCPToolClient()
        self.active_blackboards: Dict[str, Blackboard] = {}
        self.active_monitors: Dict[str, ExecutionMonitor] = {}

    def run_investigation(self, state: ClaimInvestigationState) -> ClaimInvestigationState:
        audit = AuditLogger(state.claim_id)
        audit.log_step(0, "PLAN", f"Initializing claim investigation for {state.claim_id} under policy {state.policy_id}")

        bb = Blackboard(state.claim_id)
        monitor = ExecutionMonitor(state.claim_id, audit)
        self.active_blackboards[state.claim_id] = bb
        self.active_monitors[state.claim_id] = monitor

        # Populate initial case facts in Blackboard working memory
        bb.set_fact("claim_id", state.claim_id, "LeadInvestigator")
        bb.set_fact("policy_id", state.policy_id, "LeadInvestigator")
        bb.set_fact("incident_date", state.incident_date, "LeadInvestigator")
        bb.set_fact("device_model", state.device_model, "LeadInvestigator")
        bb.set_fact("reported_issue", state.reported_issue, "LeadInvestigator")

        policy: Optional[PolicyTerms] = load_policy(state.policy_id)
        if not policy:
            state.status = "REJECTED_DETERMINISTICALLY"
            audit.log_conflict(f"Unknown policy ID: {state.policy_id}")
            bb.set_fact("policy_error", f"Unknown policy ID: {state.policy_id}", "WarrantyAgent")
            return state

        # Bounded agent loop
        while state.current_step < MAX_INVESTIGATION_STEPS and state.llm_calls_used < MAX_MODEL_CALLS:
            state.current_step += 1
            step_idx = state.current_step

            # ----------------------------------------------------
            # 1. PLAN / REPLAN: Determine next atomic action
            # ----------------------------------------------------
            action = self._plan_next_action(state, policy)
            phase = "REPLAN" if action.replan_reason else "PLAN"

            # Determine specialist agent assignment
            if action.tool_name == "extract_document_facts":
                assigned_agent = "EvidenceAgent"
            elif action.tool_name in ("check_coverage_window", "compute_settlement", "check_exclusions"):
                assigned_agent = "WarrantyAgent"
            elif action.tool_name == "flag_serial_anomalies":
                assigned_agent = "AnomalyAgent"
            else:
                assigned_agent = "LeadInvestigator"

            if action.replan_reason:
                monitor.on_replan(step_idx, action.replan_reason)
                bb.state.replan_count += 1
                bb.send_message(
                    sender="LeadInvestigator",
                    receiver="ALL",
                    intent="REPLAN_TRIGGERED",
                    payload={"step": step_idx, "reason": action.replan_reason}
                )

            # Record step and task in Blackboard
            bb_task = bb.post_task(
                goal=action.reasoning or f"Select tool {action.tool_name}",
                assigned_agent=assigned_agent,
                tool_name=action.tool_name,
                tool_args=action.tool_args
            )

            # Send A2A task request
            bb.send_message(
                sender="LeadInvestigator",
                receiver=assigned_agent,
                intent="REQUEST_ACTION",
                payload={"task_id": bb_task.task_id, "tool": action.tool_name, "args": action.tool_args}
            )

            monitor.on_step_start(step_idx, action.tool_name, action.reasoning or "")

            step = InvestigationStep(
                step_num=step_idx,
                phase=phase,
                description=action.reasoning or f"Select tool {action.tool_name}",
                tool_called=action.tool_name,
                tool_input=action.tool_args,
                reasoning=action.reasoning,
                replan_reason=action.replan_reason
            )
            audit.log_step(step_idx, phase, step.description, action.tool_name)

            # Check if planning decided we should STOP
            if action.tool_name == "STOP":
                step.phase = "STOP"
                state.steps_history.append(step)
                bb.update_task_status(bb_task.task_id, "COMPLETED", result={"stopped": True})
                break

            # ----------------------------------------------------
            # 2. EXECUTE: Call deterministic tool or extractor
            # ----------------------------------------------------
            observation = self._execute_tool(action.tool_name, action.tool_args, state, policy)
            step.tool_output = observation
            audit.log_tool_execution(action.tool_name, action.tool_args, observation)

            monitor.on_tool_executed(
                tool_name=action.tool_name,
                inputs=action.tool_args,
                output=observation,
                is_success=not bool(observation.get("error"))
            )
            bb.update_task_status(bb_task.task_id, "COMPLETED", result=observation)

            # Post intermediate findings and A2A notification
            if action.tool_name == "extract_document_facts":
                extracted = observation.get("extracted", {})
                bb.send_message(
                    sender="EvidenceAgent",
                    receiver="LeadInvestigator",
                    intent="EVIDENCE_DISCOVERED",
                    payload={"extracted_fields": list(extracted.keys())}
                )
                for f_name, item in extracted.items():
                    val = item.get("extracted_value") if isinstance(item, dict) else getattr(item, "extracted_value", item)
                    doc = item.get("source_document") if isinstance(item, dict) else getattr(item, "source_document", None)
                    bb.set_fact(f_name, val, source_agent="EvidenceAgent", provenance_doc=doc)

            elif action.tool_name in ("check_coverage_window", "compute_settlement"):
                bb.send_message(
                    sender="WarrantyAgent",
                    receiver="LeadInvestigator",
                    intent="POLICY_EVALUATED",
                    payload=observation
                )
                bb.set_fact(f"{action.tool_name}_result", observation, source_agent="WarrantyAgent")

            elif action.tool_name == "flag_serial_anomalies":
                intent = "DISCREPANCY_FLAGGED" if observation.get("has_anomalies") else "POLICY_EVALUATED"
                bb.send_message(
                    sender="AnomalyAgent",
                    receiver="LeadInvestigator",
                    intent=intent,
                    payload=observation
                )
                bb.set_fact("serial_anomalies", observation, source_agent="AnomalyAgent")

            # ----------------------------------------------------
            # 3. OBSERVE: Record raw observation & detect conflicts
            # ----------------------------------------------------
            self._record_observation(action.tool_name, observation, state, audit)

            # ----------------------------------------------------
            # 4. ASSESS: Analyze findings, check conflicts/coverage
            # ----------------------------------------------------
            should_stop, assess_notes = self._assess_state(state, policy, audit)
            step.reasoning = f"{step.reasoning} | {assess_notes}" if step.reasoning else assess_notes
            state.steps_history.append(step)
            audit.log_step(step_idx, "ASSESS", assess_notes)

            if should_stop:
                break

        # --------------------------------------------------------
        # 5. DUAL-GATE VERIFICATION (Unconditional independent check)
        # --------------------------------------------------------
        bb.send_message(
            sender="LeadInvestigator",
            receiver="VerifierAgent",
            intent="REQUEST_ACTION",
            payload={"action": "DUAL_GATE_VERIFICATION", "claim_id": state.claim_id}
        )

        verifier_report = self.verifier.verify_investigation(state, policy)
        state.verifier_report = verifier_report
        audit.log_verifier_result(
            verifier_report.is_approved_by_verifier,
            verifier_report.discrepancies
        )

        bb.send_message(
            sender="VerifierAgent",
            receiver="LeadInvestigator",
            intent="VERDICT_ISSUED",
            payload=verifier_report.model_dump()
        )
        bb.set_fact("verifier_verdict", verifier_report.model_dump(), source_agent="VerifierAgent")

        # Adjudication status mapping
        if state.conflicts:
            state.status = "CONFLICT_ESCALATED"
            state.requires_human_approval = True
        elif not verifier_report.is_approved_by_verifier:
            if any("expired" in d.lower() for d in verifier_report.discrepancies):
                state.status = "REJECTED_DETERMINISTICALLY"
            elif any("missing" in d.lower() for d in verifier_report.discrepancies):
                state.status = "NEEDS_EVIDENCE"
            else:
                state.status = "REJECTED_DETERMINISTICALLY"
            state.requires_human_approval = True
        else:
            state.status = "VERIFIED_PENDING_APPROVAL"
            payout = state.deterministic_settlement.approved_payout if state.deterministic_settlement else 0.0
            state.requires_human_approval = (payout >= HIGH_VALUE_THRESHOLD)

        return state

    def _plan_next_action(self, state: ClaimInvestigationState, policy: PolicyTerms) -> AgentAction:
        """
        Determines the next action using Gemini inference if available,
        falling back gracefully to deterministic adaptive planning.
        """
        if self.llm_client is not None and state.llm_calls_used < MAX_MODEL_CALLS:
            try:
                action = self._gemini_plan_action(state, policy)
                if action and (self.tool_registry.is_registered(action.tool_name) or action.tool_name == "STOP"):
                    state.llm_calls_used += 1
                    return action
                logger.warning(f"Gemini proposed unregistered tool '{getattr(action, 'tool_name', None)}', falling back.")
            except Exception as e:
                logger.warning(f"Gemini planning failed, falling back to deterministic planner: {e}")

        return self._fallback_plan_next_action(state, policy)

    def _gemini_plan_action(self, state: ClaimInvestigationState, policy: PolicyTerms) -> AgentAction:
        """Builds structured prompt and queries Gemini for next tool selection."""
        prompt = self._build_planner_prompt(state, policy)
        action: AgentAction = self.llm_client.generate_structured(
            prompt=prompt,
            response_schema=AgentAction,
            temperature=0.1
        )
        return action

    def _build_planner_prompt(self, state: ClaimInvestigationState, policy: PolicyTerms) -> str:
        """Constructs prompt for LLM tool selection."""
        docs = self.evidence_store.list_documents(state.claim_id)
        evidence_summary = {
            k: {
                "value": v.extracted_value,
                "source": v.source_document,
                "confidence": v.confidence,
                "status": v.status
            }
            for k, v in state.evidence.items()
        }
        past_steps = [
            f"Step {s.step_num} ({s.phase}): tool={s.tool_called}, result={s.description}"
            for s in state.steps_history
        ]
        tool_descriptions = self.tool_registry.get_tool_descriptions()

        prompt = f"""You are the Lead Investigator for ClaimSleuth, an agentic warranty claim investigation system.
Your job is to investigate a warranty claim by choosing the NEXT atomic tool to execute from the allowed tools below, or STOP if the investigation is complete or cannot proceed.

CLAIM DETAILS:
- Claim ID: {state.claim_id}
- Policy ID: {state.policy_id}
- Device Model: {state.device_model}
- Incident Date: {state.incident_date}
- Reported Issue: {state.reported_issue}

AVAILABLE DOSSIER DOCUMENTS:
{json.dumps(docs, indent=2)}

CURRENT EVIDENCE COLLECTED:
{json.dumps(evidence_summary, indent=2)}

DETECTED CONFLICTS:
{json.dumps(state.conflicts, indent=2)}

SETTLEMENT STATUS:
{"Settlement already calculated: $" + str(state.deterministic_settlement.approved_payout) if state.deterministic_settlement else "None"}

POLICY PARAMETERS:
- Duration: {policy.duration_months} months
- Deductible: ${policy.deductible}
- Max Coverage Limit: ${policy.max_coverage_limit}
- Depreciation Rate: {policy.depreciation_rate}
- Exclusions: {policy.exclusions}

STEPS TAKEN SO FAR:
{json.dumps(past_steps, indent=2)}

AVAILABLE TOOLS (You may ONLY choose from these):
{tool_descriptions}
Tool: STOP
  Description: Stop the investigation loop when sufficient evidence is gathered and settlement computed, or when a fatal conflict or expiration halts progress.
  Parameters: (none)

CRITICAL RULES:
1. Pure deterministic tools: Do NOT calculate payouts or date coverage yourself. Always call 'check_coverage_window', 'check_exclusions', and 'compute_settlement'.
2. If evidence is missing: call 'extract_document_facts' with target fields on appropriate documents.
3. If documents or serial numbers contradict each other, conflict detection flags them. Once conflict is found, call STOP.
4. When purchase_date and repair_cost are present and coverage is valid, call 'compute_settlement'.
5. Once settlement is computed, call STOP.
6. If a previous extraction attempt failed to find expected fields, specify 'replan_reason' explaining what failed and why you are choosing an alternative.
7. Return a JSON object with: tool_name, tool_args, reasoning, replan_reason (optional).
"""
        return prompt

    def _fallback_plan_next_action(self, state: ClaimInvestigationState, policy: PolicyTerms) -> AgentAction:
        """
        Deterministic, state-aware adaptive planner with replanning capabilities.
        Ensures 100% test reliability and offline support.
        """
        docs = self.evidence_store.get_all_claim_documents(state.claim_id)
        if not docs:
            state.missing_fields.append("all_documents")
            return AgentAction(
                tool_name="STOP",
                tool_args={},
                reasoning="No documents found in dossier"
            )

        # Identify documents previously inspected
        inspected_docs = {
            step.tool_input.get("document_name")
            for step in state.steps_history
            if step.tool_input and "document_name" in step.tool_input
        }

        # Step A: Missing purchase date or serial number
        if "purchase_date" not in state.evidence or "serial_number" not in state.evidence:
            inv_name = next(
                (name for name in docs if "invoice" in name.lower() or "receipt" in name.lower()),
                next(iter(docs.keys()))
            )
            # Check if replanning from a prior failed document
            replan = None
            if inv_name in inspected_docs and "purchase_date" not in state.evidence:
                # Try an alternate doc
                alt_docs = [d for d in docs if d not in inspected_docs]
                if alt_docs:
                    inv_name = alt_docs[0]
                    replan = f"Purchase date missing after inspecting earlier document; replanning with alternative: {inv_name}"

            return AgentAction(
                tool_name="extract_document_facts",
                tool_args={
                    "claim_id": state.claim_id,
                    "document_name": inv_name,
                    "target_fields": ["purchase_date", "serial_number"]
                },
                reasoning=f"Extract purchase proof and serial number from {inv_name}",
                replan_reason=replan
            )

        # Step B: Missing repair cost or damage diagnostics
        if "repair_cost" not in state.evidence:
            diag_name = next(
                (name for name in docs if "diag" in name.lower() or "repair" in name.lower() or "estimate" in name.lower()),
                list(docs.keys())[-1]
            )
            replan = None
            if diag_name in inspected_docs and "repair_cost" not in state.evidence:
                alt_docs = [d for d in docs if d not in inspected_docs]
                if alt_docs:
                    diag_name = alt_docs[0]
                    replan = f"Repair cost missing from {diag_name}; replanning with alternative document: {alt_docs[0]}"

            return AgentAction(
                tool_name="extract_document_facts",
                tool_args={
                    "claim_id": state.claim_id,
                    "document_name": diag_name,
                    "target_fields": ["repair_cost", "incident_cause", "serial_number"]
                },
                reasoning=f"Extract repair estimate and damage cause from {diag_name}",
                replan_reason=replan
            )

        # Step C: Compute settlement if all required fields are present and coverage holds
        if not state.deterministic_settlement:
            cost = float(state.evidence["repair_cost"].extracted_value)
            return AgentAction(
                tool_name="compute_settlement",
                tool_args={
                    "repair_cost": cost,
                    "deductible": policy.deductible,
                    "max_coverage_limit": policy.max_coverage_limit,
                    "depreciation_rate": policy.depreciation_rate
                },
                reasoning="Calculate deterministic settlement payout using validated policy terms"
            )

        return AgentAction(
            tool_name="STOP",
            tool_args={},
            reasoning="All required evidence collected and settlement calculated"
        )

    def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        state: ClaimInvestigationState,
        policy: PolicyTerms
    ) -> Dict[str, Any]:
        """Dispatches tool execution through the restricted ToolRegistry."""
        args_copy = dict(args)

        # Context enrichment for specific tools
        if tool_name == "extract_document_facts":
            if not args_copy.get("claim_id"):
                args_copy["claim_id"] = state.claim_id

        elif tool_name == "compute_settlement":
            if "repair_cost" not in args_copy and "repair_cost" in state.evidence:
                args_copy["repair_cost"] = float(state.evidence["repair_cost"].extracted_value)
            if "deductible" not in args_copy:
                args_copy["deductible"] = policy.deductible
            if "max_coverage_limit" not in args_copy:
                args_copy["max_coverage_limit"] = policy.max_coverage_limit
            if "depreciation_rate" not in args_copy:
                args_copy["depreciation_rate"] = policy.depreciation_rate

        elif tool_name == "check_coverage_window":
            if "purchase_date" not in args_copy and "purchase_date" in state.evidence:
                args_copy["purchase_date"] = str(state.evidence["purchase_date"].extracted_value)
            if "incident_date" not in args_copy:
                args_copy["incident_date"] = state.incident_date
            if "duration_months" not in args_copy:
                args_copy["duration_months"] = policy.duration_months

        # Attempt MCP protocol execution for deterministic tools
        if tool_name == "compute_settlement":
            try:
                mcp_res = self.mcp_client.call_tool(
                    "compute_settlement",
                    {
                        "repair_cost": float(args_copy.get("repair_cost", 0.0)),
                        "deductible": float(args_copy.get("deductible", policy.deductible)),
                        "max_limit": float(args_copy.get("max_coverage_limit", policy.max_coverage_limit)),
                        "depreciation": float(args_copy.get("depreciation_rate", policy.depreciation_rate))
                    }
                )
                if mcp_res and "approved_payout" in mcp_res:
                    return mcp_res
            except Exception as mcp_err:
                logger.warning(f"MCP compute_settlement fallback: {mcp_err}")

        elif tool_name == "check_coverage_window":
            try:
                mcp_res = self.mcp_client.call_tool(
                    "check_coverage_window",
                    {
                        "purchase_date": str(args_copy.get("purchase_date", "")),
                        "incident_date": str(args_copy.get("incident_date", state.incident_date)),
                        "duration_months": int(args_copy.get("duration_months", policy.duration_months))
                    }
                )
                if mcp_res and "is_valid" in mcp_res:
                    return mcp_res
            except Exception as mcp_err:
                logger.warning(f"MCP check_coverage_window fallback: {mcp_err}")

        elif tool_name == "flag_serial_anomalies":
            try:
                mcp_res = self.mcp_client.call_tool(
                    "flag_serial_anomalies",
                    {
                        "claim_serial": str(args_copy.get("claim_serial", "")),
                        "invoice_serial": str(args_copy.get("invoice_serial", "")),
                        "diagnostic_serial": str(args_copy.get("diagnostic_serial", ""))
                    }
                )
                if mcp_res and "has_anomalies" in mcp_res:
                    return mcp_res
            except Exception as mcp_err:
                logger.warning(f"MCP flag_serial_anomalies fallback: {mcp_err}")

        # Dispatch via registry
        if self.tool_registry.is_registered(tool_name):
            try:
                res = self.tool_registry.validate_and_execute(tool_name, args_copy)
                return res
            except Exception as e:
                logger.error(f"Error executing tool '{tool_name}': {e}")
                return {"error": str(e)}

        # Legacy tool aliases for backwards compatibility
        if tool_name == "extract_invoice":
            docs = args.get("docs") or self.evidence_store.get_all_claim_documents(state.claim_id)
            inv_name = next((n for n in docs if "invoice" in n.lower() or "receipt" in n.lower()), next(iter(docs.keys()), ""))
            inv_text = docs.get(inv_name, "")
            extracted = self.extractor.extract_from_document(
                inv_name, inv_text, ["purchase_date", "serial_number"], use_llm=bool(self.llm_client)
            )
            return {"extracted": extracted, "doc_name": inv_name}

        elif tool_name == "extract_diagnostics":
            docs = args.get("docs") or self.evidence_store.get_all_claim_documents(state.claim_id)
            diag_name = next((n for n in docs if "diag" in n.lower() or "repair" in n.lower() or "estimate" in n.lower()), list(docs.keys())[-1] if docs else "")
            diag_text = docs.get(diag_name, "")
            extracted = self.extractor.extract_from_document(
                diag_name, diag_text, ["repair_cost", "incident_cause", "serial_number"], use_llm=bool(self.llm_client)
            )
            return {"extracted": extracted, "doc_name": diag_name}

        return {"error": f"Unknown tool '{tool_name}'"}

    def _record_observation(
        self,
        tool_name: str,
        observation: Dict[str, Any],
        state: ClaimInvestigationState,
        audit: AuditLogger
    ):
        """Records observation into state and runs generalized conflict/anomaly detection."""
        if "extracted" in observation:
            for field, ev_item in observation["extracted"].items():
                if field in state.evidence:
                    existing = state.evidence[field].extracted_value
                    new_val = ev_item.extracted_value
                    if existing and new_val and str(existing).strip().lower() != str(new_val).strip().lower():
                        conflict_desc = (
                            f"{field} conflict: '{existing}' "
                            f"({state.evidence[field].source_document}) vs "
                            f"'{new_val}' ({ev_item.source_document})"
                        )
                        if conflict_desc not in state.conflicts:
                            state.conflicts.append(conflict_desc)
                            audit.log_conflict(conflict_desc)
                else:
                    state.evidence[field] = ev_item
                    audit.log_evidence_extracted(
                        ev_item.evidence_id, field, ev_item.extracted_value,
                        ev_item.confidence, ev_item.source_document
                    )

            # Anomaly detector: cross-check serial numbers when available from multiple docs
            claim_serial = ""
            if "serial_number" in state.evidence:
                claim_serial = str(state.evidence["serial_number"].extracted_value)

            inv_serial = ""
            diag_serial = ""
            if "serial_number" in observation.get("extracted", {}):
                new_serial = str(observation["extracted"]["serial_number"].extracted_value)
                new_source = observation["extracted"]["serial_number"].source_document
                if "invoice" in new_source.lower() or "receipt" in new_source.lower():
                    inv_serial = new_serial
                else:
                    diag_serial = new_serial

            if claim_serial and (inv_serial or diag_serial):
                anomaly_result = flag_serial_anomalies(
                    claim_serial=claim_serial,
                    invoice_serial=inv_serial or claim_serial,
                    diagnostic_serial=diag_serial or claim_serial
                )
                if anomaly_result.get("has_anomalies"):
                    for anom in anomaly_result.get("anomalies", []):
                        if anom not in state.conflicts:
                            state.conflicts.append(anom)
                            audit.log_conflict(anom)

        if "settlement" in observation:
            state.deterministic_settlement = observation["settlement"]
        elif isinstance(observation, SettlementCalc):
            state.deterministic_settlement = observation

        if "anomalies" in observation and isinstance(observation["anomalies"], dict):
            if observation["anomalies"].get("has_anomalies"):
                for anom in observation["anomalies"].get("anomalies", []):
                    if anom not in state.conflicts:
                        state.conflicts.append(anom)
                        audit.log_conflict(anom)

    def _assess_state(
        self,
        state: ClaimInvestigationState,
        policy: PolicyTerms,
        audit: AuditLogger
    ) -> Tuple[bool, str]:
        """Assesses investigation progress and determines whether to halt or continue."""
        # 1. Conflict escalation halts automated loop immediately
        if state.conflicts:
            return True, f"Conflict detected; halting automated loop for human review: {state.conflicts[-1]}"

        # 2. Check coverage dates deterministically
        if "purchase_date" in state.evidence:
            p_date = str(state.evidence["purchase_date"].extracted_value)
            cov = calc_coverage_window(p_date, state.incident_date, policy.duration_months)
            if cov.get("is_valid") and not cov.get("is_covered"):
                return True, f"Policy expired on {cov.get('expiration_date')}. Halting loop as claim is ineligible."

        # 3. Check for policy exclusions if incident cause has been extracted
        if "incident_cause" in state.evidence and policy.exclusions:
            cause = str(state.evidence["incident_cause"].extracted_value)
            is_excl, matched = check_exclusions(cause, policy.exclusions)
            if is_excl:
                return True, f"Reported cause '{cause}' matches policy exclusion: {matched}. Halting loop."

        # 4. Ready if all required items present and settlement calculated
        if "purchase_date" in state.evidence and "repair_cost" in state.evidence and state.deterministic_settlement:
            return True, "All required evidence collected and settlement computed. Proceeding to verifier."

        return False, "Continuing to gather required claim facts."

    def get_last_blackboard_summary(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve serialized Blackboard state for a given claim."""
        bb = self.active_blackboards.get(claim_id)
        return bb.get_summary() if bb else None

    def get_last_monitoring_summary(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve execution monitoring metrics for a given claim."""
        monitor = self.active_monitors.get(claim_id)
        return monitor.get_summary_metrics() if monitor else None

