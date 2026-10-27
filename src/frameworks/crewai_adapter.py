"""
ClaimSleuth — CrewAI-style placeholder workflow (NOT a CrewAI integration)

STATUS: The crewai package is not imported or executed here and no LLM is called.
This module mimics CrewAI concepts with plain Pydantic classes and a fixed Python
sequence. It does not satisfy the "use CrewAI" criterion; a real Crew exposed over
A2A is planned (WP-4). Every result carries an "implementation" note saying so.

Mimics the CrewAI paradigm:
- Specialist Agents with distinct roles, goals, and backstories.
- Explicit Tasks with descriptions, assigned agents, and expected outputs.
- Sequential Crew Execution coordinating across the Blackboard via structured A2A messages.
"""

from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
import uuid

from src.core.blackboard import Blackboard, BlackboardTask, A2AMessage
from src.core.monitoring import ExecutionMonitor
from src.mcp.tool_server import MCPToolClient


class CrewAIAgent(BaseModel):
    """CrewAI-compatible Agent representation."""
    role: str
    goal: str
    backstory: str
    allow_delegation: bool = True
    verbose: bool = True


class CrewAITask(BaseModel):
    """CrewAI-compatible Task representation."""
    description: str
    expected_output: str
    agent_role: str
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    is_completed: bool = False


class ClaimInvestigationCrew:
    """
    CrewAI-aligned multi-specialist crew for ClaimSleuth.
    Coordinates 4 specialist roles over the Blackboard and MCP tools:
    1. Lead Investigator (Orchestration & Planning)
    2. Evidence Specialist (Document Extraction & Provenance)
    3. Warranty Policy Specialist (Coverage Window & Exclusions)
    4. Settlement & Verifier Specialist (Deterministic Payout & Dual-Gate Check)
    """

    def __init__(
        self,
        case_id: str,
        blackboard: Blackboard,
        monitor: Optional[ExecutionMonitor] = None
    ):
        self.case_id = case_id
        self.blackboard = blackboard
        self.monitor = monitor
        self.mcp_client = MCPToolClient()

        # Define CrewAI Specialist Agents
        self.lead_investigator = CrewAIAgent(
            role="LeadInvestigator",
            goal="Formulate investigation strategy, allocate tasks, and adaptively replan upon conflicts.",
            backstory="Senior forensic warranty auditor ensuring adherence to protocol and deterministic limits."
        )

        self.evidence_specialist = CrewAIAgent(
            role="EvidenceAgent",
            goal="Extract facts with verbatim substring provenance from claim documents.",
            backstory="Document analyst specializing in detecting OCR errors, tampering, and missing fields."
        )

        self.warranty_specialist = CrewAIAgent(
            role="WarrantyAgent",
            goal="Evaluate claim eligibility deterministically against policy coverage windows and exclusions.",
            backstory="Underwriting expert executing strict contractual clauses with zero generative leeway."
        )

        self.verifier_specialist = CrewAIAgent(
            role="VerifierAgent",
            goal="Perform dual-gate verification on evidence provenance and deterministic arithmetic.",
            backstory="Independent safety auditor with absolute authority to block unproven or inaccurate payouts."
        )

    # Placeholder disclosure returned with every result so the API/UI cannot overstate it.
    IMPLEMENTATION_NOTE = (
        "Placeholder: role-labelled sequential Python workflow. The crewai package is not "
        "imported or executed; no LLM is called. To be replaced by a real CrewAI Crew (WP-4)."
    )

    def kickoff(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the role-labelled sequential investigation workflow.
        Agents communicate intermediate findings via Blackboard and A2A-style messages.

        Fail-closed rules (D-1 / D-11):
        - Facts come only from the caller (explicit inputs) or from the claim's documents.
          Nothing is defaulted; missing facts end the run as NEEDS_EVIDENCE.
        - Policy terms come from configs/policies.json (or explicit caller inputs).
        - Conflicting values across documents end the run as CONFLICT_ESCALATED.
        - The payout is recomputed independently and must match before approval is proposed.
        """
        from src.tools.evidence_store import EvidenceStore
        from src.agents.extractor import EvidenceExtractor
        from src.tools.policy_rules import load_policy, check_exclusions
        from src.tools.math_engine import compute_settlement as recompute_settlement

        claim_id = scenario_data.get("claim_id", self.case_id)
        agents_involved = [self.lead_investigator.role, self.evidence_specialist.role]
        facts: Dict[str, Any] = {}
        fact_sources: Dict[str, str] = {}

        def finish(status: str, **extra) -> Dict[str, Any]:
            result = {
                "claim_id": claim_id,
                "status": status,
                "implementation": self.IMPLEMENTATION_NOTE,
                "settlement": None,
                "conflicts": [],
                "missing_fields": [],
                "replan_triggered": False,
                "fact_sources": fact_sources,
                "crew_agents_involved": agents_involved,
            }
            result.update(extra)
            result["blackboard_facts_count"] = len(self.blackboard.state.working_facts)
            result["a2a_messages_count"] = len(self.blackboard.state.messages)
            return result

        # Step 1: Lead Investigator posts the evidence task
        t1 = self.blackboard.post_task(
            goal=f"Extract purchase date, serial number and repair cost for claim {claim_id}",
            assigned_agent="EvidenceAgent"
        )
        self.blackboard.send_message(
            sender=self.lead_investigator.role,
            receiver=self.evidence_specialist.role,
            intent="REQUEST_ACTION",
            payload={"task_id": t1.task_id, "action": "extract_document_facts"}
        )

        # Evidence Specialist: explicit caller inputs first, then documents.
        for key in ("purchase_date", "repair_cost", "serial_number", "incident_cause"):
            if scenario_data.get(key) not in (None, ""):
                facts[key] = scenario_data[key]
                fact_sources[key] = "caller_input"

        conflicts: List[str] = []
        serials_by_doc: Dict[str, str] = {}
        docs = EvidenceStore().get_all_claim_documents(claim_id)
        extractor = EvidenceExtractor()
        for doc_name, doc_text in docs.items():
            # Same document routing as the Lead Investigator: purchase facts from
            # invoices/receipts, repair facts from diagnostic/estimate documents.
            is_invoice = "invoice" in doc_name.lower() or "receipt" in doc_name.lower()
            fields = ["purchase_date", "serial_number"] if is_invoice else ["repair_cost", "incident_cause", "serial_number"]
            extracted = extractor.extract_from_document(doc_name, doc_text, fields, use_llm=False)
            for field, item in extracted.items():
                if item.status != "VERIFIED":
                    continue
                if field == "serial_number":
                    serials_by_doc[doc_name] = str(item.extracted_value)
                value = float(item.extracted_value) if field == "repair_cost" else str(item.extracted_value)
                if field not in facts:
                    facts[field] = value
                    fact_sources[field] = doc_name
                elif str(facts[field]).strip().lower() != str(value).strip().lower():
                    conflicts.append(
                        f"{field} conflict: '{facts[field]}' ({fact_sources[field]}) vs '{value}' ({doc_name})"
                    )

        if len(set(s.upper() for s in serials_by_doc.values())) > 1:
            is_inv = lambda d: "invoice" in d.lower() or "receipt" in d.lower()
            inv = next((s for d, s in serials_by_doc.items() if is_inv(d)), "")
            diag = next((s for d, s in serials_by_doc.items() if not is_inv(d)), "")
            anomaly = self.mcp_client.call_tool(
                "flag_serial_anomalies",
                {"claim_serial": inv, "invoice_serial": inv, "diagnostic_serial": diag}
            )
            for a in anomaly.get("anomalies", []):
                if a not in conflicts:
                    conflicts.append(a)

        for key, value in facts.items():
            self.blackboard.set_fact(key, value, source_agent="EvidenceAgent", provenance_doc=fact_sources.get(key))
        self.blackboard.update_task_status(t1.task_id, status="COMPLETED", result={"facts": facts, "conflicts": conflicts})
        self.blackboard.send_message(
            sender=self.evidence_specialist.role,
            receiver=self.lead_investigator.role,
            intent="DISCREPANCY_FLAGGED" if conflicts else "EVIDENCE_DISCOVERED",
            payload={"facts": facts, "sources": fact_sources, "conflicts": conflicts}
        )

        if conflicts:
            return finish("CONFLICT_ESCALATED", conflicts=conflicts,
                          reason="Conflicting evidence across documents; routed to human review")

        # Step 2: required facts and policy terms are never defaulted
        incident_date = scenario_data.get("incident_date")
        missing = [k for k, v in (("purchase_date", facts.get("purchase_date")),
                                  ("incident_date", incident_date),
                                  ("repair_cost", facts.get("repair_cost"))) if v in (None, "")]

        policy_obj = load_policy(scenario_data["policy_id"]) if scenario_data.get("policy_id") else None
        if policy_obj:
            terms = {
                "duration_months": policy_obj.duration_months,
                "deductible": policy_obj.deductible,
                "max_coverage_limit": policy_obj.max_coverage_limit,
                "depreciation_rate": policy_obj.depreciation_rate,
            }
            exclusions = policy_obj.exclusions
        else:
            terms = {k: scenario_data.get(k) for k in ("duration_months", "deductible", "max_coverage_limit")}
            terms["depreciation_rate"] = float(scenario_data.get("depreciation_rate", 0.0))
            exclusions = scenario_data.get("exclusions", [])
        if terms["duration_months"] is None:
            missing.append("policy.duration_months")

        if missing:
            return finish("NEEDS_EVIDENCE", missing_fields=missing,
                          reason=f"Required facts or policy terms missing: {missing}")

        # Step 3: Warranty Specialist checks the coverage window via MCP
        agents_involved.append(self.warranty_specialist.role)
        t2 = self.blackboard.post_task(
            goal="Verify warranty coverage duration window",
            assigned_agent="WarrantyAgent",
            tool_name="check_coverage_window"
        )
        cov_args = {"purchase_date": str(facts["purchase_date"]), "incident_date": str(incident_date),
                    "duration_months": int(terms["duration_months"])}
        self.blackboard.send_message(
            sender=self.lead_investigator.role,
            receiver=self.warranty_specialist.role,
            intent="REQUEST_ACTION",
            payload=cov_args
        )
        cov_res = self.mcp_client.call_tool("check_coverage_window", cov_args)
        self.blackboard.set_fact("coverage_status", cov_res, source_agent="WarrantyAgent")
        self.blackboard.update_task_status(t2.task_id, status="COMPLETED", result=cov_res)
        self.blackboard.send_message(
            sender=self.warranty_specialist.role,
            receiver=self.lead_investigator.role,
            intent="POLICY_EVALUATED",
            payload=cov_res
        )

        if not cov_res.get("is_valid") or not cov_res.get("is_covered"):
            # Adaptive replanning (Master Doc Section 9.2): halt and deny
            reason = cov_res.get("reason") or cov_res.get("error")
            self.blackboard.state.replan_count += 1
            if self.monitor:
                self.monitor.on_replan(step_num=2, reason=f"Coverage check failed: {reason}")
            self.blackboard.send_message(
                sender=self.lead_investigator.role,
                receiver="ALL",
                intent="REPLAN_TRIGGERED",
                payload={"reason": reason, "action": "HALT_INVESTIGATION_DENY"}
            )
            return finish("REJECTED_DETERMINISTICALLY", reason=reason, replan_triggered=True)

        cause_text = f"{scenario_data.get('reported_issue', '')} {facts.get('incident_cause', '')}"
        is_excluded, matched = check_exclusions(cause_text, exclusions)
        if is_excluded:
            return finish("REJECTED_DETERMINISTICALLY", reason=f"Matches policy exclusion: '{matched}'")

        missing_terms = [k for k in ("deductible", "max_coverage_limit") if terms[k] is None]
        if missing_terms:
            return finish("NEEDS_EVIDENCE", missing_fields=[f"policy.{k}" for k in missing_terms],
                          reason=f"Policy terms missing: {missing_terms}")

        # Step 4: settlement via MCP (depreciation included)
        repair_cost = float(facts["repair_cost"])
        calc_res = self.mcp_client.call_tool(
            "compute_settlement",
            {"repair_cost": repair_cost, "deductible": float(terms["deductible"]),
             "max_limit": float(terms["max_coverage_limit"]), "depreciation": float(terms["depreciation_rate"])}
        )
        self.blackboard.set_fact("settlement_calculation", calc_res, source_agent="WarrantyAgent")

        # Step 5: Verifier recomputes independently instead of trusting the tool result
        agents_involved.append(self.verifier_specialist.role)
        t3 = self.blackboard.post_task(
            goal="Execute independent dual-gate verification",
            assigned_agent="VerifierAgent"
        )
        self.blackboard.send_message(
            sender=self.lead_investigator.role,
            receiver=self.verifier_specialist.role,
            intent="REQUEST_ACTION",
            payload={"claim_id": claim_id, "settlement": calc_res}
        )
        expected = recompute_settlement(repair_cost, float(terms["deductible"]),
                                        float(terms["max_coverage_limit"]), float(terms["depreciation_rate"]))
        gate_1_ok = repair_cost > 0 and not conflicts
        gate_2_math_ok = abs(float(calc_res.get("approved_payout", -1)) - expected.approved_payout) < 0.01
        is_approved = gate_1_ok and gate_2_math_ok

        verdict_data = {
            "is_approved_by_verifier": is_approved,
            "gate_1_evidence_substantiated": gate_1_ok,
            "gate_2_deterministic_rules_passed": True,
            "gate_2_arithmetic_matches": gate_2_math_ok,
            "approved_payout": calc_res.get("approved_payout") if is_approved else None
        }
        self.blackboard.set_fact("verifier_report", verdict_data, source_agent="VerifierAgent")
        self.blackboard.update_task_status(t3.task_id, status="COMPLETED", result=verdict_data)
        self.blackboard.send_message(
            sender=self.verifier_specialist.role,
            receiver=self.lead_investigator.role,
            intent="VERDICT_ISSUED",
            payload=verdict_data
        )

        if not is_approved:
            return finish("NEEDS_EVIDENCE", verifier_report=verdict_data,
                          reason="Verifier could not substantiate evidence or reproduce the payout")
        return finish("VERIFIED_PENDING_APPROVAL", settlement=calc_res, verifier_report=verdict_data)
