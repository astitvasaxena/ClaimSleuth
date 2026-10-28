"""
ClaimSleuth ADK pipeline steps ("agents").

Each step is an async function:  step(state: dict, tools) -> dict  (the state delta),
where `tools` is LocalTools or McpTools (see tool_gateway.py).
It reads the shared Blackboard (ADK session state, keys prefixed "bb:") and
calls the EXISTING deterministic code, which stays the source of truth for
extraction, eligibility, money and verification. The pipeline backends in
pipeline.py wrap these functions as ADK Workflow nodes or ADK BaseAgents.

Blackboard keys
  bb:run                 run_id, backend, planner (set before the run)
  bb:claim               claim.json fields
  bb:policy              policy terms, or None if the policy is unknown
  bb:documents           dossier document names
  bb:evidence            {field: EvidenceItem dict}
  bb:conflicts           list of unresolved conflicts
  bb:missing             required evidence fields not found
  bb:anomalies           serial-number anomaly check result
  bb:eligibility         coverage window + exclusion result
  bb:settlement_proposal deterministic SettlementCalc dict, or None
  bb:model_payout_claim  payout a model stated (never authoritative), or None
  bb:model_tool_calls    tools the model called (model mode only)
  bb:tool_calls          every tool call made by the deterministic agents (local or MCP)
  bb:tool_errors         tool failures; any entry makes the VerifierAgent fail closed
  bb:forensics_raw / bb:forensics_error / bb:a2a_call   CrewAI reply over A2A (forensics mode)
  bb:forensics           validated ForensicsReport from CrewAI, or None
  bb:rag                 RAG: retrieved passages with source refs, policy citations, stated terms
  bb:verifier            VerifierReport dict
  bb:status / bb:requires_human_approval / bb:payout
  bb:final_report / bb:report_files
  bb:agent_sequence      names of the agents in the order they ran
"""
from typing import Any, Dict, List

from configs.settings import REPORTS_DIR
from src.agents.extractor import EvidenceExtractor
from src.core.adjudication import determine_status, payout_view
from src.core.audit import AuditLogger
from src.core.state import ClaimInvestigationState, EvidenceItem, SettlementCalc, VerifierReport
from src.tools.evidence_store import EvidenceStore
from src.tools.policy_rules import check_exclusions, load_policy
from src.verifier.independent_verifier import DualGateVerifier
from src.adk_app.tool_gateway import ToolCallError
from src.core.forensics_schema import FORENSICS_JSON_PREFIX, ForensicsReport
from src.core.guardrails import evaluate_claim
from src.rag.policy_evidence import check_policy_citations, gather_policy_evidence
import asyncio

REQUIRED_FIELDS = ["purchase_date", "serial_number", "repair_cost"]
INVOICE_FIELDS = ["purchase_date", "serial_number"]
DIAGNOSTIC_FIELDS = ["repair_cost", "incident_cause", "serial_number"]


def is_invoice(document_name: str) -> bool:
    name = document_name.lower()
    return "invoice" in name or "receipt" in name


def _sequence(s: Dict[str, Any], agent: str) -> List[str]:
    return list(s.get("bb:agent_sequence", [])) + [agent]


def _audit(s: Dict[str, Any], agent: str, description: str, tool: str = None) -> AuditLogger:
    audit = AuditLogger(s["bb:claim"]["claim_id"])
    step_num = len(s.get("bb:agent_sequence", [])) + 1
    audit.log_step(step_num, "ADK", f"[{s['bb:run']['run_id']}] {agent}: {description}", tool)
    return audit


# ─── Evidence helpers (shared by EvidenceAgent and the model's extraction tool) ───

def extract_document(claim_id: str, document_name: str, fields: List[str]) -> Dict[str, EvidenceItem]:
    """Deterministic regex extraction with snippet-provenance check (no LLM)."""
    text = EvidenceStore().read_document(claim_id, document_name) or ""
    return EvidenceExtractor(llm_client=None).extract_from_document(document_name, text, fields, use_llm=False)


def merge_evidence(s: Dict[str, Any], extracted: Dict[str, EvidenceItem]) -> Dict[str, Any]:
    """
    Add newly extracted items to bb:evidence. A second, different value for a field
    already held is recorded as a conflict (same rule as the LeadInvestigator loop).
    """
    evidence = dict(s.get("bb:evidence", {}))
    conflicts = list(s.get("bb:conflicts", []))
    audit = AuditLogger(s["bb:claim"]["claim_id"])
    for field, item in extracted.items():
        if field in evidence:
            existing = evidence[field]
            if str(existing["extracted_value"]).strip().lower() != str(item.extracted_value).strip().lower():
                desc = (f"{field} conflict: '{existing['extracted_value']}' ({existing['source_document']}) "
                        f"vs '{item.extracted_value}' ({item.source_document})")
                if desc not in conflicts:
                    conflicts.append(desc)
                    audit.log_conflict(desc)
        else:
            evidence[field] = item.model_dump()
            audit.log_evidence_extracted(item.evidence_id, field, item.extracted_value,
                                         item.confidence, item.source_document)
    return {"bb:evidence": evidence, "bb:conflicts": conflicts}


def serial_by_source(s: Dict[str, Any]) -> Dict[str, str]:
    """Serial numbers per document, read locally (used by the model's local tool)."""
    claim_id = s["bb:claim"]["claim_id"]
    serials = {}
    for doc in s.get("bb:documents", []):
        item = extract_document(claim_id, doc, ["serial_number"]).get("serial_number")
        if item:
            serials[doc] = str(item.extracted_value)
    return serials


def _tool_error(s: Dict[str, Any], agent: str, exc: Exception) -> List[str]:
    """Record a tool failure; the VerifierAgent turns any recorded failure into a fail-closed outcome."""
    AuditLogger(s["bb:claim"]["claim_id"]).log_conflict(f"Tool error in {agent}: {exc}")
    return list(s.get("bb:tool_errors", [])) + [f"{agent}: {exc}"]


# ─── Agents ─────────────────────────────────────────────────────────────
# Each agent: async (state, tools) -> state delta. `tools` is LocalTools or McpTools.

async def intake_agent(s: Dict[str, Any], tools) -> Dict[str, Any]:
    """Loads the policy terms and the dossier listing; an unknown policy is recorded as missing."""
    claim = s["bb:claim"]
    tool_errors = list(s.get("bb:tool_errors", []))
    policy, documents = None, []
    # Guardrails first, on the claim form and the raw dossier (read locally: this is a safety
    # check, so it does not depend on a remote tool being honest or available).
    guard = evaluate_claim(claim, EvidenceStore().get_all_claim_documents(claim.get("claim_id", "")))
    audit = AuditLogger(claim["claim_id"])
    for violation in guard.violations:
        audit.log_conflict(f"Guardrail {violation['action']}: {violation['kind']} in {violation['field']}")
    claim = guard.sanitized_claim
    try:
        policy = await tools.get_policy_terms(claim["policy_id"])
        documents = await tools.list_documents(claim["claim_id"])
    except ToolCallError as exc:
        tool_errors = _tool_error(s, "IntakeAgent", exc)
    _audit(s, "IntakeAgent", f"guardrail={guard.action}; policy={claim['policy_id']} found={policy is not None}; "
                             f"{len(documents)} documents; tools={tools.backend}")
    return {
        "bb:claim": claim,
        "bb:guardrail": guard.to_dict(),
        "bb:policy": policy,
        "bb:documents": sorted(documents),
        "bb:evidence": {}, "bb:conflicts": [], "bb:missing": [], "bb:tool_errors": tool_errors,
        "bb:model_payout_claim": None, "bb:model_tool_calls": [],
        "bb:agent_sequence": _sequence(s, "IntakeAgent"),
    }


async def evidence_agent(s: Dict[str, Any], tools) -> Dict[str, Any]:
    """No-key mode: extract purchase facts from invoices and repair facts from diagnostics."""
    claim_id = s["bb:claim"]["claim_id"]
    delta: Dict[str, Any] = {}
    working = dict(s)
    for doc in s.get("bb:documents", []):
        fields = INVOICE_FIELDS if is_invoice(doc) else DIAGNOSTIC_FIELDS
        try:
            extracted = await tools.extract(claim_id, doc, fields)
        except ToolCallError as exc:
            working["bb:tool_errors"] = delta["bb:tool_errors"] = _tool_error(working, "EvidenceAgent", exc)
            continue
        merged = merge_evidence(working, extracted)
        working.update(merged)
        delta.update(merged)
    _audit(s, "EvidenceAgent", f"evidence fields={sorted(working.get('bb:evidence', {}))}", "extract_document_facts")
    delta["bb:agent_sequence"] = _sequence(s, "EvidenceAgent")
    return delta


async def anomaly_agent(s: Dict[str, Any], tools) -> Dict[str, Any]:
    """Cross-document serial consistency (deterministic anomaly detector)."""
    claim_id = s["bb:claim"]["claim_id"]
    delta: Dict[str, Any] = {}
    serials: Dict[str, str] = {}
    result = {"has_anomalies": False, "anomalies": [], "clean_serial": ""}
    try:
        for doc in s.get("bb:documents", []):
            item = (await tools.extract(claim_id, doc, ["serial_number"])).get("serial_number")
            if item:
                serials[doc] = str(item.extracted_value)
        inv = next((v for d, v in serials.items() if is_invoice(d)), "")
        diag = next((v for d, v in serials.items() if not is_invoice(d)), "")
        result = await tools.flag_serials(inv or diag, inv, diag)
    except ToolCallError as exc:
        delta["bb:tool_errors"] = _tool_error(s, "AnomalyAgent", exc)

    conflicts = list(s.get("bb:conflicts", []))
    audit = _audit(s, "AnomalyAgent", f"has_anomalies={result['has_anomalies']}", "flag_serial_anomalies")
    for anomaly in result["anomalies"]:
        if anomaly not in conflicts:
            conflicts.append(anomaly)
            audit.log_conflict(anomaly)
    delta.update({"bb:anomalies": result, "bb:conflicts": conflicts,
                  "bb:agent_sequence": _sequence(s, "AnomalyAgent")})
    return delta


async def forensics_review_agent(s: Dict[str, Any], tools=None) -> Dict[str, Any]:
    """
    Checks the CrewAI forensics reply that arrived over A2A (forensics mode only).
    - missing reply / transport error / timeout -> recorded as a tool error (fail closed);
    - reply that is not a valid ForensicsReport (e.g. carries a payout or approval
      field, or names another claim) -> rejected as a tool error (fail closed);
    - CrewAI and ADK disagreeing about the serial check -> unresolved conflict;
    - any CrewAI FLAG -> conflict, so a human reviews the claim;
    - LLM stage (CrewAI Agent/Task/Crew): not configured -> disclosed as deterministic-only;
      configured but failed, timed out or returned invalid output -> tool error (fail closed);
      execution flags that do not add up -> report rejected; medium/high findings,
      suspicious instructions, insufficient evidence, a review recommendation or a
      contradiction of a deterministic check -> conflict (escalate).
    The CrewAI result can only add caution; it can never approve or pay.
    """
    claim_id = s["bb:claim"]["claim_id"]
    conflicts = list(s.get("bb:conflicts", []))
    tool_errors = list(s.get("bb:tool_errors", []))
    raw, error = s.get("bb:forensics_raw"), s.get("bb:forensics_error")
    report = None
    audit = AuditLogger(claim_id)

    if raw is None:
        tool_errors = _tool_error(s, "ForensicsReviewAgent", RuntimeError(f"CrewAI forensics unavailable: {error}"))
    else:
        try:
            if not raw.startswith(FORENSICS_JSON_PREFIX):
                raise ValueError("reply is not a ClaimSleuth forensics report")
            report = ForensicsReport.model_validate_json(raw[len(FORENSICS_JSON_PREFIX):])
            if report.claim_id != claim_id:
                raise ValueError(f"report is for claim {report.claim_id}, expected {claim_id}")
            review = report.llm_review
            if review is not None and review.status == "COMPLETED" and not (
                    review.crewai_crew_executed and review.llm_call_performed):
                raise ValueError("llm_review claims COMPLETED without an executed Crew and LLM call")
            if review is not None and review.status != "COMPLETED" and review.findings:
                raise ValueError(f"llm_review carries findings although its status is {review.status}")
        except Exception as exc:
            report = None
            tool_errors = _tool_error(s, "ForensicsReviewAgent",
                                      RuntimeError(f"Invalid CrewAI forensics report rejected: {str(exc)[:300]}"))

    if report is not None:
        crew_serial = next((f.result for f in report.findings if f.check == "serial_consistency"), "NOT_EVALUATED")
        adk_serial_flag = bool((s.get("bb:anomalies") or {}).get("has_anomalies"))
        if crew_serial != "NOT_EVALUATED" and (crew_serial == "FLAG") != adk_serial_flag:
            msg = (f"Agent disagreement: ADK AnomalyAgent reports serial anomaly={adk_serial_flag} but "
                   f"CrewAI forensics reports serial_consistency={crew_serial}")
            conflicts.append(msg)
            audit.log_conflict(msg)
        for finding in report.flagged():
            msg = f"CrewAI forensic flag [{finding.check}]: {finding.detail}"
            if msg not in conflicts:
                conflicts.append(msg)
                audit.log_conflict(msg)
        review = report.llm_review
        if review is not None and review.status not in ("COMPLETED", "LLM_NOT_CONFIGURED"):
            tool_errors = _tool_error(s, "ForensicsReviewAgent", RuntimeError(
                f"CrewAI LLM review failed: {review.status} ({review.error_category})"))
        elif review is not None and review.status == "COMPLETED":
            llm_msgs = [f"CrewAI LLM finding [{f.finding_type}/{f.severity}]: {f.explanation[:200]} "
                        f"(sources: {', '.join(f.source_refs[:4]) or 'none'})"
                        for f in review.findings if f.severity in ("medium", "high")
                        or f.finding_type in ("suspicious_instruction", "insufficient_evidence")]
            llm_msgs += [f"CrewAI LLM contradicted a deterministic check: {c}" for c in review.contradictions]
            if not llm_msgs and review.recommendation in ("RECOMMEND_HUMAN_REVIEW", "INSUFFICIENT_EVIDENCE"):
                llm_msgs.append(f"CrewAI LLM recommendation: {review.recommendation}")
            for msg in llm_msgs:
                if msg not in conflicts:
                    conflicts.append(msg)
                    audit.log_conflict(msg)
    a2a_call = {**(s.get("bb:a2a_call") or {}), "ok": report is not None, "report_valid": report is not None}
    llm = report.llm_review if report else None
    _audit(s, "ForensicsReviewAgent",
           f"report={'valid' if report else 'missing/invalid'} recommendation="
           f"{report.recommendation if report else None} llm={llm.status if llm else None} "
           f"crew_executed={llm.crewai_crew_executed if llm else None} "
           f"llm_calls={llm.llm_calls_succeeded if llm else None} model={llm.llm_model if llm else None}",
           "crewai_a2a")
    sequence = list(s.get("bb:agent_sequence", [])) + ["CrewAIForensics", "ForensicsReviewAgent"]
    return {"bb:forensics": report.model_dump() if report else None, "bb:conflicts": conflicts,
            "bb:tool_errors": tool_errors, "bb:agent_sequence": sequence, "bb:a2a_call": a2a_call}


async def policy_retrieval_agent(s: Dict[str, Any], tools=None) -> Dict[str, Any]:
    """
    RAG: retrieve the policy wording (and the claim's own document passages) relevant to this
    claim, with source references. Writes bb:rag. Retrieval failures are recorded, never
    papered over; the VerifierAgent decides whether they block a settlement.
    """
    claim = s["bb:claim"]
    evidence = s.get("bb:evidence", {})
    cause = str((evidence.get("incident_cause") or {}).get("extracted_value") or claim.get("reported_issue", ""))
    if s.get("bb:policy") is None:
        rag = {"error": f"unknown policy {claim.get('policy_id')}: nothing to retrieve", "citations": {},
               "stated_terms": {}, "missing_citations": [], "passages": [], "quarantined": [], "warnings": []}
    else:
        try:
            documents = EvidenceStore().get_all_claim_documents(claim["claim_id"])
            rag = await asyncio.to_thread(gather_policy_evidence, claim, cause, documents)
        except Exception as exc:   # index or model failure: record it, invent nothing
            rag = {"error": f"retrieval failed: {type(exc).__name__}: {str(exc)[:200]}", "citations": {},
                   "stated_terms": {}, "missing_citations": [], "passages": [], "quarantined": [], "warnings": []}
    audit = _audit(s, "PolicyRetrievalAgent",
                   f"retriever={rag.get('retriever')} citations={sorted(rag['citations'])} "
                   f"passages={len(rag['passages'])} quarantined={len(rag['quarantined'])} error={rag.get('error')}",
                   "rag_retrieve")
    for q in rag["quarantined"]:
        audit.log_conflict(f"RAG passage quarantined (instruction-like text): {q['source_file']} §{q['section']}")
    return {"bb:rag": rag, "bb:agent_sequence": _sequence(s, "PolicyRetrievalAgent")}


async def warranty_agent(s: Dict[str, Any], tools) -> Dict[str, Any]:
    """Coverage window and exclusions (deterministic policy rules)."""
    claim, policy, evidence = s["bb:claim"], s.get("bb:policy"), s.get("bb:evidence", {})
    delta: Dict[str, Any] = {}
    eligibility: Dict[str, Any] = {"covered": None, "coverage_reason": None, "expiration_date": None,
                                   "excluded": False, "matched_exclusion": None}
    cause = claim["reported_issue"]
    if "incident_cause" in evidence:
        cause += f" {evidence['incident_cause']['extracted_value']}"

    if policy is None:
        eligibility["coverage_reason"] = f"Unknown policy {claim['policy_id']}"
    elif "purchase_date" in evidence:
        try:
            res = await tools.check_coverage(claim["policy_id"], str(evidence["purchase_date"]["extracted_value"]),
                                             claim["incident_date"], cause)
            cov = res["coverage"]
            eligibility.update(covered=cov.get("is_covered"), coverage_reason=cov.get("reason"),
                               expiration_date=cov.get("expiration_date"),
                               excluded=res["excluded"], matched_exclusion=res["matched_exclusion"])
        except ToolCallError as exc:
            delta["bb:tool_errors"] = _tool_error(s, "WarrantyAgent", exc)
            eligibility["coverage_reason"] = f"Coverage check failed: {exc}"
    else:
        eligibility["coverage_reason"] = "Purchase date not established"
        excluded, matched = check_exclusions(cause, policy["exclusions"])
        eligibility.update(excluded=excluded, matched_exclusion=matched)
    _audit(s, "WarrantyAgent", f"covered={eligibility['covered']} excluded={eligibility['excluded']}",
           "check_coverage_window")
    delta.update({"bb:eligibility": eligibility, "bb:agent_sequence": _sequence(s, "WarrantyAgent")})
    return delta


async def settlement_agent(s: Dict[str, Any], tools) -> Dict[str, Any]:
    """
    Requests a settlement PROPOSAL only when the claim is conflict-free, covered,
    not excluded and the repair cost is evidenced. The proposal is checked by the
    VerifierAgent; it never authorizes anything.
    """
    evidence, policy, elig = s.get("bb:evidence", {}), s.get("bb:policy"), s.get("bb:eligibility", {})
    delta: Dict[str, Any] = {}
    reasons = []
    if (s.get("bb:guardrail") or {}).get("action") == "block":
        reasons.append("blocked by input guardrail")
    if s.get("bb:conflicts"):
        reasons.append("unresolved conflicts")
    if policy is None:
        reasons.append("unknown policy")
    if elig.get("covered") is not True:
        reasons.append("coverage not established")
    if elig.get("excluded"):
        reasons.append("excluded peril")
    if "repair_cost" not in evidence:
        reasons.append("repair cost not evidenced")

    proposal = None
    if not reasons:
        try:
            proposal = await tools.calculate_settlement(s["bb:claim"]["policy_id"],
                                                        float(evidence["repair_cost"]["extracted_value"]))
        except ToolCallError as exc:
            delta["bb:tool_errors"] = _tool_error(s, "SettlementAgent", exc)
            reasons.append("settlement tool error")
    _audit(s, "SettlementAgent",
           f"proposed={proposal['approved_payout'] if proposal else None}; withheld because: {reasons or 'n/a'}",
           "compute_settlement")
    delta.update({"bb:settlement_proposal": proposal, "bb:agent_sequence": _sequence(s, "SettlementAgent")})
    return delta


def build_claim_state(s: Dict[str, Any]) -> ClaimInvestigationState:
    """Rebuild the canonical Pydantic state from the Blackboard for the verifier."""
    claim = s["bb:claim"]
    state = ClaimInvestigationState(**{k: claim[k] for k in (
        "claim_id", "policy_id", "claimant_id", "device_model", "incident_date", "reported_issue")},
                                     claimant_statement=claim.get("claimant_statement"))
    state.evidence = {k: EvidenceItem(**v) for k, v in s.get("bb:evidence", {}).items()}
    state.conflicts = list(s.get("bb:conflicts", []))
    state.guardrail_flags = list((s.get("bb:guardrail") or {}).get("flags", []))
    proposal = s.get("bb:settlement_proposal")
    state.deterministic_settlement = SettlementCalc(**proposal) if proposal else None
    return state


async def verifier_agent(s: Dict[str, Any], tools=None) -> Dict[str, Any]:
    """
    Final authority. Always uses the LOCAL deterministic code (never the tool
    gateway), so a wrong or tampered remote result cannot approve itself:
    - a model-stated payout must equal the deterministic proposal (else conflict);
    - a settlement proposal that differs from the local recomputation is a conflict;
    - any recorded tool error makes the evidence incomplete (fail closed).
    """
    state = build_claim_state(s)
    audit = AuditLogger(state.claim_id)
    model_claim = s.get("bb:model_payout_claim")
    if model_claim is not None:
        proposal = state.deterministic_settlement
        if proposal is None or abs(float(model_claim) - proposal.approved_payout) > 0.009:
            mismatch = (f"Model-stated payout ${float(model_claim):.2f} does not match the deterministic "
                        f"settlement {'$%.2f' % proposal.approved_payout if proposal else '(none)'}; "
                        f"model output is not authoritative")
            state.conflicts.append(mismatch)
            audit.log_conflict(mismatch)

    policy = load_policy(state.policy_id)
    if policy is None:
        state.missing_fields.append("policy")
        report = VerifierReport(is_approved_by_verifier=False, gate_1_evidence_substantiated=False,
                                gate_2_deterministic_rules_passed=False, gate_2_arithmetic_matches=False,
                                discrepancies=[f"Missing policy terms: unknown policy {state.policy_id}"])
    else:
        report = DualGateVerifier().verify_investigation(state, policy)
        if not report.gate_2_arithmetic_matches and state.deterministic_settlement is not None:
            # The proposal (possibly from a remote tool) disagrees with the local recomputation
            mismatch = "Settlement proposal does not match the verifier's independent recomputation"
            state.conflicts.append(mismatch)
            audit.log_conflict(mismatch)
            report = DualGateVerifier().verify_investigation(state, policy)

    # RAG check: the retrieved policy wording must agree with the terms used for the money.
    rag = s.get("bb:rag")
    if rag is None and s["bb:run"].get("rag"):   # RAG was on but its evidence never arrived: fail closed
        rag = {"error": "policy wording evidence did not reach the verifier", "citations": {}, "stated_terms": {},
               "missing_citations": [], "quarantined": []}
    if rag is not None and policy is not None:
        check = check_policy_citations(rag, policy)
        for conflict in check["conflicts"]:
            if conflict not in state.conflicts:
                state.conflicts.append(conflict)
                audit.log_conflict(conflict)
        if check["conflicts"]:
            report = DualGateVerifier().verify_investigation(state, policy)
        if state.deterministic_settlement is not None and (rag.get("error") or check["missing"]):
            reason = rag.get("error") or f"no retrieved policy wording for {check['missing']}"
            report.is_approved_by_verifier = False
            report.gate_1_evidence_substantiated = False
            report.discrepancies.append(f"Missing policy wording citation: {reason}")

    tool_errors = list(s.get("bb:tool_errors", []))
    if tool_errors:
        report.is_approved_by_verifier = False
        report.gate_1_evidence_substantiated = False
        report.discrepancies.append(f"Missing reliable tool results ({len(tool_errors)} tool error(s)): {tool_errors[0]}")

    state.verifier_report = report
    state.status, state.requires_human_approval = determine_status(state, report)
    audit.log_verifier_result(report.is_approved_by_verifier, report.discrepancies)
    _audit(s, "VerifierAgent", f"approved={report.is_approved_by_verifier} status={state.status}", "dual_gate_verifier")

    missing = [f for f in REQUIRED_FIELDS if f not in s.get("bb:evidence", {})] + state.missing_fields
    return {
        "bb:conflicts": state.conflicts,
        "bb:missing": missing,
        "bb:verifier": report.model_dump(),
        "bb:status": state.status,
        "bb:requires_human_approval": state.requires_human_approval,
        "bb:payout": payout_view(state),
        "bb:agent_sequence": _sequence(s, "VerifierAgent"),
    }


def _errors(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Structured error list for monitoring: tool errors, A2A failures and guardrail blocks."""
    errors = []
    for e in s.get("bb:tool_errors", []):
        agent, _, message = e.partition(": ")
        errors.append({"agent": agent, "kind": "tool_error", "message": message[:300]})
    call = s.get("bb:a2a_call") or {}
    if call and not call.get("ok"):
        errors.append({"agent": "CrewAIForensics", "kind": "a2a_error", "message": str(call.get("error"))[:300]})
    for v in (s.get("bb:guardrail") or {}).get("violations", []):
        if v.get("action") == "block":
            errors.append({"agent": "IntakeAgent", "kind": f"guardrail_{v['kind']}", "message": v["field"]})
    for v in s.get("bb:model_guardrail", []):
        errors.append({"agent": "InvestigatorLLM", "kind": f"model_guardrail_{v['kind']}", "message": v["action"]})
    return errors


def _agent_metrics(s: Dict[str, Any], sequence: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Per-agent metrics (C-5) from what was measured: latency from the timing callbacks, tool
    calls, errors, and LLM token usage ONLY where a model reported it (otherwise None).
    """
    metrics: Dict[str, Dict[str, Any]] = {}
    for agent in dict.fromkeys(sequence):
        metrics[agent] = {"latency_ms": None, "tool_calls": 0, "errors": 0, "llm_calls": None,
                          "input_tokens": None, "output_tokens": None}
    for t in s.get("bb:agent_timings", []):
        m = metrics.setdefault(t["agent"], {"latency_ms": None, "tool_calls": 0, "errors": 0, "llm_calls": None,
                                            "input_tokens": None, "output_tokens": None})
        m["latency_ms"] = round((m["latency_ms"] or 0) + (t["end_ms"] - t["start_ms"]), 1)
    for c in s.get("bb:tool_calls", []):
        if c.get("agent") in metrics:
            metrics[c["agent"]]["tool_calls"] += 1
    for c in s.get("bb:model_tool_calls", []):
        metrics.setdefault("InvestigatorLLM", metrics.get("InvestigatorLLM") or {"latency_ms": None, "tool_calls": 0,
                           "errors": 0, "llm_calls": None, "input_tokens": None, "output_tokens": None})
        metrics["InvestigatorLLM"]["tool_calls"] += 1
    for e in _errors(s):
        if e["agent"] in metrics:
            metrics[e["agent"]]["errors"] += 1
    for u in s.get("bb:llm_usage", []):
        m = metrics.setdefault(u["agent"], {"latency_ms": None, "tool_calls": 0, "errors": 0, "llm_calls": None,
                                            "input_tokens": None, "output_tokens": None})
        m["llm_calls"] = (m["llm_calls"] or 0) + 1
        if u.get("input_tokens") is not None:
            m["input_tokens"] = (m["input_tokens"] or 0) + u["input_tokens"]
        if u.get("output_tokens") is not None:
            m["output_tokens"] = (m["output_tokens"] or 0) + u["output_tokens"]
    return metrics


def _messaging_summary(s: Dict[str, Any]) -> Dict[str, Any]:
    """How much context the agents received under the messaging mode (computed, not estimated)."""
    stats = s.get("bb:context_stats", [])
    messages = s.get("bb:messages", [])
    return {
        "messages_sent": len(messages),
        "deliveries": sum(c["messages_received"] for c in stats),
        "context_bytes_total": sum(c["bytes_visible"] for c in stats),
        "context_bytes_per_agent": round(sum(c["bytes_visible"] for c in stats) / max(1, len(stats)), 1),
        "keys_visible_per_agent": {c["agent"]: c["keys_visible"] for c in stats},
    }


ACTION_BY_STATUS = {
    "VERIFIED_PENDING_APPROVAL": "PROPOSE_PAYOUT_FOR_HUMAN_APPROVAL",
    "REJECTED_DETERMINISTICALLY": "REJECT",
    "CONFLICT_ESCALATED": "ESCALATE_TO_HUMAN",
    "NEEDS_EVIDENCE": "REQUEST_EVIDENCE",
}


async def report_agent(s: Dict[str, Any], tools=None) -> Dict[str, Any]:
    """Builds the Pydantic FinalReport and writes JSON + Markdown files."""
    from src.adk_app.report import FinalReport, write_report_files

    claim, run, status = s["bb:claim"], s["bb:run"], s["bb:status"]
    sequence = _sequence(s, "ReportAgent")
    _audit(s, "ReportAgent", f"status={status}")

    risk_flags = []
    if s.get("bb:conflicts"):
        risk_flags.append("EVIDENCE_CONFLICT")
    if (s.get("bb:anomalies") or {}).get("has_anomalies"):
        risk_flags.append("SERIAL_MISMATCH")
    if any("Model-stated payout" in c for c in s.get("bb:conflicts", [])):
        risk_flags.append("MODEL_PAYOUT_MISMATCH")
    if any("independent recomputation" in c for c in s.get("bb:conflicts", [])):
        risk_flags.append("TOOL_RESULT_MISMATCH")
    if s.get("bb:tool_errors"):
        risk_flags.append("TOOL_ERROR")
    rag = s.get("bb:rag") or {}
    if rag.get("quarantined"):
        risk_flags.append("RAG_PASSAGE_QUARANTINED")
    if any("Policy wording (" in c for c in s.get("bb:conflicts", [])):
        risk_flags.append("POLICY_WORDING_MISMATCH")
    if any("Missing policy wording citation" in d for d in (s.get("bb:verifier") or {}).get("discrepancies", [])):
        risk_flags.append("RAG_CITATION_MISSING")
    if any("Agent disagreement" in c for c in s.get("bb:conflicts", [])):
        risk_flags.append("AGENT_DISAGREEMENT")
    if any("Invalid CrewAI forensics report" in e for e in s.get("bb:tool_errors", [])):
        risk_flags.append("CREW_RESULT_INVALID")
    if any("CrewAI forensics unavailable" in e for e in s.get("bb:tool_errors", [])):
        risk_flags.append("CREW_SERVICE_UNAVAILABLE")
    llm_review = (s.get("bb:forensics") or {}).get("llm_review") or {}
    if s.get("bb:forensics") and llm_review.get("status", "LLM_NOT_CONFIGURED") == "LLM_NOT_CONFIGURED":
        risk_flags.append("CREWAI_LLM_NOT_CONFIGURED")
    if any("CrewAI LLM review failed" in e for e in s.get("bb:tool_errors", [])):
        risk_flags.append("CREWAI_LLM_FAILED")
    if any(c.startswith("CrewAI LLM") for c in s.get("bb:conflicts", [])):
        risk_flags.append("CREWAI_LLM_FLAG")
    if llm_review.get("rejected_findings"):
        risk_flags.append("CREWAI_LLM_FINDING_REJECTED")
    if any("SECURITY WARNING" in (ev.get("verification_notes") or "") for ev in s.get("bb:evidence", {}).values()):
        risk_flags.append("PROMPT_INJECTION_PATTERN")

    payout = s["bb:payout"]
    summary = {
        "VERIFIED_PENDING_APPROVAL": f"Evidence verified and claim eligible. A settlement of "
                                     f"${(payout.get('proposed_payout') or 0):.2f} is proposed and awaits human approval.",
        "REJECTED_DETERMINISTICALLY": "Claim fails deterministic eligibility rules: "
                                      f"{s.get('bb:eligibility', {}).get('coverage_reason')}.",
        "CONFLICT_ESCALATED": f"Unresolved conflict(s) found; escalated to a human reviewer. "
                              f"First conflict: {(s.get('bb:conflicts') or [''])[0]}",
        "NEEDS_EVIDENCE": f"Required evidence is missing or unreliable: {s.get('bb:missing')}"
                          f"{'; tool errors: ' + str(len(s.get('bb:tool_errors'))) if s.get('bb:tool_errors') else ''}.",
    }.get(status, f"Status {status}.")

    report = FinalReport(
        run_id=run["run_id"], claim_id=claim["claim_id"], policy_id=claim["policy_id"],
        pipeline_backend=run["backend"], planner=run["planner"], tool_backend=run["tool_backend"],
        status=status, summary=summary,
        recommended_action=ACTION_BY_STATUS.get(status, "ESCALATE_TO_HUMAN"),
        requires_human_approval=s["bb:requires_human_approval"],
        evidence=[{"field_name": k, "value": v["extracted_value"], "source_document": v["source_document"],
                   "raw_snippet": v["raw_snippet"], "confidence": v["confidence"], "status": v["status"]}
                  for k, v in s.get("bb:evidence", {}).items()],
        conflicts=s.get("bb:conflicts", []), missing_evidence=s.get("bb:missing", []),
        eligibility=s.get("bb:eligibility", {}),
        settlement_proposal=s.get("bb:settlement_proposal"),
        model_payout_claim=s.get("bb:model_payout_claim"),
        verifier=s["bb:verifier"], payout=payout, risk_flags=risk_flags,
        model_tool_calls=s.get("bb:model_tool_calls", []), tool_calls=s.get("bb:tool_calls", []),
        tool_errors=s.get("bb:tool_errors", []), agent_sequence=sequence,
        forensics_mode=run.get("forensics", "none"), forensics=s.get("bb:forensics"),
        a2a_call=s.get("bb:a2a_call"),
        pattern=run.get("pattern", "sequential"), messaging=run.get("messaging", "blackboard"),
        agent_timings=s.get("bb:agent_timings", []), manager_decisions=s.get("bb:manager_decisions", []),
        messaging_summary=_messaging_summary(s),
        trace_id=run.get("trace_id"), guardrail=s.get("bb:guardrail"),
        rag=({k: rag.get(k) for k in ("retriever", "error", "warnings", "missing_citations", "stated_terms")}
             if rag else None),
        policy_citations=rag.get("citations", {}) if rag else {},
        retrieved_evidence=(rag.get("passages", []) + [{**q, "quarantined": True} for q in rag.get("quarantined", [])])
        if rag else [],
        agent_metrics=_agent_metrics(s, sequence), errors=_errors(s),
        audit_chain_valid=AuditLogger(claim["claim_id"]).verify_chain()["valid"],
    )
    files = write_report_files(report, REPORTS_DIR)
    return {"bb:final_report": report.model_dump(mode="json"), "bb:report_files": files,
            "bb:agent_sequence": sequence}
