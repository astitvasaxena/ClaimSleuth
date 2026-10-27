"""
ClaimSleuth MCP server over a real transport (MCP Python SDK 2.3.0).

Run as its own process:
    python -m src.mcp.remote_server --transport stdio
    python -m src.mcp.remote_server --transport http --host 127.0.0.1 --port 8102   # endpoint: /mcp

Unlike the legacy in-process server in tool_server.py (kept unchanged for its
existing callers), these tools are POLICY-GROUNDED: the caller names a policy_id
and the server looks up deductible, limit, duration and depreciation itself, so a
remote client cannot omit, invent or override policy terms. Every tool wraps the
existing deterministic code; nothing here uses an LLM.

Invalid input raises ToolError, which MCP returns to the client as an
isError result instead of a fabricated answer.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from src.agents.extractor import EvidenceExtractor
from src.tools.anomaly_detector import flag_serial_anomalies as base_flag_serial
from src.tools.evidence_store import EvidenceStore
from src.tools.math_engine import compute_settlement
from src.tools.policy_rules import calc_coverage_window, check_exclusions, load_policy

# Logs go to stderr: in stdio mode stdout carries the MCP protocol itself.
# CLAIMSLEUTH_MCP_LOG_LEVEL=INFO prints every call; the JSONL call log (below) is the durable record.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s mcp-server %(message)s")
log = logging.getLogger("claimsleuth.mcp")
log.setLevel(os.environ.get("CLAIMSLEUTH_MCP_LOG_LEVEL", "WARNING").upper())

EXTRACTABLE_FIELDS = ["purchase_date", "serial_number", "repair_cost", "incident_cause"]

remote_server = MCPServer(
    "ClaimSleuthTools",
    instructions="Deterministic warranty-claim tools. Policy terms are always looked up by policy_id.",
)


def _caller_trace_id(ctx) -> Optional[str]:
    """Trace id from the W3C traceparent the client put in the request _meta (ADK's McpTool does this)."""
    try:
        meta = ctx.request_context.meta
    except Exception:
        return None
    if meta is None:
        return None
    data = meta if isinstance(meta, dict) else {**getattr(meta, "model_extra", {}), **meta.model_dump()}
    parts = str(data.get("traceparent") or "").split("-")
    return parts[1] if len(parts) == 4 and len(parts[1]) == 32 else None


def _record_call(tool: str, ctx=None, **args) -> None:
    """
    Log every tool call to stderr and, if CLAIMSLEUTH_MCP_CALL_LOG is set, append it
    (with this server's process id) to that JSONL file. The file is evidence that a
    call was executed by the server process, not in the client.
    """
    log.info("%s %s", tool, args)
    path = os.environ.get("CLAIMSLEUTH_MCP_CALL_LOG")
    if path:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "tool": tool, "args": args,
                 "trace_id": _caller_trace_id(ctx)}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def _policy_or_error(policy_id: str):
    policy = load_policy(policy_id)
    if policy is None:
        raise ToolError(f"Unknown policy_id '{policy_id}'")
    return policy


@remote_server.tool(description="List the evidence documents in a claim's dossier.")
def list_claim_documents(claim_id: str, ctx: Context = None) -> Dict[str, Any]:
    _record_call("list_claim_documents", ctx, claim_id=claim_id)
    store = EvidenceStore()
    if store.get_claim_folder(claim_id) is None:
        raise ToolError(f"Unknown claim_id '{claim_id}'")
    return {"claim_id": claim_id, "documents": sorted(store.list_documents(claim_id))}


@remote_server.tool(description="Extract facts (purchase_date, serial_number, repair_cost, incident_cause) from one "
                                "dossier document. Each fact includes the verbatim source snippet and a status.")
def extract_claim_document_facts(claim_id: str, document_name: str, fields: List[str],
                                 ctx: Context = None) -> Dict[str, Any]:
    _record_call("extract_claim_document_facts", ctx, claim_id=claim_id, document_name=document_name, fields=fields)
    bad = [f for f in fields if f not in EXTRACTABLE_FIELDS]
    if bad or not fields:
        raise ToolError(f"fields must be a non-empty subset of {EXTRACTABLE_FIELDS}; rejected {bad}")
    store = EvidenceStore()
    # Only names that are actually in the dossier listing are readable (blocks path traversal)
    if document_name not in store.list_documents(claim_id):
        raise ToolError(f"Document '{document_name}' is not in the dossier of claim '{claim_id}'")
    text = store.read_document(claim_id, document_name) or ""
    extracted = EvidenceExtractor(llm_client=None).extract_from_document(document_name, text, fields, use_llm=False)
    return {
        "claim_id": claim_id,
        "document_name": document_name,
        "facts": {k: v.model_dump() for k, v in extracted.items()},
        "not_found": [f for f in fields if f not in extracted],
    }


@remote_server.tool(description="Look up a warranty policy's terms (duration, deductible, limit, depreciation, exclusions).")
def get_policy_terms(policy_id: str, ctx: Context = None) -> Dict[str, Any]:
    _record_call("get_policy_terms", ctx, policy_id=policy_id)
    return _policy_or_error(policy_id).model_dump()


@remote_server.tool(description="Check whether an incident falls inside the policy's coverage window and whether "
                                "the reported cause matches a policy exclusion. Dates are YYYY-MM-DD.")
def check_policy_coverage(policy_id: str, purchase_date: str, incident_date: str,
                          reported_cause: str = "", ctx: Context = None) -> Dict[str, Any]:
    _record_call("check_policy_coverage", ctx, policy_id=policy_id, purchase_date=purchase_date, incident_date=incident_date)
    policy = _policy_or_error(policy_id)
    window = calc_coverage_window(purchase_date, incident_date, policy.duration_months)
    if not window.get("is_valid"):
        raise ToolError(window.get("error", "Invalid dates"))
    excluded, matched = check_exclusions(reported_cause, policy.exclusions)
    return {"policy_id": policy_id, "coverage": window, "excluded": excluded, "matched_exclusion": matched,
            "eligible": bool(window.get("is_covered")) and not excluded}


@remote_server.tool(description="Deterministic settlement for a repair cost under a policy: "
                                "min(cost - deductible, limit) minus depreciation. Proposes, never authorizes.")
def calculate_policy_settlement(policy_id: str, repair_cost: float, ctx: Context = None) -> Dict[str, Any]:
    _record_call("calculate_policy_settlement", ctx, policy_id=policy_id, repair_cost=repair_cost)
    if repair_cost < 0:
        raise ToolError("repair_cost must be non-negative")
    policy = _policy_or_error(policy_id)
    settlement = compute_settlement(repair_cost, policy.deductible, policy.max_coverage_limit,
                                    policy.depreciation_rate)
    return {"policy_id": policy_id, "settlement": settlement.model_dump(),
            "authorization": "NOT_AUTHORIZED: proposal only; requires verifier and human approval"}


@remote_server.tool(description="Cross-check serial numbers from the claim, purchase invoice and repair report.")
def flag_serial_anomalies(claim_serial: str, invoice_serial: str, diagnostic_serial: str,
                          ctx: Context = None) -> Dict[str, Any]:
    _record_call("flag_serial_anomalies", ctx)
    return base_flag_serial(claim_serial=claim_serial, invoice_serial=invoice_serial,
                            diagnostic_serial=diagnostic_serial)


REMOTE_TOOL_NAMES = ["list_claim_documents", "extract_claim_document_facts", "get_policy_terms",
                     "check_policy_coverage", "calculate_policy_settlement", "flag_serial_anomalies"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="ClaimSleuth MCP tool server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--path", default="/mcp")
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        remote_server.run("stdio")
    else:
        log.info("streamable-http listening on http://%s:%s%s", args.host, args.port, args.path)
        remote_server.run("streamable-http", host=args.host, port=args.port, streamable_http_path=args.path)


if __name__ == "__main__":
    main()
