"""
How the ADK pipeline's agents reach their tools.

  LocalTools  calls the deterministic Python functions directly (no network).
  McpTools    calls the same operations on the ClaimSleuth MCP server through
              ADK's McpToolset, over stdio (spawned server) or streamable HTTP.

Both expose the same async methods, so the agents in steps.py do not change.
MCP results are inputs to the pipeline, never decisions: the VerifierAgent always
recomputes with the local deterministic code and has the final say.
"""
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from src.agents.extractor import EvidenceExtractor
from src.core.state import EvidenceItem
from src.tools.anomaly_detector import flag_serial_anomalies
from src.tools.evidence_store import EvidenceStore
from src.tools.math_engine import compute_settlement
from src.tools.policy_rules import calc_coverage_window, check_exclusions, load_policy

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MCP_URL = "http://127.0.0.1:8102/mcp"


class ToolCallError(RuntimeError):
    """A tool ran but reported an error (MCP isError, or invalid input)."""


class MCPUnavailableError(RuntimeError):
    """The MCP server could not be reached or started."""


@dataclass
class McpConfig:
    transport: Literal["stdio", "http"] = "stdio"
    # Read when the config is created, so CLAIMSLEUTH_MCP_URL can change without a restart
    url: str = field(default_factory=lambda: os.environ.get("CLAIMSLEUTH_MCP_URL", DEFAULT_MCP_URL))
    # stdio only: the server command's arguments (tests point this at a fault-injection server)
    server_args: Optional[List[str]] = None
    timeout: float = 30.0


class LocalTools:
    """Direct calls into the deterministic code (the default, no-network path)."""
    backend = "local"

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def _log(self, tool: str, args: Dict[str, Any]):
        self.calls.append({"tool": tool, "args": args, "via": "local", "ok": True})

    async def get_policy_terms(self, policy_id: str) -> Optional[Dict[str, Any]]:
        self._log("get_policy_terms", {"policy_id": policy_id})
        policy = load_policy(policy_id)
        return policy.model_dump() if policy else None

    async def list_documents(self, claim_id: str) -> List[str]:
        self._log("list_claim_documents", {"claim_id": claim_id})
        return sorted(EvidenceStore().list_documents(claim_id))

    async def extract(self, claim_id: str, document_name: str, fields: List[str]) -> Dict[str, EvidenceItem]:
        self._log("extract_claim_document_facts", {"claim_id": claim_id, "document_name": document_name, "fields": fields})
        text = EvidenceStore().read_document(claim_id, document_name) or ""
        return EvidenceExtractor(llm_client=None).extract_from_document(document_name, text, fields, use_llm=False)

    async def check_coverage(self, policy_id: str, purchase_date: str, incident_date: str,
                             reported_cause: str) -> Dict[str, Any]:
        self._log("check_policy_coverage", {"policy_id": policy_id})
        policy = load_policy(policy_id)
        window = calc_coverage_window(purchase_date, incident_date, policy.duration_months)
        if not window.get("is_valid"):
            raise ToolCallError(window.get("error", "Invalid dates"))
        excluded, matched = check_exclusions(reported_cause, policy.exclusions)
        return {"coverage": window, "excluded": excluded, "matched_exclusion": matched}

    async def calculate_settlement(self, policy_id: str, repair_cost: float) -> Dict[str, Any]:
        self._log("calculate_policy_settlement", {"policy_id": policy_id, "repair_cost": repair_cost})
        p = load_policy(policy_id)
        return compute_settlement(repair_cost, p.deductible, p.max_coverage_limit, p.depreciation_rate).model_dump()

    async def flag_serials(self, claim_serial: str, invoice_serial: str, diagnostic_serial: str) -> Dict[str, Any]:
        self._log("flag_serial_anomalies", {})
        return flag_serial_anomalies(claim_serial, invoice_serial, diagnostic_serial)


def build_mcp_toolset(config: McpConfig, tool_filter: Optional[List[str]] = None):
    """Create an ADK McpToolset connected to the ClaimSleuth MCP server."""
    from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams, StreamableHTTPConnectionParams
    from mcp.client.stdio import StdioServerParameters

    if config.transport == "stdio":
        params = StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=config.server_args or ["-m", "src.mcp.remote_server", "--transport", "stdio"],
                cwd=str(PROJECT_ROOT),
                env=dict(os.environ)),   # the MCP SDK passes only a few variables by default
            timeout=config.timeout)
    else:
        params = StreamableHTTPConnectionParams(url=config.url, timeout=config.timeout)
    return McpToolset(connection_params=params, tool_filter=tool_filter)


def unwrap_mcp_result(tool: str, response: Any) -> Dict[str, Any]:
    """Turn an McpTool response dict into the tool's own result, or raise ToolCallError."""
    if not isinstance(response, dict):
        raise ToolCallError(f"{tool}: unexpected MCP response type {type(response).__name__}")
    if response.get("isError"):
        text = " ".join(c.get("text", "") for c in response.get("content", []) if isinstance(c, dict))
        raise ToolCallError(f"{tool}: {text}")
    structured = response.get("structuredContent")
    if isinstance(structured, dict):
        return structured.get("result", structured) if set(structured) == {"result"} else structured
    # Servers may return results as JSON text instead of structured content
    text = "".join(c.get("text", "") for c in response.get("content", []) if isinstance(c, dict))
    try:
        parsed = json.loads(text)
    except ValueError:
        raise ToolCallError(f"{tool}: MCP response had neither structured content nor JSON text")
    if not isinstance(parsed, dict):
        raise ToolCallError(f"{tool}: MCP response was not a JSON object")
    return parsed


class McpTools:
    """The same operations, executed by the MCP server via ADK's McpToolset."""

    def __init__(self, toolset, tool_context, transport: str):
        self._toolset = toolset
        self._ctx = tool_context
        self.backend = f"mcp_{transport}"
        self.calls: List[Dict[str, Any]] = []
        self._tools = None

    async def _call(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._tools is None:
            try:
                self._tools = {t.name: t for t in await self._toolset.get_tools()}
            except Exception as exc:
                raise MCPUnavailableError(f"MCP server unavailable ({self.backend}): {type(exc).__name__}: {exc}") from exc
        if tool not in self._tools:
            raise MCPUnavailableError(f"MCP server does not provide tool '{tool}'")
        try:
            response = await self._tools[tool].run_async(args=args, tool_context=self._ctx)
        except Exception as exc:
            self.calls.append({"tool": tool, "args": args, "via": self.backend, "ok": False, "error": str(exc)})
            raise MCPUnavailableError(f"MCP call {tool} failed: {type(exc).__name__}: {exc}") from exc
        try:
            result = unwrap_mcp_result(tool, response)
        except ToolCallError as exc:
            self.calls.append({"tool": tool, "args": args, "via": self.backend, "ok": False, "error": str(exc)})
            raise
        self.calls.append({"tool": tool, "args": args, "via": self.backend, "ok": True})
        return result

    async def get_policy_terms(self, policy_id: str) -> Optional[Dict[str, Any]]:
        try:
            return await self._call("get_policy_terms", {"policy_id": policy_id})
        except ToolCallError as exc:
            if "Unknown policy_id" in str(exc):
                return None
            raise

    async def list_documents(self, claim_id: str) -> List[str]:
        try:
            return (await self._call("list_claim_documents", {"claim_id": claim_id}))["documents"]
        except ToolCallError as exc:
            if "Unknown claim_id" in str(exc):
                return []
            raise

    async def extract(self, claim_id: str, document_name: str, fields: List[str]) -> Dict[str, EvidenceItem]:
        result = await self._call("extract_claim_document_facts",
                                  {"claim_id": claim_id, "document_name": document_name, "fields": fields})
        return {k: EvidenceItem(**v) for k, v in result["facts"].items()}

    async def check_coverage(self, policy_id: str, purchase_date: str, incident_date: str,
                             reported_cause: str) -> Dict[str, Any]:
        return await self._call("check_policy_coverage", {
            "policy_id": policy_id, "purchase_date": purchase_date,
            "incident_date": incident_date, "reported_cause": reported_cause})

    async def calculate_settlement(self, policy_id: str, repair_cost: float) -> Dict[str, Any]:
        result = await self._call("calculate_policy_settlement", {"policy_id": policy_id, "repair_cost": repair_cost})
        return result["settlement"]

    async def flag_serials(self, claim_serial: str, invoice_serial: str, diagnostic_serial: str) -> Dict[str, Any]:
        return await self._call("flag_serial_anomalies", {
            "claim_serial": claim_serial, "invoice_serial": invoice_serial, "diagnostic_serial": diagnostic_serial})
