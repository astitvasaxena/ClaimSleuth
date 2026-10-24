"""
ClaimSleuth — Model Context Protocol (MCP) Tool Server & Gateway
Implements Master Document Section 12.2 (MCP Interoperability Boundary)
and satisfies Teacher Requirement for at least two working MCP tools.

Exposes deterministic, auditable domain tools via the official Model Context Protocol (MCP 2.x):
1. check_coverage_window: Deterministic calendar date validation for warranty duration.
2. compute_settlement: Deterministic mathematical formula execution for payout.
3. flag_serial_anomalies: Serial number discrepancy detection across documents.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from mcp.server.mcpserver import MCPServer

from src.tools.policy_rules import calc_coverage_window as base_check_coverage
from src.tools.math_engine import compute_settlement as base_compute_settlement
from src.tools.anomaly_detector import flag_serial_anomalies as base_flag_serial


# Initialize official MCP Server instance
mcp_server = MCPServer("ClaimSleuthDeterministicTools")


# --- Tool 1: check_coverage_window ---
@mcp_server.tool(
    name="check_coverage_window",
    description="Deterministic verification of whether an incident date falls within the policy's warranty duration from purchase date."
)
def check_coverage_window_mcp(
    purchase_date: str,
    incident_date: str,
    duration_months: int = 12
) -> Dict[str, Any]:
    """MCP tool for coverage window validation."""
    return base_check_coverage(
        purchase_date_str=purchase_date,
        incident_date_str=incident_date,
        duration_months=duration_months
    )


# --- Tool 2: compute_settlement ---
@mcp_server.tool(
    name="compute_settlement",
    description="Deterministic mathematical formula execution for claim payout (repair_cost - deductible - depreciation, capped at max_limit)."
)
def compute_settlement_mcp(
    repair_cost: float,
    deductible: float,
    max_limit: float = 1500.0,
    depreciation: float = 0.0
) -> Dict[str, Any]:
    """MCP tool for settlement payout calculation."""
    res = base_compute_settlement(
        repair_cost=repair_cost,
        deductible=deductible,
        depreciation_rate=depreciation,
        max_coverage_limit=max_limit
    )
    return res.model_dump()


# --- Tool 3: flag_serial_anomalies ---
@mcp_server.tool(
    name="flag_serial_anomalies",
    description="Cross-document serial number consistency check to identify fraudulent or conflicting equipment tags."
)
def flag_serial_anomalies_mcp(
    claim_serial: str = "",
    invoice_serial: str = "",
    diagnostic_serial: str = ""
) -> Dict[str, Any]:
    """MCP tool for serial number anomaly detection."""
    return base_flag_serial(
        claim_serial=claim_serial,
        invoice_serial=invoice_serial,
        diagnostic_serial=diagnostic_serial
    )


class MCPToolClient:
    """
    Client gateway for agents to discover and invoke tools via the MCP protocol.
    Works synchronously or asynchronously.
    """
    def __init__(self, server: Optional[MCPServer] = None):
        self.server = server or mcp_server

    async def list_tools_async(self) -> List[Dict[str, Any]]:
        """List all tools registered in the MCP server."""
        tools = await self.server.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": getattr(t, "inputSchema", {})
            }
            for t in tools
        ]

    async def call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke an MCP tool asynchronously."""
        result = await self.server.call_tool(tool_name, arguments)
        # Parse MCP text content into structured python dict
        if hasattr(result, "content") and result.content:
            for item in result.content:
                if hasattr(item, "text"):
                    try:
                        return json.loads(item.text)
                    except json.JSONDecodeError:
                        return {"raw_text": item.text}
        return {"result": str(result)}

    def _run_coroutine(self, coro):
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            else:
                return asyncio.run(coro)
        except Exception:
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()

    def list_tools(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper to list registered MCP tools."""
        return self._run_coroutine(self.list_tools_async())

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper to invoke an MCP tool."""
        return self._run_coroutine(self.call_tool_async(tool_name, arguments))
