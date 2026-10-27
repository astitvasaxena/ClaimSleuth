"""
Client for the ClaimSleuth MCP server over a real transport.

    async with RemoteMCPClient.stdio() as client:              # spawns the server process
        tools = await client.list_tools()
        result = await client.call_tool("calculate_policy_settlement",
                                        {"policy_id": "POL-TECH-CARE-01", "repair_cost": 280})

    async with RemoteMCPClient.http("http://127.0.0.1:8102/mcp") as client:   # server already running
        ...

    call_remote_tool("get_policy_terms", {"policy_id": "POL-TECH-CARE-01"})   # one-shot, synchronous

A tool error (isError=True) raises MCPToolError; it is never converted into a result.
"""
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TIMEOUT_SECONDS = 30.0


class MCPToolError(RuntimeError):
    """The server executed the request but the tool reported an error."""


class RemoteMCPClient:
    def __init__(self, transport: str, url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 env: Optional[Dict[str, str]] = None):
        if transport not in ("stdio", "http"):
            raise ValueError("transport must be 'stdio' or 'http'")
        self.transport = transport
        self.url = url
        self.timeout = timeout
        self.env = env
        self._stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None
        self.server_name: Optional[str] = None

    @classmethod
    def stdio(cls, **kwargs) -> "RemoteMCPClient":
        return cls("stdio", **kwargs)

    @classmethod
    def http(cls, url: str, **kwargs) -> "RemoteMCPClient":
        return cls("http", url=url, **kwargs)

    async def __aenter__(self) -> "RemoteMCPClient":
        self._stack = AsyncExitStack()
        try:
            if self.transport == "stdio":
                params = StdioServerParameters(
                    command=sys.executable, args=["-m", "src.mcp.remote_server", "--transport", "stdio"],
                    cwd=str(PROJECT_ROOT), env={**os.environ, **(self.env or {})})
                read, write = await self._stack.enter_async_context(stdio_client(params))
            else:
                streams = await self._stack.enter_async_context(streamable_http_client(self.url))
                read, write = streams[0], streams[1]
            self.session = await self._stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=self.timeout))
            init = await asyncio.wait_for(self.session.initialize(), timeout=self.timeout)
            self.server_name = init.server_info.name
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def list_tools(self) -> List[Dict[str, Any]]:
        result = await self.session.list_tools()
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in result.tools]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = await self.session.call_tool(name, arguments)
        text = "\n".join(getattr(c, "text", "") for c in (result.content or []))
        if result.is_error:
            raise MCPToolError(f"{name} failed: {text}")
        structured = result.structured_content
        if isinstance(structured, dict):
            # The SDK wraps non-object return types as {"result": ...}; our tools return objects.
            return structured.get("result", structured) if set(structured) == {"result"} else structured
        return json.loads(text)


def call_remote_tool(name: str, arguments: Dict[str, Any], transport: str = "stdio",
                     url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """One-shot synchronous call (opens a session, calls one tool, closes). Not for use inside a running loop."""
    async def _go():
        async with RemoteMCPClient(transport, url=url, timeout=timeout) as client:
            return await client.call_tool(name, arguments)
    return asyncio.run(_go())
