"""
Per-run monitoring: the callbacks add to one RunRecorder per run (time, tokens, model calls, tool calls, errors
and guardrail hits per agent) and pass each step to the live view. The runner writes the rows to agent_metrics.
"""
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable, Optional

from claimsleuth import config

METRIC_COLUMNS = ("seconds", "prompt_tokens", "completion_tokens", "llm_calls", "tool_calls", "errors", "guardrail_hits")


@dataclass
class AgentRow:
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    errors: int = 0
    guardrail_hits: int = 0


class RunRecorder:

    def __init__(self, run_id: str, sink: Optional[Callable[[dict], Awaitable[None]]] = None):
        self.run_id = run_id
        self.sink = sink
        self.rows: dict[str, AgentRow] = defaultdict(AgentRow)
        self._started: dict[str, list[float]] = defaultdict(list)

    def start(self, agent: str) -> None:
        self._started[agent].append(time.perf_counter())

    def stop(self, agent: str) -> None:
        if self._started[agent]:
            self.rows[agent].seconds += time.perf_counter() - self._started[agent].pop()

    async def emit(self, agent: str, kind: str, text: str = "", tool: Optional[str] = None) -> None:
        if self.sink:
            await self.sink({"agent": agent, "type": kind, "tool": tool, "text": text[:500], "ts": round(time.time(), 3)})

    def as_rows(self) -> list[dict]:
        return [{"agent": name, **{k: (round(v, 2) if k == "seconds" else v) for k, v in asdict(row).items()}}
                for name, row in sorted(self.rows.items())]

    def totals(self) -> dict:
        prompt = sum(r.prompt_tokens for r in self.rows.values())
        completion = sum(r.completion_tokens for r in self.rows.values())
        return {"prompt_tokens": prompt, "completion_tokens": completion, "cost_usd": cost_usd(prompt, completion)}


def cost_usd(prompt_tokens: int, completion_tokens: int, model: Optional[str] = None) -> Optional[float]:
    """Tokens x the provider's price for the model (config.PRICES); None when no price is recorded for it."""
    price = config.PRICES.get(model or config.GEMINI_MODEL)
    if price is None:
        return None
    return round(prompt_tokens / 1e6 * price["input_per_million"] + completion_tokens / 1e6 * price["output_per_million"], 6)


RECORDERS: dict[str, RunRecorder] = {}


def start_run(run_id: str, sink=None) -> RunRecorder:
    RECORDERS[run_id] = RunRecorder(run_id, sink)
    return RECORDERS[run_id]


def finish_run(run_id: str) -> Optional[RunRecorder]:
    return RECORDERS.pop(run_id, None)


def recorder(run_id: Optional[str]) -> Optional[RunRecorder]:
    return RECORDERS.get(run_id) if run_id else None
