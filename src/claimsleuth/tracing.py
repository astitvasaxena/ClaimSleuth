"""
Tracing (C-9): one W3C trace id per investigation, carried across ADK, MCP and A2A.

- An OpenTelemetry root span "claimsleuth.investigation" is opened per run; ADK's own
  agent/node spans become its children. Spans are exported to
  <CLAIMSLEUTH_TRACES_DIR>/<trace_id>.jsonl (local files; no external service).
- CURRENT_TRACE_ID is a contextvar the audit ledger stamps on every entry.
- MCP: ADK's McpTool already injects the W3C traceparent into each request's _meta;
  the ClaimSleuth MCP server reads it and logs it with each call.
- A2A: traceparent_headers() is added as an HTTP header to ADK's RemoteA2aAgent
  requests; the CrewAI service reads it and logs it.
"""
import contextvars
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

import claimsleuth.config as settings

CURRENT_TRACE_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("claimsleuth_trace_id", default=None)
_LOCK = threading.Lock()
_SETUP = {"done": False}


def traces_dir() -> Path:
    return Path(settings.TRACES_DIR)


class JsonlSpanExporter(SpanExporter):
    """Appends each finished span to <traces_dir>/<trace_id>.jsonl."""

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            ctx = span.get_span_context()
            trace_id = format(ctx.trace_id, "032x")
            record = {
                "trace_id": trace_id,
                "span_id": format(ctx.span_id, "016x"),
                "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
                "name": span.name,
                "start_ns": span.start_time, "end_ns": span.end_time,
                "duration_ms": round((span.end_time - span.start_time) / 1e6, 3) if span.end_time else None,
                "status": span.status.status_code.name,
                "attributes": {k: v for k, v in (span.attributes or {}).items()
                               if isinstance(v, (str, int, float, bool)) and len(str(v)) < 300},
            }
            folder = traces_dir()
            folder.mkdir(parents=True, exist_ok=True)
            with _LOCK, open(folder / f"{trace_id}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def setup_tracing() -> None:
    """Install an SDK tracer provider with the JSONL exporter (once per process)."""
    if _SETUP["done"]:
        return
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter()))
    _SETUP["done"] = True


@contextmanager
def investigation_span(claim_id: str, run_id: str, attributes: Optional[Dict] = None) -> Iterator[str]:
    """Open the root span for one investigation and yield its 32-hex trace id."""
    setup_tracing()
    tracer = trace.get_tracer("claimsleuth")
    with tracer.start_as_current_span("claimsleuth.investigation",
                                      attributes={"claim_id": claim_id, "run_id": run_id, **(attributes or {})}) as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        token = CURRENT_TRACE_ID.set(trace_id)
        try:
            yield trace_id
        finally:
            CURRENT_TRACE_ID.reset(token)


def traceparent_headers() -> Dict[str, str]:
    """W3C trace-context headers for the current span (empty if no span is active)."""
    carrier: Dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def trace_id_from_traceparent(value: Optional[str]) -> Optional[str]:
    """'00-<32 hex trace id>-<16 hex span id>-<flags>' -> trace id, else None."""
    parts = (value or "").split("-")
    return parts[1] if len(parts) == 4 and len(parts[1]) == 32 else None


def read_trace(trace_id: str, limit: int = 500) -> List[Dict]:
    path = traces_dir() / f"{trace_id}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()][:limit]
