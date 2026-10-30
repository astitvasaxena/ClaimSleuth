"""
Model rate limits on the free tier.

- pace(): spaces model calls to config.MODEL_MAX_RPM per process and returns how long it waited, so that wait
  can be kept out of the agent timings. The app (ADK) and the crew service (CrewAI) each pace their own calls.
- classify(): tells a daily quota error from a per-minute one, from the full error text.
"""
import asyncio
import re
import threading
import time
from typing import Optional

from claimsleuth import config

DAILY, MINUTE = "RATE_LIMIT_DAILY", "RATE_LIMIT_MINUTE"
_lock = threading.Lock()
_next_slot = 0.0


def _reserve() -> float:
    """Reserve the next call slot; return the seconds to wait for it."""
    global _next_slot
    with _lock:
        now = time.monotonic()
        start = max(now, _next_slot)
        _next_slot = start + 60.0 / config.MODEL_MAX_RPM
        return start - now


async def pace() -> float:
    delay = _reserve()
    if delay > 0:
        await asyncio.sleep(delay)
    return delay


def pace_sync() -> float:
    delay = _reserve()
    if delay > 0:
        time.sleep(delay)
    return delay


def classify(text: Optional[str]) -> Optional[str]:
    """DAILY, MINUTE or None (not a rate limit). Gemini's 429 names the quota, e.g. '... requests per day ...'."""
    text = text or ""
    if not re.search(r"\b429\b|RESOURCE_EXHAUSTED|ResourceExhausted|rate.?limit|quota", text, re.IGNORECASE):
        return None
    if re.search(r"per.?day|PerDay|daily", text, re.IGNORECASE):
        return DAILY
    return MINUTE


def quota_detail(text: str) -> str:
    """The part of a quota error that names the limit."""
    match = re.search(r"Quota exceeded for metric[^']{0,300}", text or "")
    return match.group(0) if match else (text or "")[:300]
