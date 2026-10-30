"""
Model rate limits on the free tier.

- pace(): spaces model calls to config.MODEL_MAX_RPM and returns how long it waited, so that wait can be kept
  out of the agent timings. The free-tier limit counts the app's and the crew's calls together, so there is one
  pacer, in the app: the crew service asks for its slots at /internal/rate-slot (request_slot()).
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
SERVES_SLOTS = False          # True inside the app process, which owns the pacer


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


def request_slot() -> float:
    """Outside the app (the crew service, or a run started from a script or pytest): wait for a model-call slot
    from the app's pacer; returns the seconds spent waiting. Only when no app is running at all (offline tests)
    does it pace locally."""
    import httpx
    started = time.monotonic()
    try:
        response = httpx.post(f"{config.APP_URL}/internal/rate-slot",
                              headers={"X-ClaimSleuth-Secret": config.A2A_SHARED_SECRET}, timeout=300)
    except httpx.ConnectError:
        return pace_sync()
    response.raise_for_status()
    return min(time.monotonic() - started, float(response.json()["waited"]) + 1.0)


async def acquire() -> float:
    """Wait for a model-call slot from the one pacer: directly in the app, over HTTP from anywhere else."""
    if SERVES_SLOTS:
        return await pace()
    return await asyncio.to_thread(request_slot)


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
