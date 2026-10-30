"""
LIVE checks with the real Gemini model (they spend model calls). Not part of the normal test run:

    pytest -m live -v

They need GOOGLE_API_KEY, the app running at APP_URL (for /mcp) and the CrewAI service at CREW_URL.
The three demo scenarios in all three patterns (step 4: 9 runs), and Scenario 3's fraud flag (step 5).
"""
import asyncio

import httpx
import pytest

from claimsleuth import config, runner

pytestmark = pytest.mark.live

EXPECTED = {"CLM-2024-001": ({"approve"}, "251.00", False),
            "CLM-2024-002": ({"reject"}, None, False),
            "CLM-2024-003": ({"reject", "request_evidence"}, None, True)}


def services_up() -> bool:
    try:
        return all(httpx.get(f"{url}/healthz", timeout=5).status_code == 200 for url in (config.APP_URL, config.CREW_URL))
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module", autouse=True)
def need_services():
    if not config.model_configured():
        pytest.skip("GOOGLE_API_KEY is not set")
    if not services_up():
        pytest.skip(f"start the app ({config.APP_URL}) and the CrewAI service ({config.CREW_URL}) first")


@pytest.mark.parametrize("pattern", ["sequential", "parallel", "hierarchical"])
@pytest.mark.parametrize("claim_id", sorted(EXPECTED))
def test_demo_decision(claim_id, pattern):
    result = asyncio.run(runner.run_claim(claim_id, pattern, "direct"))
    assert result["status"] == "completed", result["error"]
    report = result["report"]
    decisions, payout, fraud = EXPECTED[claim_id]
    assert report["decision"] in decisions
    if payout:
        assert report["payout"]["amount"] == payout
    if fraud:
        assert report["fraud"]["flagged"], "the crew should flag Scenario 3's serial mismatch"
        assert result["blackboard"]["bb_fraud"]["flagged"]
