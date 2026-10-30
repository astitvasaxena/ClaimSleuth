"""The runner with ADK's real Runner and a fake model (a test double, no API calls)."""
import asyncio
import unittest
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm, LlmResponse
from google.genai import types

from claimsleuth import config, db, runner
from claimsleuth.agents.common import submit_report
from claimsleuth.callbacks.hooks import attach_callbacks


class LoopingFakeLlm(BaseLlm):
    """TEST DOUBLE: always asks for another submit_report call with an approval that cannot be accepted."""
    model: str = "fake-looping-model"
    calls: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        call = types.FunctionCall(name="submit_report", args={"submission": {"decision": "approve", "reasons": ["again"]}})
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]),
                          usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=10,
                                                                                     candidates_token_count=2))


class TestCallCap(unittest.TestCase):

    def test_run_stops_cleanly_at_the_model_call_cap(self):
        db.init()
        fake = LoopingFakeLlm()
        agent = attach_callbacks(LlmAgent(name="claim_manager", model=fake, instruction="Decide claim {bb_claim}.",
                                          tools=[submit_report]))
        with mock.patch.object(config, "GOOGLE_API_KEY", "test"), \
                mock.patch("claimsleuth.agents.patterns.build", return_value=(agent, [])):
            result = asyncio.run(runner.run_claim("CLM-2024-001", "hierarchical", "direct"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("LlmCallsLimitExceeded", result["error"])
        self.assertIsNone(result["report"])                             # no report is made up
        self.assertEqual(fake.calls, config.MAX_LLM_CALLS)
        with db.connect() as conn:
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (result["run_id"],)).fetchone()
            rows = conn.execute("SELECT * FROM agent_metrics WHERE run_id = ?", (result["run_id"],)).fetchall()
        self.assertEqual(run["status"], "failed")
        self.assertEqual(rows[0]["llm_calls"], config.MAX_LLM_CALLS)
        self.assertEqual(rows[0]["prompt_tokens"], 10 * config.MAX_LLM_CALLS)
