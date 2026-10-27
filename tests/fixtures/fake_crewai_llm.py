"""
TEST DOUBLE for CrewAI: a crewai BaseLLM that returns a fixed answer without any
network call. It exercises the real Crew/Agent/Task execution path; it is NOT a
live model. Selected in the forensics service only by CLAIMSLEUTH_CREWAI_LLM=fake.
"""
from typing import Any

from crewai.llms.base_llm import BaseLLM


CALLS = {"count": 0}   # module-level: BaseLLM is a pydantic model, so class attributes become fields


class FakeReviewerLLM(BaseLLM):

    def __init__(self, **kwargs: Any):
        super().__init__(model="fake-reviewer", **kwargs)

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None, **kwargs) -> str:
        CALLS["count"] += 1
        return ("Thought: I can summarise the findings.\n"
                "Final Answer: Deterministic forensic checks were run on the dossier. "
                "Any FLAG result needs a human reviewer; this summary makes no payout decision.")

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192
