"""Settlement agent: gets the payout from compute_payout when the claim is covered (bb_payout)."""
from google.adk.agents import LlmAgent

from claimsleuth.agents.common import content_config, planner, mcp_tools, model, prompt, record_payout

DESCRIPTION = "Computes the payout with compute_payout and records it. Needs a coverage decision that says covered."


def build(style: str) -> tuple[LlmAgent, list]:
    toolset = mcp_tools("compute_payout")
    agent = LlmAgent(name="settlement_agent", model=model(), generate_content_config=content_config(), description=DESCRIPTION,
                     instruction=prompt(style, "settlement"), planner=planner(style), tools=[toolset, record_payout], include_contents="none")
    return agent, [toolset]
