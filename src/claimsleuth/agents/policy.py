"""Policy agent: finds the policy wording, checks the warranty window and exclusions (bb_coverage)."""
from google.adk.agents import LlmAgent

from claimsleuth.agents.common import content_config, planner, mcp_tools, model, prompt, record_coverage

DESCRIPTION = ("Searches the policy wording and checks the warranty window and exclusions; records the coverage "
               "decision with the deciding clause and the policy terms. Needs the facts.")


def build(style: str) -> tuple[LlmAgent, list]:
    toolset = mcp_tools("search_policy", "check_coverage")
    agent = LlmAgent(name="policy_agent", model=model(), generate_content_config=content_config(), description=DESCRIPTION, instruction=prompt(style, "policy"), planner=planner(style),
                     tools=[toolset, record_coverage], include_contents="none")
    return agent, [toolset]
