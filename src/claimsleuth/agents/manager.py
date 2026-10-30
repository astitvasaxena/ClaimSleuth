"""Claim Manager: in the hierarchical pattern it chooses which specialist to call; in the others it only decides."""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from claimsleuth.agents.common import planner, model, prompt, submit_report


def build_delegating(style: str, specialists: list) -> LlmAgent:
    return LlmAgent(name="claim_manager", model=model(), instruction=prompt(style, "manager"), planner=planner(style),
                    description="Leads the investigation and makes the final decision.",
                    tools=[AgentTool(agent) for agent in specialists] + [submit_report])


def build_decider(style: str) -> LlmAgent:
    return LlmAgent(name="claim_manager", model=model(), instruction=prompt(style, "manager_decide"), planner=planner(style),
                    description="Makes the final decision from the blackboard.", tools=[submit_report],
                    include_contents="none")
