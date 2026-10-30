"""Evidence agent: reads every document and records quoted facts (bb_facts)."""
from google.adk.agents import LlmAgent

from claimsleuth.agents.common import planner, mcp_tools, model, prompt, record_facts

DESCRIPTION = "Reads every claim document with parse_document and records the facts, each with an exact quote."


def build(style: str) -> tuple[LlmAgent, list]:
    toolset = mcp_tools("parse_document")
    agent = LlmAgent(name="evidence_agent", model=model(), description=DESCRIPTION, instruction=prompt(style, "evidence"), planner=planner(style),
                     tools=[toolset, record_facts], include_contents="none")
    return agent, [toolset]
