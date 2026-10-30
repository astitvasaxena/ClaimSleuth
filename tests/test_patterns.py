"""The three pattern factories build fresh agent trees for both prompt styles (no model calls)."""
import unittest

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.planners import PlanReActPlanner
from google.adk.tools.agent_tool import AgentTool

from claimsleuth.agents import patterns


def all_agents(agent):
    yield agent
    for sub in getattr(agent, "sub_agents", []) or []:
        yield from all_agents(sub)
    for tool in getattr(agent, "tools", []) or []:
        if isinstance(tool, AgentTool):
            yield from all_agents(tool.agent)


class TestPatterns(unittest.TestCase):

    def test_every_pattern_and_style_builds(self):
        for pattern in patterns.PATTERNS:
            for style in ("direct", "react"):
                with self.subTest(pattern=pattern, style=style):
                    root, toolsets = patterns.build(pattern, style)
                    names = {a.name for a in all_agents(root)}
                    self.assertTrue({"evidence_agent", "policy_agent", "settlement_agent", "fraud_crew",
                                     "claim_manager"} <= names)
                    self.assertEqual(len(toolsets), 3)
                    for agent in all_agents(root):
                        if isinstance(agent, LlmAgent):
                            self.assertEqual(isinstance(agent.planner, PlanReActPlanner), style == "react")

    def test_shapes(self):
        seq, _ = patterns.build("sequential", "direct")
        self.assertIsInstance(seq, SequentialAgent)
        self.assertEqual([a.name for a in seq.sub_agents],
                         ["evidence_agent", "policy_agent", "fraud_crew", "settlement_agent", "claim_manager"])
        par, _ = patterns.build("parallel", "direct")
        self.assertIsInstance(par.sub_agents[1], ParallelAgent)
        self.assertEqual({a.name for a in par.sub_agents[1].sub_agents}, {"policy_agent", "fraud_crew"})
        hier, _ = patterns.build("hierarchical", "direct")
        self.assertEqual(hier.name, "claim_manager")
        delegated = [t.agent.name for t in hier.tools if isinstance(t, AgentTool)]
        self.assertEqual(delegated, ["evidence_agent", "policy_agent", "fraud_crew", "settlement_agent"])
        self.assertIsInstance(next(t.agent for t in hier.tools if isinstance(t, AgentTool) and t.agent.name == "fraud_crew"),
                              RemoteA2aAgent)

    def test_factories_build_fresh_agents(self):
        a, _ = patterns.build("hierarchical", "direct")
        b, _ = patterns.build("hierarchical", "direct")
        self.assertIsNot(a, b)
        self.assertTrue(all(x.agent is not y.agent for x, y in zip(a.tools, b.tools) if isinstance(x, AgentTool)))


class TestCallbacksAttached(unittest.TestCase):

    def test_every_agent_has_the_callbacks(self):
        from claimsleuth.callbacks.hooks import has_callbacks
        for pattern in patterns.PATTERNS:
            root, _ = patterns.build(pattern, "direct")
            for agent in all_agents(root):
                with self.subTest(pattern=pattern, agent=agent.name):
                    self.assertTrue(has_callbacks(agent))
