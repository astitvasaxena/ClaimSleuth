"""
The three ways the same agents are wired together. Each factory builds fresh agent objects (ADK does not let
one agent have two parents) and returns the root agent plus the MCP toolsets to close after the run.
"""
from google.adk.agents import ParallelAgent, SequentialAgent

from claimsleuth.agents import evidence, fraud_remote, manager, policy, settlement
from claimsleuth.callbacks.hooks import attach_callbacks

PATTERNS = ("sequential", "parallel", "hierarchical")


def _specialists(style: str):
    ev, t1 = evidence.build(style)
    po, t2 = policy.build(style)
    se, t3 = settlement.build(style)
    return ev, po, se, fraud_remote.build(), t1 + t2 + t3


def build_sequential(style: str):
    ev, po, se, fr, toolsets = _specialists(style)
    root = SequentialAgent(name="sequential_investigation", sub_agents=[ev, po, fr, se, manager.build_decider(style)])
    return root, toolsets


def build_parallel(style: str):
    ev, po, se, fr, toolsets = _specialists(style)
    root = SequentialAgent(name="parallel_investigation", sub_agents=[
        ev, ParallelAgent(name="policy_and_fraud", sub_agents=[po, fr]), se, manager.build_decider(style)])
    return root, toolsets


def build_hierarchical(style: str):
    ev, po, se, fr, toolsets = _specialists(style)
    return manager.build_delegating(style, [ev, po, fr, se]), toolsets


def build(pattern: str, style: str):
    builders = {"sequential": build_sequential, "parallel": build_parallel, "hierarchical": build_hierarchical}
    if pattern not in builders:
        raise ValueError(f"unknown pattern {pattern!r}; use one of {PATTERNS}")
    root, toolsets = builders[pattern](style)
    return attach_callbacks(root), toolsets
