"""
C-4 agent patterns and C-1 messaging architectures, verified by real ADK execution.

Every combination of pattern (sequential / parallel / hierarchical) and messaging
(blackboard / direct / broadcast) must reach the same verified outcome for the three
benchmark claims; the patterns must differ in HOW they run (graph shape, concurrency,
delegation decisions, context delivered), which these tests also check.
"""
import importlib
import json
import os
import socket
import unittest
from pathlib import Path

from src.adk_app.patterns import (
    DIRECT_ROUTES, MAX_MANAGER_DECISIONS, manager_decision, merge_branch_outputs,
)
from src.adk_app.pipeline import run_adk_investigation
from src.adk_app.tool_gateway import McpConfig
from tests.test_adk_pipeline import ScriptedFakeLlm, investigator_script, load_claim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAULTY_MCP = str(PROJECT_ROOT / "tests" / "fixtures" / "faulty_mcp_server.py")
PATTERNS = ("sequential", "parallel", "hierarchical")
MESSAGING = ("blackboard", "direct", "broadcast")
EXPECTED = {"scenario_1_valid": ("VERIFIED_PENDING_APPROVAL", "PROPOSED_PENDING_APPROVAL", 251.0),
            "scenario_2_expired": ("REJECTED_DETERMINISTICALLY", "NOT_PAYABLE", None),
            "scenario_3_conflict": ("CONFLICT_ESCALATED", "NOT_PAYABLE", None)}


def outcome(report):
    return report.status, report.payout.payout_status, report.payout.proposed_payout


def node_names(result):
    return [e["node_path"].split("/")[-1].split("@")[0] for e in result.events if e["node_path"]]


class TestAllCombinationsReachTheVerifiedOutcome(unittest.TestCase):

    def test_27_combinations(self):
        for pattern in PATTERNS:
            for messaging in MESSAGING:
                for scenario, expected in EXPECTED.items():
                    with self.subTest(pattern=pattern, messaging=messaging, scenario=scenario):
                        report = run_adk_investigation(load_claim(scenario), pattern=pattern,
                                                       messaging=messaging).report
                        self.assertEqual(outcome(report), expected)
                        self.assertEqual((report.pattern, report.messaging), (pattern, messaging))
                        self.assertIsNone(report.payout.authorized_payout)
                        self.assertTrue(report.audit_chain_valid)


class TestParallelPattern(unittest.TestCase):

    def test_runs_as_adk_fan_out_join_and_merge(self):
        result = run_adk_investigation(load_claim("scenario_3_conflict"), pattern="parallel")
        nodes = node_names(result)
        self.assertIn("ParallelJoin", nodes)
        self.assertLess(nodes.index("ParallelJoin"), nodes.index("MergeAgent"))
        self.assertIn("MergeAgent", result.report.agent_sequence)
        self.assertIsNone(result.state["bb:branch_out:AnomalyAgent"])        # applied and cleared
        branches = {t["agent"] for t in result.report.agent_timings if t["branch"]}
        self.assertEqual(branches, {"AnomalyAgent", "WarrantyAgent"})
        self.assertTrue(result.report.conflicts)   # the anomaly branch's conflicts survived the merge

    def test_branches_cannot_see_each_other(self):
        seq = run_adk_investigation(load_claim("scenario_1_valid"), pattern="sequential").report
        par = run_adk_investigation(load_claim("scenario_1_valid"), pattern="parallel").report
        # Sequentially, WarrantyAgent runs after AnomalyAgent and sees bb:anomalies; in a branch it must not.
        self.assertEqual(par.messaging_summary["keys_visible_per_agent"]["WarrantyAgent"],
                         seq.messaging_summary["keys_visible_per_agent"]["WarrantyAgent"] - 1)

    def test_branches_overlap_in_time_when_waiting_on_mcp(self):
        report = run_adk_investigation(load_claim("scenario_1_valid"), pattern="parallel", tools="mcp").report
        span = {t["agent"]: (t["start_ms"], t["end_ms"]) for t in report.agent_timings}
        anomaly, warranty = span["AnomalyAgent"], span["WarrantyAgent"]
        self.assertLess(max(anomaly[0], warranty[0]), min(anomaly[1], warranty[1]),
                        f"branches did not overlap: {anomaly} vs {warranty}")
        self.assertEqual(outcome(report), EXPECTED["scenario_1_valid"])

    def test_merge_conflict_fails_closed(self):
        state = {"bb:conflicts": [], "bb:tool_errors": [], "bb:agent_sequence": ["IntakeAgent"],
                 "bb:branch_out:A": {"bb:eligibility": {"covered": True}},
                 "bb:branch_out:B": {"bb:eligibility": {"covered": False}}}
        merged = merge_branch_outputs(state, ["A", "B"])
        self.assertTrue(any("wrote different values" in e for e in merged["bb:tool_errors"]))
        self.assertEqual(merged["bb:eligibility"], {"covered": True})
        missing = merge_branch_outputs({"bb:tool_errors": []}, ["A"])
        self.assertIn("MergeAgent: branch A produced no output", missing["bb:tool_errors"])


class TestHierarchicalPattern(unittest.TestCase):

    def decisions(self, scenario):
        result = run_adk_investigation(load_claim(scenario), pattern="hierarchical")
        return result, [d["route"] for d in result.report.manager_decisions]

    def test_manager_takes_different_paths_for_different_claims(self):
        r1, d1 = self.decisions("scenario_1_valid")
        r2, d2 = self.decisions("scenario_2_expired")
        r3, d3 = self.decisions("scenario_3_conflict")
        self.assertEqual(d1, ["evidence", "anomaly", "warranty", "settlement", "verify"])
        self.assertEqual(d2, ["evidence", "anomaly", "warranty", "verify"])            # ineligible: no settlement
        self.assertEqual(d3, ["evidence", "anomaly", "verify"])                        # conflict: escalate early
        self.assertNotIn("SettlementAgent", r2.report.agent_sequence)
        self.assertNotIn("WarrantyAgent", r3.report.agent_sequence)
        self.assertIn("conflict", r3.report.manager_decisions[-1]["reason"])
        # The manager node was executed three separate times (ADK numbers executions ClaimManager@1, @2, ...)
        executions = {e["node_path"].split("/")[-1] for e in r3.events
                      if e["node_path"] and "/ClaimManager@" in e["node_path"]}
        self.assertEqual(executions, {"ClaimManager@1", "ClaimManager@2", "ClaimManager@3"})
        self.assertLess(len(r3.report.tool_calls), len(r1.report.tool_calls))

    def test_manager_policy_units(self):
        self.assertEqual(manager_decision({"bb:documents": []}, False, [])[0], "verify")
        self.assertEqual(manager_decision({"bb:documents": ["a"]}, False, [{}] * MAX_MANAGER_DECISIONS)[0], "verify")
        view = {"bb:documents": ["a"], "bb:agent_sequence": ["IntakeAgent", "EvidenceAgent", "AnomalyAgent"]}
        self.assertEqual(manager_decision(view, True, [])[0], "forensics")
        self.assertEqual(manager_decision(view, False, [])[0], "warranty")

    def test_model_mode_inside_hierarchy(self):
        model = ScriptedFakeLlm(model="fake", script=investigator_script("invoice_001.txt", "repair_estimate_001.txt"))
        report = run_adk_investigation(load_claim("scenario_1_valid"), pattern="hierarchical", model=model).report
        self.assertIn("InvestigatorLLM", report.agent_sequence)
        self.assertEqual(outcome(report), EXPECTED["scenario_1_valid"])


class TestMessagingArchitectures(unittest.TestCase):

    def summary(self, messaging, pattern="sequential", **kw):
        return run_adk_investigation(load_claim("scenario_1_valid"), pattern=pattern, messaging=messaging, **kw).report

    def test_delivery_counts_follow_the_architecture(self):
        bb, direct, bc = (self.summary(m).messaging_summary for m in MESSAGING)
        self.assertEqual((bb["messages_sent"], bb["deliveries"]), (0, 0))
        self.assertEqual(direct["messages_sent"], 6)
        self.assertEqual(bc["deliveries"], sum(range(7)))          # each agent gets every earlier message
        expected_direct = sum(1 for producer in ("IntakeAgent", "EvidenceAgent", "AnomalyAgent", "WarrantyAgent",
                                                 "SettlementAgent", "VerifierAgent")
                              for consumer in DIRECT_ROUTES[producer]
                              if consumer in ("EvidenceAgent", "AnomalyAgent", "WarrantyAgent", "SettlementAgent",
                                              "VerifierAgent", "ReportAgent"))
        self.assertEqual(direct["deliveries"], expected_direct)
        self.assertLess(direct["deliveries"], bc["deliveries"])
        self.assertLess(direct["context_bytes_per_agent"], bc["context_bytes_per_agent"])

    def test_direct_mode_dropped_route_fails_closed(self):
        report = self.summary("direct", drop_routes={("EvidenceAgent", "SettlementAgent")})
        self.assertIsNone(report.settlement_proposal)          # SettlementAgent never received repair_cost
        self.assertEqual(report.status, "NEEDS_EVIDENCE")
        self.assertEqual(report.payout.payout_status, "NOT_PAYABLE")
        for messaging in ("blackboard", "broadcast"):           # not affected: no routing to get wrong
            self.assertEqual(outcome(self.summary(messaging)), EXPECTED["scenario_1_valid"])

    def test_messages_are_real_session_state(self):
        result = run_adk_investigation(load_claim("scenario_1_valid"), messaging="direct")
        first = result.state["bb:messages"][0]
        self.assertEqual(first["from"], "IntakeAgent")
        self.assertIn("bb:policy", first["payload"])
        self.assertEqual(first["to"], DIRECT_ROUTES["IntakeAgent"])

    def test_deprecated_sequential_backend_supports_messaging(self):
        report = run_adk_investigation(load_claim("scenario_3_conflict"), backend="sequential",
                                       messaging="broadcast").report
        self.assertEqual(outcome(report), EXPECTED["scenario_3_conflict"])
        self.assertEqual(report.messaging_summary["messages_sent"], 6)


class TestValidationAndFailuresPerPattern(unittest.TestCase):

    def test_invalid_combinations_rejected(self):
        with self.assertRaises(ValueError):
            run_adk_investigation(load_claim("scenario_1_valid"), backend="sequential", pattern="parallel")
        with self.assertRaises(ValueError):
            run_adk_investigation(load_claim("scenario_1_valid"), pattern="star")
        model = ScriptedFakeLlm(model="fake", script=investigator_script("invoice_001.txt", "repair_estimate_001.txt"))
        with self.assertRaises(ValueError):
            run_adk_investigation(load_claim("scenario_1_valid"), model=model, messaging="direct",
                                  fallback_on_model_error=False)

    def test_mcp_tool_failures_fail_closed_in_every_pattern(self):
        for pattern in PATTERNS:
            with self.subTest(pattern=pattern):
                err = run_adk_investigation(load_claim("scenario_1_valid"), pattern=pattern, tools="mcp",
                                            mcp=McpConfig(server_args=[FAULTY_MCP, "--fault", "settlement_error"])).report
                self.assertEqual((err.status, err.payout.payout_status), ("NEEDS_EVIDENCE", "NOT_PAYABLE"))
                tampered = run_adk_investigation(load_claim("scenario_1_valid"), pattern=pattern, tools="mcp",
                                                 mcp=McpConfig(server_args=[FAULTY_MCP, "--fault", "tamper_settlement"])).report
                self.assertEqual((tampered.status, tampered.payout.payout_status), ("CONFLICT_ESCALATED", "NOT_PAYABLE"))

    def test_crewai_service_down_fails_closed_in_every_pattern(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        for pattern in PATTERNS:
            with self.subTest(pattern=pattern):
                report = run_adk_investigation(load_claim("scenario_1_valid"), pattern=pattern, forensics="crewai_a2a",
                                               a2a_url=f"http://127.0.0.1:{port}", a2a_timeout=5).report
                self.assertEqual(report.status, "NEEDS_EVIDENCE")
                self.assertIn("CREW_SERVICE_UNAVAILABLE", report.risk_flags)
                self.assertIn("CrewAIForensics", report.agent_sequence)

    def test_endpoint_accepts_pattern_and_messaging(self):
        from fastapi.testclient import TestClient
        os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        import configs.settings
        importlib.reload(configs.settings)
        import src.ui.app as app_module
        client = TestClient(importlib.reload(app_module).app)
        data = client.post("/api/adk/investigate/scenario_3_conflict?pattern=hierarchical&messaging=direct").json()
        self.assertEqual(data["report"]["pattern"], "hierarchical")
        self.assertEqual([d["route"] for d in data["report"]["manager_decisions"]], ["evidence", "anomaly", "verify"])
        self.assertEqual(client.post("/api/adk/investigate/scenario_1_valid?backend=sequential&pattern=parallel")
                         .status_code, 422)
        self.assertEqual(client.post("/api/adjudicate/CLM-2024-003", json={"decision": "APPROVED"}).status_code, 409)


if __name__ == "__main__":
    unittest.main()
