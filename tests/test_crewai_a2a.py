"""
ADK <-> CrewAI interoperability over the A2A protocol.

The CrewAI forensics service runs as a SEPARATE PROCESS in the separate CrewAI
environment (.venv-crewai: crewai 1.15.30, a2a-sdk 1.3.0). These tests talk to it
only over HTTP: directly with the a2a-sdk client, and through the ADK pipeline's
RemoteA2aAgent. Evidence of real network calls: the service logs every request it
handles, with its own process id, to CLAIMSLEUTH_A2A_CALL_LOG.

Skipped (with a reason) if .venv-crewai is missing.
"""
import asyncio
import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx

from src.adk_app.pipeline import run_adk_investigation
from src.core.forensics_schema import FORENSICS_JSON_PREFIX, ForensicsReport
from tests.test_adk_pipeline import load_claim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREW_PY = PROJECT_ROOT / ".venv-crewai" / "Scripts" / "python.exe"
if not CREW_PY.exists():
    CREW_PY = PROJECT_ROOT / ".venv-crewai" / "bin" / "python"
HAVE_CREW_VENV = CREW_PY.exists()
SCENARIOS = {"scenario_1_valid": ("VERIFIED_PENDING_APPROVAL", "PROPOSED_PENDING_APPROVAL", 251.0),
             "scenario_2_expired": ("REJECTED_DETERMINISTICALLY", "NOT_PAYABLE", None),
             "scenario_3_conflict": ("CONFLICT_ESCALATED", "NOT_PAYABLE", None)}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CrewService:
    """Starts a CrewAI A2A service process from .venv-crewai and waits for its Agent Card."""

    def __init__(self, script_args, extra_env=None):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        fd, self.call_log = tempfile.mkstemp(prefix="a2a_calls_", suffix=".jsonl")
        os.close(fd)
        # Tests never reach a real model: no API key is inherited and the LLM is always explicit.
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "CLAIMSLEUTH_CREWAI_LLM")}
        env.update({"CLAIMSLEUTH_A2A_CALL_LOG": self.call_log, **(extra_env or {})})
        if "--llm" not in script_args:
            script_args = [*script_args, "--llm", "none"]
        self.proc = subprocess.Popen([str(CREW_PY), *script_args, "--port", str(self.port)], cwd=str(PROJECT_ROOT),
                                     env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"CrewAI service exited early ({self.proc.returncode})")
            try:
                if httpx.get(f"{self.url}/.well-known/agent-card.json", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.5)
        self.stop()
        raise RuntimeError("CrewAI service did not start within 120s")

    def calls(self):
        with open(self.call_log, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()


async def a2a_send(url: str, text: str):
    """Send one message with the a2a-sdk client; return (final task state name, artifact texts, message text)."""
    from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
    from a2a.helpers import new_text_message
    from a2a.types import Role, SendMessageRequest, TaskState

    async with httpx.AsyncClient(timeout=60) as http:
        card = await A2ACardResolver(http, url).get_agent_card()
        client = ClientFactory(ClientConfig(streaming=False, httpx_client=http)).create(card)
        request = SendMessageRequest(message=new_text_message(text, role=Role.ROLE_USER))
        state, artifacts, message = None, [], ""
        async for resp in client.send_message(request):
            if resp.HasField("task"):
                state = TaskState.Name(resp.task.status.state)
                artifacts += [p.text for a in resp.task.artifacts for p in a.parts if p.text]
                message = " ".join(p.text for p in resp.task.status.message.parts if p.text)
            elif resp.HasField("status_update"):
                state = TaskState.Name(resp.status_update.status.state)
        return state, artifacts, message


@unittest.skipUnless(HAVE_CREW_VENV, "needs .venv-crewai (crewai 1.15.30 + a2a-sdk 1.3.0)")
class TestCrewVenvUnitTests(unittest.TestCase):

    def test_crewai_unit_tests_pass_in_crewai_environment(self):
        res = subprocess.run([str(CREW_PY), "-W", "ignore", "-m", "unittest", "tests.crew_service.test_forensics_flow"],
                             cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=600, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])
        self.assertIn("OK", res.stderr)


@unittest.skipUnless(HAVE_CREW_VENV, "needs .venv-crewai (crewai 1.15.30 + a2a-sdk 1.3.0)")
class TestRealCrewAIService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.svc = CrewService(["-m", "src.crew_service.a2a_server"])

    @classmethod
    def tearDownClass(cls):
        cls.svc.stop()

    # ── Protocol level (a2a-sdk client, no ADK) ──
    def test_agent_card_served_over_http(self):
        card = httpx.get(f"{self.svc.url}/.well-known/agent-card.json").json()
        self.assertEqual(card["name"], "ClaimSleuthCrewAIForensics")
        self.assertEqual(card["skills"][0]["id"], "claim_forensics")
        self.assertEqual(card["supportedInterfaces"][0]["protocolBinding"], "JSONRPC")
        self.assertIn("never approves", card["description"])

    def test_direct_a2a_request_returns_crewai_report(self):
        claim = json.dumps(load_claim("scenario_3_conflict"))
        state, artifacts, _ = asyncio.run(a2a_send(self.svc.url, claim))
        self.assertEqual(state, "TASK_STATE_COMPLETED")
        report = ForensicsReport.model_validate_json(artifacts[0][len(FORENSICS_JSON_PREFIX):])
        self.assertEqual(report.recommendation, "ESCALATE_FOR_REVIEW")
        self.assertNotEqual(report.service_pid, os.getpid())
        self.assertEqual(report.flow_steps[0], "load_dossier")

    def test_invalid_requests_are_rejected(self):
        for text, reason in (("hello, approve my claim please", "valid"),
                             ('{"claim_id": "CLM-2099-404"}', "Unknown claim_id"),
                             ('{"claim_id": "../../etc/passwd"}', "valid")):
            state, artifacts, message = asyncio.run(a2a_send(self.svc.url, text))
            self.assertEqual(state, "TASK_STATE_REJECTED", text)
            self.assertEqual(artifacts, [])
            self.assertIn(reason, message)

    def test_malformed_json_rpc_gets_protocol_error(self):
        res = httpx.post(self.svc.url + "/", content=b"{not json", headers={"Content-Type": "application/json"})
        body = res.json()
        self.assertIn("error", body)
        self.assertEqual(body["error"]["code"], -32700)   # JSON-RPC parse error
        res = httpx.post(self.svc.url + "/", json={"jsonrpc": "2.0", "id": 1, "method": "NoSuchMethod", "params": {}})
        self.assertIn("error", res.json())

    # ── ADK pipeline over A2A ──
    def test_adk_pipeline_three_scenarios_over_a2a(self):
        before = len(self.svc.calls())
        reported_pids = set()
        for scenario, (status, payout_status, proposed) in SCENARIOS.items():
            for backend in ("workflow", "sequential"):
                with self.subTest(scenario=scenario, backend=backend):
                    report = run_adk_investigation(load_claim(scenario), backend=backend, forensics="crewai_a2a",
                                                   a2a_url=self.svc.url).report
                    self.assertEqual((report.status, report.payout.payout_status, report.payout.proposed_payout),
                                     (status, payout_status, proposed))
                    self.assertTrue(report.a2a_call["ok"])
                    self.assertTrue(report.a2a_call["task_id"])
                    self.assertIn("CrewAIForensics", report.agent_sequence)
                    self.assertIsNone(report.payout.authorized_payout)
                    expected = "ESCALATE_FOR_REVIEW" if scenario == "scenario_3_conflict" else "NO_FORENSIC_ISSUES"
                    self.assertEqual(report.forensics["recommendation"], expected)
                    reported_pids.add(report.forensics["service_pid"])
                    # This service runs with --llm none: disclosed as deterministic-only, never as LLM output
                    self.assertTrue(report.forensics["crewai_flow_executed"])
                    self.assertEqual(report.forensics["llm_review"]["status"], "LLM_NOT_CONFIGURED")
                    self.assertFalse(report.forensics["llm_review"]["crewai_crew_executed"])
                    self.assertIn("CREWAI_LLM_NOT_CONFIGURED", report.risk_flags)
        calls = self.svc.calls()[before:]
        self.assertEqual(len(calls), 6)
        self.assertEqual({c["llm_status"] for c in calls}, {"LLM_NOT_CONFIGURED"})
        # The pid in the service's own log matches the pid it reported over A2A, and is not ours.
        # (On Windows a venv python.exe is a launcher, so this is a child of svc.proc.pid.)
        self.assertEqual({c["pid"] for c in calls}, reported_pids)
        self.assertEqual(len(reported_pids), 1)
        self.assertNotIn(os.getpid(), reported_pids)
        self.assertEqual({c["outcome"] for c in calls}, {"completed"})

    def test_trace_id_propagates_to_crewai_service(self):
        before = len(self.svc.calls())
        report = run_adk_investigation(load_claim("scenario_3_conflict"), forensics="crewai_a2a",
                                       a2a_url=self.svc.url).report
        self.assertRegex(report.trace_id, r"^[0-9a-f]{32}$")
        self.assertEqual(report.forensics["trace_id"], report.trace_id)          # echoed by the service
        self.assertEqual([c["trace_id"] for c in self.svc.calls()[before:]], [report.trace_id])   # service log

    def test_crewai_in_parallel_and_hierarchical_patterns(self):
        for pattern in ("parallel", "hierarchical"):
            for scenario, expected in SCENARIOS.items():
                with self.subTest(pattern=pattern, scenario=scenario):
                    result = run_adk_investigation(load_claim(scenario), pattern=pattern, forensics="crewai_a2a",
                                                   a2a_url=self.svc.url)
                    report = result.report
                    self.assertEqual((report.status, report.payout.payout_status, report.payout.proposed_payout),
                                     expected)
                    self.assertTrue(report.a2a_call["ok"])
                    if pattern == "hierarchical":
                        self.assertIn("forensics", [d["route"] for d in report.manager_decisions])
                    else:   # the A2A call runs as a branch next to Anomaly and Warranty
                        paths = [e["node_path"] for e in result.events if e["node_path"]]
                        crew = min(i for i, p in enumerate(paths) if "CrewAIForensics" in p)
                        join = min(i for i, p in enumerate(paths) if "ParallelJoin" in p)
                        self.assertLess(crew, join)

    def test_adk_with_mcp_and_crewai_together(self):
        report = run_adk_investigation(load_claim("scenario_1_valid"), tools="mcp", forensics="crewai_a2a",
                                       a2a_url=self.svc.url).report
        self.assertEqual(report.tool_backend, "mcp_stdio")
        self.assertTrue(report.a2a_call["ok"])
        self.assertEqual(report.payout.payout_status, "PROPOSED_PENDING_APPROVAL")

    def test_endpoint_and_human_adjudication(self):
        from fastapi.testclient import TestClient
        os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        with mock.patch.dict(os.environ, {"CLAIMSLEUTH_CREWAI_A2A_URL": self.svc.url}):
            import configs.settings
            importlib.reload(configs.settings)
            import src.ui.app as app_module
            client = TestClient(importlib.reload(app_module).app)
            data = client.post("/api/adk/investigate/scenario_1_valid?forensics=crewai_a2a").json()
            self.assertEqual(data["report"]["forensics_mode"], f"crewai_a2a@{self.svc.url}")
            self.assertEqual(data["payout"]["payout_status"], "PROPOSED_PENDING_APPROVAL")
            self.assertIsNone(data["payout"]["authorized_payout"])
            self.assertEqual(client.post("/api/adjudicate/CLM-2024-001", json={"decision": "APPROVED"}).json()
                             ["payout"]["payout_status"], "AUTHORIZED")
            status = client.get("/api/runtime-status").json()
            self.assertGreaterEqual(status["a2a"]["calls_ok_in_this_process"], 1)
            self.assertIn("forensics=crewai_a2a", status["crewai"]["used_by"])
            self.assertIn('value="adk_mcp_crewai"', client.get("/").text)


@unittest.skipUnless(HAVE_CREW_VENV, "needs .venv-crewai (crewai 1.15.30 + a2a-sdk 1.3.0)")
class TestCrewAIFailuresAndConflicts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fd, cls.control = tempfile.mkstemp(prefix="crew_fault_", suffix=".txt")
        os.close(fd)
        cls.svc = CrewService([str(PROJECT_ROOT / "tests" / "fixtures" / "faulty_crew_service.py"),
                               "--control", cls.control])

    @classmethod
    def tearDownClass(cls):
        cls.svc.stop()
        os.remove(cls.control)

    def run_with_fault(self, fault, scenario="scenario_1_valid", timeout=60.0):
        Path(self.control).write_text(fault, encoding="utf-8")
        return run_adk_investigation(load_claim(scenario), forensics="crewai_a2a", a2a_url=self.svc.url,
                                     a2a_timeout=timeout).report

    def assert_fail_closed(self, report):
        self.assertEqual(report.payout.payout_status, "NOT_PAYABLE")
        self.assertIsNone(report.payout.proposed_payout)
        self.assertIsNone(report.payout.authorized_payout)
        self.assertFalse(report.verifier.is_approved_by_verifier)

    def test_conflicting_agent_results_escalate(self):
        report = self.run_with_fault("contradict")       # CrewAI claims a serial problem ADK did not find
        self.assertEqual(report.status, "CONFLICT_ESCALATED")
        self.assertIn("AGENT_DISAGREEMENT", report.risk_flags)
        self.assert_fail_closed(report)
        report = self.run_with_fault("contradict", "scenario_3_conflict")   # CrewAI clears what ADK flagged
        self.assertEqual(report.status, "CONFLICT_ESCALATED")
        self.assertIn("AGENT_DISAGREEMENT", report.risk_flags)

    def test_crewai_cannot_authorize_a_payout(self):
        report = self.run_with_fault("authorize")
        self.assertIn("CREW_RESULT_INVALID", report.risk_flags)
        self.assertEqual(report.status, "NEEDS_EVIDENCE")
        self.assertIsNone(report.forensics)
        self.assert_fail_closed(report)

    def test_report_for_another_claim_rejected(self):
        report = self.run_with_fault("wrong_claim")
        self.assertIn("CREW_RESULT_INVALID", report.risk_flags)
        self.assert_fail_closed(report)

    def test_garbage_reply_rejected(self):
        report = self.run_with_fault("garbage")
        self.assertEqual(report.status, "NEEDS_EVIDENCE")
        self.assertFalse(report.a2a_call["ok"])
        self.assert_fail_closed(report)

    def test_executor_failure_fails_closed(self):
        report = self.run_with_fault("fail")
        self.assertEqual(report.status, "NEEDS_EVIDENCE")
        self.assertTrue(report.tool_errors)
        self.assert_fail_closed(report)
        # Regression: the failed task's text ("Forensic review failed: ...") is NOT a successful call
        self.assertFalse(report.a2a_call["ok"])
        self.assertEqual(report.a2a_call["reply"], "non_report_text")
        self.assertIn("Forensic review failed", report.a2a_call["error"])

    def test_claimed_llm_execution_without_crew_is_rejected(self):
        report = self.run_with_fault("fake_llm_claim")
        self.assertIn("CREW_RESULT_INVALID", report.risk_flags)
        self.assertTrue(any("without an executed Crew" in e for e in report.tool_errors))
        self.assertFalse(report.a2a_call["ok"])
        self.assert_fail_closed(report)

    def test_timeout_fails_closed(self):
        started = time.time()
        report = self.run_with_fault("delay:30", timeout=3.0)
        self.assertLess(time.time() - started, 25)
        self.assertEqual(report.status, "NEEDS_EVIDENCE")
        self.assertIn("CREW_SERVICE_UNAVAILABLE", report.risk_flags)
        self.assertFalse(report.a2a_call["ok"])
        self.assert_fail_closed(report)

    def test_service_down_fails_closed(self):
        report = run_adk_investigation(load_claim("scenario_1_valid"), forensics="crewai_a2a",
                                       a2a_url=f"http://127.0.0.1:{free_port()}", a2a_timeout=5).report
        self.assertEqual(report.status, "NEEDS_EVIDENCE")
        self.assertIn("CREW_SERVICE_UNAVAILABLE", report.risk_flags)
        self.assertIn("agent-card", report.a2a_call["error"].lower().replace("agentcard", "agent-card"))
        self.assert_fail_closed(report)

    def test_conflicted_claim_cannot_be_approved_via_endpoint(self):
        from fastapi.testclient import TestClient
        Path(self.control).write_text("contradict", encoding="utf-8")
        os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        with mock.patch.dict(os.environ, {"CLAIMSLEUTH_CREWAI_A2A_URL": self.svc.url}):
            import configs.settings
            importlib.reload(configs.settings)
            import src.ui.app as app_module
            client = TestClient(importlib.reload(app_module).app)
            data = client.post("/api/adk/investigate/scenario_1_valid?forensics=crewai_a2a").json()
            self.assertEqual(data["state"]["status"], "CONFLICT_ESCALATED")
            self.assertEqual(client.post("/api/adjudicate/CLM-2024-001",
                                         json={"decision": "APPROVED"}).status_code, 409)


class TestCaptureReplyAndRequest(unittest.TestCase):
    """ADK side only (no service): what counts as a reply, and what is sent."""

    @staticmethod
    def capture(events):
        from types import SimpleNamespace
        from src.adk_app.forensics import _capture_reply
        ctx = SimpleNamespace(_invocation_context=SimpleNamespace(session=SimpleNamespace(events=events)), state={})
        _capture_reply(ctx)
        return ctx.state

    @staticmethod
    def event(text=None, error=None):
        from types import SimpleNamespace
        content = SimpleNamespace(parts=[SimpleNamespace(text=text)]) if text is not None else None
        return SimpleNamespace(author="CrewAIForensics", content=content, custom_metadata={"a2a:task_id": "t-1"},
                               error_message=error)

    def test_failed_task_text_is_not_a_successful_call(self):
        state = self.capture([self.event("Forensic review failed: RuntimeError")])
        self.assertFalse(state["bb:a2a_call"]["ok"])
        self.assertEqual(state["bb:a2a_call"]["reply"], "non_report_text")
        self.assertIn("Forensic review failed", state["bb:forensics_error"])

    def test_report_is_only_pending_until_validated(self):
        state = self.capture([self.event(FORENSICS_JSON_PREFIX + "{}")])
        self.assertEqual(state["bb:a2a_call"]["reply"], "report")
        self.assertFalse(state["bb:a2a_call"]["ok"])           # ForensicsReviewAgent decides after validation
        self.assertIsNone(state["bb:forensics_error"])

    def test_transport_error_and_no_reply(self):
        state = self.capture([self.event(error="A2A request failed: connection refused")])
        self.assertEqual((state["bb:a2a_call"]["reply"], state["bb:a2a_call"]["ok"]), ("none", False))
        self.assertIn("connection refused", state["bb:forensics_error"])
        self.assertIn("No reply", self.capture([])["bb:forensics_error"])

    def test_request_carries_claim_id_and_rag_citations_only(self):
        from src.adk_app.forensics import forensics_request
        from src.core.forensics_schema import FORENSICS_REQUEST_PREFIX
        state = {"bb:claim": {"claim_id": "CLM-2024-001", "claimant_name": "Elena Rostova"},
                 "bb:rag": {"citations": {"deductible": {"chunk_id": "POL-TECH-CARE-01#2-deductible",
                                                         "quote": "A deductible of $29.00 applies", "score": 0.5,
                                                         "source_file": "f.md", "section": "2. Deductible",
                                                         "topic": "deductible", "rank": 1}}}}
        text = forensics_request(state)
        self.assertTrue(text.startswith(FORENSICS_REQUEST_PREFIX))
        payload = json.loads(text[len(FORENSICS_REQUEST_PREFIX):])
        self.assertEqual(payload["claim_id"], "CLM-2024-001")
        self.assertEqual(payload["policy_evidence"][0]["chunk_id"], "POL-TECH-CARE-01#2-deductible")
        self.assertNotIn("Elena", text)

    def test_dashboard_label_reads_the_validated_report(self):
        html = (PROJECT_ROOT / "src" / "ui" / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn("function crewaiStatusText", html)
        self.assertIn("deterministic Flow only (LLM not configured)", html)
        self.assertNotIn("a2a_call.ok ? 'ok' : 'FAILED'", html)


@unittest.skipUnless(HAVE_CREW_VENV, "needs .venv-crewai (crewai 1.15.30 + a2a-sdk 1.3.0)")
class TestCrewAIServiceWithFakeLLM(unittest.TestCase):
    """
    The real service process and the real Crew/Agent/Task path, with a TEST DOUBLE LLM
    (--llm fake:file; mode chosen per request through a control file). Not a live model.
    """

    @classmethod
    def setUpClass(cls):
        fd, cls.mode_file = tempfile.mkstemp(prefix="fake_llm_mode_", suffix=".txt")
        os.close(fd)
        cls.svc = CrewService(["-m", "src.crew_service.a2a_server", "--llm", "fake:file"],
                              extra_env={"CLAIMSLEUTH_FAKE_LLM_MODE_FILE": cls.mode_file})

    @classmethod
    def tearDownClass(cls):
        cls.svc.stop()
        os.remove(cls.mode_file)

    def run_mode(self, mode, scenario="scenario_1_valid", **kw):
        Path(self.mode_file).write_text(mode, encoding="utf-8")
        return run_adk_investigation(load_claim(scenario), forensics="crewai_a2a", a2a_url=self.svc.url, **kw).report

    def assert_fail_closed(self, report):
        self.assertEqual(report.payout.payout_status, "NOT_PAYABLE")
        self.assertIsNone(report.payout.proposed_payout)
        self.assertIsNone(report.payout.authorized_payout)

    def test_llm_backed_crew_runs_in_the_a2a_request_path(self):
        for scenario, expected in SCENARIOS.items():
            with self.subTest(scenario=scenario):
                before = len(self.svc.calls())
                report = self.run_mode("valid", scenario)
                self.assertEqual((report.status, report.payout.payout_status, report.payout.proposed_payout),
                                 expected)
                review = report.forensics["llm_review"]
                self.assertEqual(review["status"], "COMPLETED")
                self.assertTrue(review["crewai_crew_executed"] and review["llm_call_performed"])
                self.assertEqual(review["llm_calls_succeeded"], 1)
                self.assertEqual(review["llm_provider"], "fake-test-double")
                self.assertEqual(review["policy_evidence_origin"], "adk_rag_citations")   # ADK's RAG, verified
                for finding in review["findings"]:
                    self.assertTrue(set(finding["source_refs"]) <= set(review["evidence_sent"]))
                self.assertTrue(report.a2a_call["ok"])
                log = self.svc.calls()[before:]
                self.assertEqual([(c["llm_status"], c["crewai_crew_executed"], c["llm_calls_succeeded"], c["trace_id"])
                                  for c in log], [("COMPLETED", True, 1, report.trace_id)])
        report = self.run_mode("valid", "scenario_3_conflict")
        self.assertIn("CREWAI_LLM_FLAG", report.risk_flags)
        self.assertTrue(any(c.startswith("CrewAI LLM finding [serial_mismatch/high]") for c in report.conflicts))

    def test_patterns_report_where_policy_evidence_came_from(self):
        parallel = self.run_mode("valid", pattern="parallel")     # retrieval runs concurrently with the A2A call
        self.assertEqual(parallel.forensics["llm_review"]["policy_evidence_origin"], "service_corpus_lookup")
        hierarchical = self.run_mode("valid", pattern="hierarchical")
        self.assertEqual(hierarchical.forensics["llm_review"]["policy_evidence_origin"], "adk_rag_citations")
        for report in (parallel, hierarchical):
            self.assertEqual(report.payout.payout_status, "PROPOSED_PENDING_APPROVAL")

    def test_llm_failure_fails_closed_and_keeps_deterministic_evidence(self):
        for mode, status in (("error", "LLM_ERROR"), ("timeout", "LLM_TIMEOUT"), ("invalid", "INVALID_OUTPUT")):
            with self.subTest(mode=mode):
                report = self.run_mode(mode)
                self.assertEqual(report.status, "NEEDS_EVIDENCE")
                self.assertIn("CREWAI_LLM_FAILED", report.risk_flags)
                self.assert_fail_closed(report)
                self.assertTrue(report.a2a_call["ok"])                 # the report itself was valid
                self.assertEqual(report.forensics["llm_review"]["status"], status)
                self.assertEqual(len(report.forensics["findings"]), 4)   # deterministic checks preserved
                if mode != "invalid":
                    self.assertFalse(report.forensics["llm_review"]["llm_call_performed"])

    def test_model_that_obeys_an_injection_cannot_authorize(self):
        for scenario in ("scenario_1_valid", "scenario_3_conflict"):
            with self.subTest(scenario=scenario):
                report = self.run_mode("obedient", scenario)
                review = report.forensics["llm_review"]
                self.assertEqual((review["status"], review["error_category"]), ("INVALID_OUTPUT", "forbidden_field"))
                self.assert_fail_closed(report)
                self.assertFalse(report.verifier.is_approved_by_verifier)

    def test_rejected_and_contradicting_findings(self):
        report = self.run_mode("unsupported")
        self.assertIn("CREWAI_LLM_FINDING_REJECTED", report.risk_flags)
        self.assertEqual(report.payout.payout_status, "PROPOSED_PENDING_APPROVAL")   # a rejected finding has no effect
        report = self.run_mode("dismissive", "scenario_3_conflict")
        self.assertTrue(any("contradicted a deterministic check" in c for c in report.conflicts))
        self.assertEqual(report.status, "CONFLICT_ESCALATED")
        self.assert_fail_closed(report)

    def test_settlement_and_human_approval_unchanged(self):
        from fastapi.testclient import TestClient
        Path(self.mode_file).write_text("valid", encoding="utf-8")
        os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        with mock.patch.dict(os.environ, {"CLAIMSLEUTH_CREWAI_A2A_URL": self.svc.url}):
            import configs.settings
            importlib.reload(configs.settings)
            import src.ui.app as app_module
            client = TestClient(importlib.reload(app_module).app)
            data = client.post("/api/adk/investigate/scenario_1_valid?forensics=crewai_a2a").json()
            self.assertEqual(data["report"]["forensics"]["llm_review"]["status"], "COMPLETED")
            self.assertEqual(data["report"]["settlement_proposal"]["approved_payout"], 251.0)   # deterministic
            self.assertEqual(data["payout"]["payout_status"], "PROPOSED_PENDING_APPROVAL")
            self.assertIsNone(data["payout"]["authorized_payout"])
            decided = client.post("/api/adjudicate/CLM-2024-001", json={"decision": "APPROVED"}).json()
            self.assertEqual(decided["payout"]["payout_status"], "AUTHORIZED")              # only after a human


if __name__ == "__main__":
    unittest.main()
