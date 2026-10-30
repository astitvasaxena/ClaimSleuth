"""Rate-limit pacing: calls are spaced to MODEL_MAX_RPM, and waits are kept out of the agent timings."""
import asyncio
import time
import unittest
from unittest import mock

from claimsleuth import config, ratelimit
from claimsleuth.callbacks.metrics import RunRecorder


class TestPacing(unittest.TestCase):

    def test_calls_are_spaced_to_the_rate(self):
        with mock.patch.object(config, "MODEL_MAX_RPM", 600):           # one call per 0.1 s
            ratelimit._next_slot = 0.0
            started = time.monotonic()
            waits = [asyncio.run(ratelimit.pace()) for _ in range(4)]
        self.assertAlmostEqual(time.monotonic() - started, 0.3, delta=0.08)
        self.assertEqual(waits[0], 0.0)
        self.assertTrue(all(w > 0.05 for w in waits[1:]))


class TestWaitsExcluded(unittest.TestCase):

    def test_an_agents_own_wait_is_not_its_time(self):
        rec = RunRecorder("r")
        rec.start("policy_agent")
        time.sleep(0.2)
        rec.add_wait("policy_agent", 0.15)                               # 0.15 s of the 0.2 s was pacing
        rec.stop("policy_agent")
        self.assertAlmostEqual(rec.rows["policy_agent"].seconds, 0.05, delta=0.04)
        self.assertAlmostEqual(rec.wait_seconds, 0.15)

    def test_parallel_agent_keeps_its_time_while_another_waits(self):
        rec = RunRecorder("r")
        rec.start("claim_manager")
        rec.start("fraud_crew")
        rec.start("policy_agent")
        time.sleep(0.2)
        rec.add_wait("policy_agent", 0.1)
        for agent in ("policy_agent", "fraud_crew", "claim_manager"):
            rec.stop(agent)
        self.assertAlmostEqual(rec.rows["fraud_crew"].seconds, 0.2, delta=0.05)     # not reduced by policy's wait
        self.assertAlmostEqual(rec.rows["policy_agent"].seconds, 0.1, delta=0.05)
        self.assertAlmostEqual(rec.rows["claim_manager"].seconds, 0.1, delta=0.05)  # container: every wait excluded

    def test_crew_wait_reported_by_the_service_is_excluded(self):
        rec = RunRecorder("r")
        rec.start("fraud_crew")
        time.sleep(0.2)
        rec.add_wait("fraud_crew", 0.12)
        rec.stop("fraud_crew")
        self.assertAlmostEqual(rec.rows["fraud_crew"].seconds, 0.08, delta=0.05)
        self.assertEqual(rec.totals()["rate_limit_wait_s"], 0.12)
