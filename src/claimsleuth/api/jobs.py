"""Background investigation runs: at most MAX_CONCURRENT_RUNS at a time, one per claim, with an event log per run."""
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional

from claimsleuth import runner

MAX_CONCURRENT_RUNS = 2


class AlreadyRunning(Exception):
    """An investigation of this claim is already running."""


@dataclass
class Job:
    run_id: str
    claim_id: str
    events: list = field(default_factory=list)
    done: bool = False
    result: Optional[dict] = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    async def add(self, event: dict) -> None:
        self.events.append(event)
        self.changed.set()


class Jobs:

    def __init__(self):
        self.by_run: dict[str, Job] = {}
        self.running_claims: set[str] = set()
        self.slots = asyncio.Semaphore(MAX_CONCURRENT_RUNS)

    def start(self, claim_id: str, pattern: str, prompt_style: str) -> Job:
        if claim_id in self.running_claims:
            raise AlreadyRunning(f"an investigation of {claim_id} is already running")
        job = Job(run_id=uuid.uuid4().hex[:12], claim_id=claim_id)
        self.by_run[job.run_id] = job
        self.running_claims.add(claim_id)
        asyncio.create_task(self._run(job, pattern, prompt_style))
        return job

    async def _run(self, job: Job, pattern: str, prompt_style: str) -> None:
        try:
            await job.add({"agent": "system", "type": "queued", "text": f"{pattern} / {prompt_style}"})
            async with self.slots:
                job.result = await runner.run_claim(job.claim_id, pattern, prompt_style, on_event=job.add,
                                                    run_id=job.run_id)
        except Exception as exc:                                 # bad input or a setup failure before the run
            job.result = {"run_id": job.run_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.running_claims.discard(job.claim_id)
            job.done = True
            job.changed.set()

    def get(self, run_id: str) -> Optional[Job]:
        return self.by_run.get(run_id)
