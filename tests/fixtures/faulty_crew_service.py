"""
TEST FIXTURE (runs in .venv-crewai): the real CrewAI A2A forensics service with an
injectable fault, read from a control file before each request, so one process can
serve every failure scenario.

    python tests/fixtures/faulty_crew_service.py --port 8131 --control <file>

Control file contents (one word, optionally ":<seconds>"):
    none          normal behaviour
    delay:<s>     sleep <s> seconds, then answer normally (timeout tests)
    contradict    flip the serial-consistency finding (agent-disagreement tests)
    authorize     add payout/approval fields to the report (must be rejected by ADK)
    wrong_claim   return a report for a different claim
    fail          raise inside the executor (A2A task fails)
    garbage       return text that is not a ForensicsReport
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.crew_service.a2a_server import CrewForensicsExecutor, main  # noqa: E402

CONTROL = {"path": None}


def _fault() -> str:
    path = CONTROL["path"]
    return Path(path).read_text(encoding="utf-8").strip() if path and Path(path).exists() else "none"


class FaultyExecutor(CrewForensicsExecutor):
    def produce_report_json(self, claim):
        fault = _fault()
        if fault.startswith("delay:"):
            time.sleep(float(fault.split(":", 1)[1]))
            fault = "none"
        if fault == "fail":
            raise RuntimeError("injected executor failure")
        if fault == "garbage":
            return "this is not a forensics report"
        report = json.loads(super().produce_report_json(claim))
        if fault == "contradict":
            for f in report["findings"]:
                if f["check"] == "serial_consistency":
                    f["result"] = "PASS" if f["result"] == "FLAG" else "FLAG"
                    f["detail"] = "INJECTED: contradicting finding"
            flagged = any(f["result"] == "FLAG" for f in report["findings"])
            report["recommendation"] = "ESCALATE_FOR_REVIEW" if flagged else "NO_FORENSIC_ISSUES"
        elif fault == "authorize":
            report["approved_payout"] = 999.0
            report["authorize_payout"] = True
        elif fault == "wrong_claim":
            report["claim_id"] = "CLM-2024-999"
        return json.dumps(report)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--control" in args:
        i = args.index("--control")
        CONTROL["path"] = args[i + 1]
        del args[i:i + 2]
    main(args, executor_factory=lambda parsed: FaultyExecutor(llm_spec=parsed.llm))
