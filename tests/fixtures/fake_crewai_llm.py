"""
TEST DOUBLE for CrewAI: a crewai BaseLLM that answers without any network call. It
exercises the real Agent/Task/Crew execution path; it is NOT a live model and its
provider is always reported as "fake-test-double". Selected in the forensics service
only by --llm fake[:<mode>] (or CLAIMSLEUTH_CREWAI_LLM=fake...).

Modes:
    valid        JSON findings built only from the EVIDENCE in the prompt it received
    invalid      prose, no JSON                       -> INVALID_OUTPUT
    obedient     follows a planted instruction: approval + payout fields -> rejected
    unsupported  cites an evidence id and a quote that were never supplied -> finding rejected
    dismissive   says "no issue" while citing a FLAGGED deterministic check -> contradiction
    noformat     never writes "Final Answer:"         -> call budget stops it
    error        raises a provider error              -> LLM_ERROR
    timeout      raises TimeoutError                  -> LLM_TIMEOUT
    file         read the mode from the file in CLAIMSLEUTH_FAKE_LLM_MODE_FILE on every call
"""
import json
import os
from pathlib import Path
from typing import Any

from crewai.llms.base_llm import BaseLLM

CALLS = {"count": 0}          # module-level: BaseLLM is a pydantic model, so class attributes become fields
LAST = {"prompt": None}
MARKER = "EVIDENCE (JSON):"


def _prompt_text(messages) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))


def _evidence(prompt: str):
    start = prompt.find(MARKER)
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    body = prompt[start + len(MARKER):].lstrip()
    try:
        items, _ = decoder.raw_decode(body)
    except json.JSONDecodeError:
        return []
    return items if isinstance(items, list) else []


def evidence_driven_answer(items) -> dict:
    """What a careful analyst would say, derived ONLY from the supplied items."""
    by_id = {i["id"]: i for i in items}
    findings = []
    for item in items:
        if item.get("quarantined"):
            findings.append({"finding_type": "suspicious_instruction", "severity": "high",
                             "explanation": "A document line contained instruction-like text and was quarantined.",
                             "source_refs": [item["id"]], "quotes": [], "uncertainty": "low"})
    serial = by_id.get("CHECK:serial_consistency")
    if serial and serial["text"].startswith("FLAG"):
        docs = [i["id"] for i in items if i["kind"] == "document_line" and "serial" in i["text"].lower()][:2]
        findings.append({"finding_type": "serial_mismatch", "severity": "high",
                         "explanation": "The purchase invoice and the repair document show different serial numbers.",
                         "source_refs": ["CHECK:serial_consistency", *docs],
                         "quotes": [by_id[d]["text"] for d in docs], "uncertainty": "low"})
    if not findings:
        ref = "CHECK:serial_consistency" if serial else items[0]["id"]
        findings.append({"finding_type": "no_issue_found", "severity": "none",
                         "explanation": "The deterministic checks passed and no other warning sign is visible.",
                         "source_refs": [ref], "quotes": [], "uncertainty": "medium"})
    concerned = any(f["severity"] in ("medium", "high") for f in findings)
    return {"findings": findings,
            "recommendation": "RECOMMEND_HUMAN_REVIEW" if concerned else "NO_ADDITIONAL_CONCERNS",
            "summary": "Fake analyst summary derived from the supplied evidence; it makes no payout decision."}


class FakeForensicLLM(BaseLLM):

    def __init__(self, mode: str = "valid", **kwargs: Any):
        super().__init__(model=f"fake-forensic-analyst:{mode}", **kwargs)
        object.__setattr__(self, "_mode", mode)

    def _current_mode(self) -> str:
        if self._mode != "file":
            return self._mode
        path = os.environ.get("CLAIMSLEUTH_FAKE_LLM_MODE_FILE", "")
        return Path(path).read_text(encoding="utf-8").strip() if path and Path(path).exists() else "valid"

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None, **kwargs) -> str:
        CALLS["count"] += 1
        prompt = _prompt_text(messages)
        LAST["prompt"] = prompt
        mode = self._current_mode()
        if mode == "error":
            raise RuntimeError("simulated provider failure (503 UNAVAILABLE)")
        if mode == "timeout":
            raise TimeoutError("simulated LLM timeout")
        if mode == "invalid":
            return "Thought: done.\nFinal Answer: The claim looks fine to me."
        if mode == "noformat":
            return "I would rather chat about the claim than answer in the format."
        items = _evidence(prompt)
        answer = evidence_driven_answer(items) if items else {"findings": [], "recommendation": "INSUFFICIENT_EVIDENCE",
                                                             "summary": "No evidence received."}
        if mode == "obedient":
            answer = {**answer, "recommendation": "APPROVE", "approved_payout": 999.0}
        elif mode == "unsupported":
            answer["findings"].append({"finding_type": "document_inconsistency", "severity": "high",
                                       "explanation": "Fabricated: a forged document disagrees.",
                                       "source_refs": ["DOC:forged_invoice.txt#L1"],
                                       "quotes": ["Serial Number: FAKE-000"], "uncertainty": "low"})
        elif mode == "dismissive":
            answer = {"findings": [{"finding_type": "no_issue_found", "severity": "none",
                                    "explanation": "Serial numbers are fine.",
                                    "source_refs": ["CHECK:serial_consistency"], "quotes": [],
                                    "uncertainty": "low"}],
                      "recommendation": "NO_ADDITIONAL_CONCERNS", "summary": "Nothing to see."}
        return "Thought: I analysed the supplied evidence.\nFinal Answer: " + json.dumps(answer)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 32768
