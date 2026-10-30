"""The quote check shared by the ADK callbacks and the CrewAI task guardrails: a quote must appear at its line."""
import re
from typing import Optional


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def quote_problem(text: str, line: int, lines_by_number: dict[int, str]) -> Optional[str]:
    """None if `text` appears (whitespace normalised) in line `line`, else why not."""
    if line not in lines_by_number:
        return f"line {line} does not exist or is empty"
    if not normalise(text):
        return "empty quote"
    if normalise(text) not in normalise(lines_by_number[line]):
        return f"text not found at line {line}"
    return None
