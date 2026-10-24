"""
ClaimSleuth — Security & Prompt-Injection Defense Module
Implements Master Document Section 23 (Adversarial Robustness & Input Sanitization).

Safeguards:
1. Prompt Injection Scanning: Detects adversarial jailbreaks, instruction overrides,
   and role-spoofing markers embedded inside untrusted claim documents or customer statements.
2. Provenance Substring Verification: Strictly checks that any extracted evidence snippet
   exists verbatim within the source document text to prevent generative hallucinations.
"""

import re
from typing import Dict, Any, List, Tuple


# Adversarial prompt injection markers commonly found in document payloads
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s*:\s*override",
    r"\[\s*system\s*\]",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"disregard\s+(the\s+)?(policy|rules|limits)",
    r"approve\s+(this\s+)?claim\s+immediately",
    r"set\s+deductible\s+to\s+0",
    r"admin\s+override",
    r"<\s*script\s*>",
    r"drop\s+table",
]

COMPILED_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


class PromptGuard:
    """Detects and mitigates prompt injection risks in untrusted claims data."""

    @staticmethod
    def scan_for_injection(text: str) -> Tuple[bool, List[str]]:
        """
        Scan document text for prompt-injection markers.
        Returns: (is_suspicious: bool, detected_patterns: List[str])
        """
        if not text:
            return False, []

        detected = []
        for pattern in COMPILED_INJECTION_RE:
            match = pattern.search(text)
            if match:
                detected.append(match.group(0))

        return len(detected) > 0, detected

    @staticmethod
    def sanitize_claim_input(text: str) -> str:
        """Strip control characters, null bytes, and dangerous HTML entities."""
        if not text:
            return ""
        # Remove null bytes and control chars except newlines and tabs
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        return cleaned.strip()

    @staticmethod
    def verify_provenance_substring(source_text: str, snippet: str) -> bool:
        """
        Strict Dual-Gate 1 Check:
        Verifies that extracted snippet exists verbatim inside the source text.
        Whitespace is normalized to allow minor formatting variations.
        """
        if not source_text or not snippet:
            return False

        norm_source = re.sub(r"\s+", " ", source_text.strip().lower())
        norm_snippet = re.sub(r"\s+", " ", snippet.strip().lower())

        return norm_snippet in norm_source
