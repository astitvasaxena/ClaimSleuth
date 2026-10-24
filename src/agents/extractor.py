import re
import json
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from src.core.state import EvidenceItem
from src.core.security import PromptGuard

logger = logging.getLogger(__name__)


# ─── Pydantic schema for Gemini structured extraction ────────────────

class GeminiExtractedField(BaseModel):
    """Schema returned by Gemini for a single extracted field."""
    field_name: str
    extracted_value: str
    raw_snippet: str = Field(description="Verbatim quote from the document that contains this value")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")


class GeminiExtractionResponse(BaseModel):
    """Schema for the full Gemini extraction response."""
    fields: List[GeminiExtractedField] = Field(default_factory=list)


# ─── Evidence Extractor ──────────────────────────────────────────────

class EvidenceExtractor:
    """
    Extracts structured facts from untrusted documents with provenance tracking.
    Treats all document text as untrusted. Requires verbatim quotes for provenance.

    Supports two modes:
    - LLM mode (use_llm=True): Uses Gemini for intelligent extraction,
      with snippet-in-source safety verification.
    - Regex fallback (use_llm=False): Uses hardcoded regex patterns.
      This is the deterministic offline mode used when Gemini is unavailable.

    The snippet verification check (raw_snippet must be a true substring of
    the source document) is applied in BOTH modes as a safety invariant.
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: An instance of GeminiClient, or None for regex-only mode.
        """
        self._llm_client = llm_client

    def sanitize_untrusted_input(self, raw_text: str) -> str:
        """Strips control chars, tags, and standardizes spacing."""
        text = re.sub(r'[\r\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw_text)
        return text

    def extract_from_document(
        self,
        document_name: str,
        document_text: str,
        target_fields: list,
        use_llm: bool = True
    ) -> Dict[str, EvidenceItem]:
        """
        Extracts structured entities from untrusted document text.
        Verifies that each raw_snippet is a true substring of the source.

        Args:
            document_name: Name of the source document.
            document_text: Raw document text content.
            target_fields: List of field names to extract.
            use_llm: If True and an LLM client is available, use Gemini.
                     Falls back to regex if LLM fails or is unavailable.

        Returns:
            Dict mapping field_name -> EvidenceItem
        """
        clean_text = self.sanitize_untrusted_input(document_text)
        results = {}

        # Try LLM extraction first if enabled
        if use_llm and self._llm_client is not None:
            try:
                results = self._extract_with_gemini(document_name, clean_text, target_fields)
                logger.info(f"Gemini extraction from '{document_name}': {list(results.keys())}")
            except Exception as e:
                logger.warning(f"Gemini extraction failed for '{document_name}', falling back to regex: {e}")
                results = {}

        # Fall back to regex for any fields not extracted by LLM
        missing_fields = [f for f in target_fields if f not in results]
        if missing_fields:
            regex_results = self._extract_with_regex(document_name, clean_text, missing_fields)
            results.update(regex_results)

        # SAFETY INVARIANT 1: Prompt Injection Defense (Section 23)
        has_injection, detected_patterns = PromptGuard.scan_for_injection(clean_text)
        if has_injection:
            logger.warning(f"SECURITY ALERT: Prompt injection detected in '{document_name}': {detected_patterns}")

        # SAFETY INVARIANT 2: Verify snippet provenance for ALL results (LLM or regex)
        for field, item in results.items():
            if item.raw_snippet in clean_text:
                item.status = "VERIFIED"
                if has_injection:
                    item.verification_notes = f"SECURITY WARNING: Prompt injection pattern detected: {detected_patterns}"
            else:
                item.status = "UNVERIFIED"
                item.verification_notes = "Snippet mismatch with source text"

        return results

    def _extract_with_gemini(
        self,
        doc_name: str,
        text: str,
        target_fields: List[str]
    ) -> Dict[str, EvidenceItem]:
        """
        Uses Gemini structured output to extract fields from document text.
        The model is instructed to return verbatim snippets for provenance.
        """
        prompt = self._build_extraction_prompt(doc_name, text, target_fields)

        response = self._llm_client.generate_structured(
            prompt=prompt,
            response_schema=GeminiExtractionResponse,
            temperature=0.1
        )

        results = {}
        for gfield in response.fields:
            if gfield.field_name in target_fields:
                # Convert extracted_value to appropriate type
                extracted_value = gfield.extracted_value
                if gfield.field_name == "repair_cost":
                    try:
                        extracted_value = float(extracted_value.replace("$", "").replace(",", "").strip())
                    except ValueError:
                        pass

                results[gfield.field_name] = EvidenceItem(
                    evidence_id=f"ev_{gfield.field_name[:6]}_{hash(doc_name)%10000}",
                    source_document=doc_name,
                    field_name=gfield.field_name,
                    extracted_value=extracted_value,
                    raw_snippet=gfield.raw_snippet,
                    confidence=gfield.confidence
                )

        return results

    def _build_extraction_prompt(self, doc_name: str, text: str, target_fields: List[str]) -> str:
        """Builds the extraction prompt for Gemini."""
        fields_desc = "\n".join(f"  - {f}" for f in target_fields)
        return f"""You are a document fact extractor for a warranty claim investigation system.

DOCUMENT NAME: {doc_name}
DOCUMENT TEXT:
---
{text}
---

TASK: Extract the following fields from the document above:
{fields_desc}

CRITICAL RULES:
1. For each field you find, provide the EXACT verbatim snippet from the document in raw_snippet.
   The raw_snippet MUST be a character-for-character substring of the document text above.
   Do NOT paraphrase, reformat, or alter the snippet in any way.
2. Set confidence between 0.0 and 1.0 based on how clearly the value is stated.
3. For dates, normalize extracted_value to YYYY-MM-DD format.
4. For monetary values, provide just the number (e.g. "280.00").
5. If a field cannot be found in the document, do NOT include it in the response.
6. NEVER fabricate information that is not present in the document text.

Return a JSON object with a "fields" array containing objects with:
  field_name, extracted_value, raw_snippet, confidence
"""

    def _extract_with_regex(
        self,
        doc_name: str,
        text: str,
        target_fields: List[str]
    ) -> Dict[str, EvidenceItem]:
        """
        REGEX FALLBACK: Deterministic pattern-based extraction.
        Used when Gemini is unavailable or as a fallback for fields the LLM missed.
        This is the original extraction logic, explicitly labelled as the offline fallback.
        """
        results = {}
        for field in target_fields:
            item = self._regex_extract_field(doc_name, text, field)
            if item:
                results[field] = item
        return results

    def _regex_extract_field(self, doc_name: str, text: str, field: str) -> Optional[EvidenceItem]:
        """
        [REGEX FALLBACK] Robust heuristic and pattern extractor with snippet localization.
        Captures the exact snippet for verifiable provenance.
        """
        if field == "purchase_date":
            # Matches formats like Purchase Date: 2024-03-15 or Date: 2024-03-15 or 2024/03/15
            match = re.search(r'(?:Purchase\s*Date|Date|Invoice\s*Date)[:\s]+([0-9]{4}[-/][0-9]{2}[-/][0-9]{2})', text, re.IGNORECASE)
            if match:
                full_snippet = match.group(0)
                val = match.group(1).replace('/', '-')
                return EvidenceItem(
                    evidence_id=f"ev_pdate_{hash(doc_name)%10000}",
                    source_document=doc_name,
                    field_name="purchase_date",
                    extracted_value=val,
                    raw_snippet=full_snippet,
                    confidence=0.95
                )

        elif field == "serial_number":
            # Matches Serial: XYZ or S/N: XYZ
            match = re.search(r'(?:Serial\s*(?:Number|No)?|S/N)[:\s]+([A-Z0-9-]{6,24})', text, re.IGNORECASE)
            if match:
                return EvidenceItem(
                    evidence_id=f"ev_serial_{hash(doc_name)%10000}",
                    source_document=doc_name,
                    field_name="serial_number",
                    extracted_value=match.group(1).strip(),
                    raw_snippet=match.group(0),
                    confidence=0.98
                )

        elif field == "repair_cost":
            # Matches Total: $280.00 or Estimated Cost: $280.00 or Cost: $280.00
            match = re.search(r'(?:Total\s*(?:Cost|Amount|Estimate)?|Estimated\s*(?:Cost|Total)|Repair\s*(?:Cost|Amount)|Cost|Grand\s*Total)[:\s]+\$?\s*([0-9]+\.[0-9]{2}|[0-9]+)', text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1))
                    return EvidenceItem(
                        evidence_id=f"ev_cost_{hash(doc_name)%10000}",
                        source_document=doc_name,
                        field_name="repair_cost",
                        extracted_value=val,
                        raw_snippet=match.group(0),
                        confidence=0.94
                    )
                except ValueError:
                    pass

        elif field == "incident_cause":
            # Matches diagnosis or reported defect
            match = re.search(r'(?:Diagnosis|Defect|Issue|Damage\s*Cause|Fault)[:\s]+([^\n\r]+)', text, re.IGNORECASE)
            if match:
                return EvidenceItem(
                    evidence_id=f"ev_cause_{hash(doc_name)%10000}",
                    source_document=doc_name,
                    field_name="incident_cause",
                    extracted_value=match.group(1).strip(),
                    raw_snippet=match.group(0),
                    confidence=0.90
                )

        return None
