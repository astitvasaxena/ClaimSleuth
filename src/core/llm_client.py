"""
Thin wrapper around the Google Generative AI SDK for ClaimSleuth.

- API key from GOOGLE_API_KEY environment variable only (never hardcoded).
- Structured JSON output with Pydantic schema validation.
- Timeout and retry with exponential backoff.
- Graceful degradation: raises GeminiUnavailableError if SDK is missing or API key is absent.
"""
import os
import json
import time
import logging
from typing import Type, Optional, Any, Dict

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GeminiUnavailableError(Exception):
    """Raised when Gemini cannot be used (missing SDK, no API key, network error)."""
    pass


class GeminiClient:
    """
    Thin Gemini wrapper. All ClaimSleuth LLM calls go through this class.
    If the google-generativeai SDK is not installed or GOOGLE_API_KEY is not set,
    construction raises GeminiUnavailableError so callers can fall back.
    """

    def __init__(self, model_name: Optional[str] = None):
        from configs.settings import GEMINI_MODEL
        self.model_name = model_name or GEMINI_MODEL

        # Check for SDK availability
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise GeminiUnavailableError(
                "google-generativeai package is not installed. "
                "Install with: pip install google-generativeai"
            )

        # Check for API key
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise GeminiUnavailableError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Set it with: export GOOGLE_API_KEY=your-key-here"
            )

        self._genai.configure(api_key=api_key)
        self._model = self._genai.GenerativeModel(self.model_name)

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        max_retries: int = 2,
        temperature: float = 0.2
    ) -> BaseModel:
        """
        Calls Gemini with JSON mode and validates the response against a Pydantic schema.
        Returns a validated Pydantic model instance.
        Raises GeminiUnavailableError on persistent failure after retries.
        """
        json_schema = response_schema.model_json_schema()

        generation_config = self._genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=json_schema,
            temperature=temperature,
        )

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                raw_text = response.text.strip()
                parsed = json.loads(raw_text)
                validated = response_schema(**parsed)
                return validated

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"Gemini JSON decode error (attempt {attempt+1}): {e}")
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini call failed (attempt {attempt+1}): {e}")

            if attempt < max_retries:
                backoff = 2 ** attempt
                time.sleep(backoff)

        raise GeminiUnavailableError(
            f"Gemini structured call failed after {max_retries+1} attempts: {last_error}"
        )

    def generate_text(
        self,
        prompt: str,
        max_retries: int = 2,
        temperature: float = 0.3
    ) -> str:
        """
        Simple text generation for extraction tasks.
        Returns raw text response.
        """
        generation_config = self._genai.GenerationConfig(
            temperature=temperature,
        )

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                return response.text.strip()

            except Exception as e:
                last_error = e
                logger.warning(f"Gemini text call failed (attempt {attempt+1}): {e}")

            if attempt < max_retries:
                backoff = 2 ** attempt
                time.sleep(backoff)

        raise GeminiUnavailableError(
            f"Gemini text call failed after {max_retries+1} attempts: {last_error}"
        )


def try_create_client(model_name: Optional[str] = None) -> Optional[GeminiClient]:
    """
    Attempt to create a GeminiClient. Returns None if unavailable.
    This is the recommended way to get a client — callers check for None
    and fall back to deterministic logic.
    """
    try:
        return GeminiClient(model_name=model_name)
    except GeminiUnavailableError as e:
        logger.info(f"Gemini unavailable, using deterministic fallback: {e}")
        return None
