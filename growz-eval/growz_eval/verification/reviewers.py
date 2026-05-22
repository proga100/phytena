"""Reviewer ABC and concrete Gemini/Claude implementations."""

from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from growz_eval.config import CLAUDE_MODEL, GEMINI_MODEL
from growz_eval.logging_utils import get_logger
from growz_eval.verification.models import (
    ClaudeVerdict,
    GeminiVerdict,
    ReviewerError,
)
from growz_eval.verification.prompts import CLAUDE_PROMPT, GEMINI_PROMPT

log = get_logger(__name__)


def _mime_for(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"


def _strip_code_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


class GeminiReviewer(ABC):
    @abstractmethod
    def review(self, image_path: Path, crop: str, dataset_label: str) -> GeminiVerdict:
        ...


class ClaudeReviewer(ABC):
    @abstractmethod
    def review(
        self,
        image_path: Path,
        crop: str,
        dataset_label: str,
        gemini_proposed: str,
        gemini_reasoning: str,
    ) -> ClaudeVerdict:
        ...


class GeminiApiReviewer(GeminiReviewer):
    def __init__(self, *, model: str = GEMINI_MODEL) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ReviewerError("GEMINI_API_KEY not set")
        try:
            from google import genai  # noqa: F401
        except ImportError as exc:
            raise ReviewerError("Run: pip install google-genai") from exc
        self._model = model
        self._api_key = api_key

    def review(self, image_path: Path, crop: str, dataset_label: str) -> GeminiVerdict:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        prompt = GEMINI_PROMPT.format(crop=crop, dataset_label=dataset_label)

        try:
            image_bytes = image_path.read_bytes()
            response = client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=_mime_for(image_path)),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(response.text.strip())
            return GeminiVerdict(
                verdict=data.get("verdict", "ERROR"),
                proposed_disease=data.get("proposed_disease", ""),
                reasoning=data.get("reasoning", ""),
                photo_quality=data.get("photo_quality", ""),
            )
        except Exception as exc:  # noqa: BLE001 — model/network errors are diverse
            return GeminiVerdict(verdict="ERROR", reasoning=str(exc)[:200])


class ClaudeApiReviewer(ClaudeReviewer):
    def __init__(self, *, model: str = CLAUDE_MODEL) -> None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ReviewerError("ANTHROPIC_API_KEY not set")
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ReviewerError("Run: pip install anthropic") from exc
        self._model = model

    def review(
        self,
        image_path: Path,
        crop: str,
        dataset_label: str,
        gemini_proposed: str,
        gemini_reasoning: str,
    ) -> ClaudeVerdict:
        import anthropic

        client = anthropic.Anthropic()
        prompt = CLAUDE_PROMPT.format(
            crop=crop,
            dataset_label=dataset_label,
            gemini_proposed=gemini_proposed,
            gemini_reasoning=gemini_reasoning,
        )

        try:
            image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
            response = client.messages.create(
                model=self._model,
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64",
                                    "media_type": _mime_for(image_path),
                                    "data": image_data}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = _strip_code_fences(response.content[0].text)
            data = json.loads(text)
            return ClaudeVerdict(
                verdict=data.get("verdict", "ERROR"),
                proposed_disease=data.get("proposed_disease", ""),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", ""),
            )
        except Exception as exc:  # noqa: BLE001
            return ClaudeVerdict(verdict="ERROR", reasoning=str(exc)[:200])
