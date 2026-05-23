from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from app.schemas import AssistantAnswer


class GeminiClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeminiCompletion:
    answer: AssistantAnswer
    raw_text: str
    raw_response: dict[str, Any]
    input_tokens: int
    output_tokens: int


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 45.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client

    async def generate_structured_answer(
        self, prompt: str, image_b64: str | None = None
    ) -> GeminiCompletion:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        parts = [{"text": prompt}]
        if image_b64:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",  # Assume JPEG for base64 from admin
                        "data": image_b64,
                    }
                }
            )

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        params = {"key": self.api_key}

        if self.http_client is not None:
            response = await self.http_client.post(url, params=params, json=payload)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, params=params, json=payload)

        if response.status_code >= 400:
            raise GeminiClientError(f"Gemini API returned HTTP {response.status_code}")

        data = response.json()
        raw_text = _extract_text(data)
        answer = _parse_answer(raw_text)
        usage = data.get("usageMetadata", {})
        return GeminiCompletion(
            answer=answer,
            raw_text=raw_text,
            raw_response=data,
            input_tokens=int(usage.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
        )


def _extract_text(data: dict[str, Any]) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiClientError("Gemini response did not include candidate text.") from exc

    text = "".join(str(part.get("text", "")) for part in parts)
    if not text.strip():
        raise GeminiClientError("Gemini response text was empty.")
    return text


def _parse_answer(raw_text: str) -> AssistantAnswer:
    json_text = _extract_json_object(raw_text)
    try:
        return AssistantAnswer.model_validate_json(json_text)
    except ValidationError as exc:
        raise GeminiClientError("Gemini response did not match AssistantAnswer schema.") from exc


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise GeminiClientError("Gemini response did not contain a JSON object.")
    json.loads(match.group(0))
    return match.group(0)
