import httpx
import pytest

from app.clients.gemini import GeminiClient


@pytest.mark.asyncio
async def test_gemini_client_parses_structured_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": """
                                    {
                                      "diagnoses": [
                                        {
                                          "name": "Недостаточно данных",
                                          "category": "unknown",
                                          "confidence": 0.3,
                                          "evidence": ["Нет фото"]
                                        }
                                      ],
                                      "confidence": "low",
                                      "answer": "Нужно больше данных.",
                                      "actions": ["Пришлите фото"],
                                      "warnings": ["Не применяйте препараты без подтверждения"],
                                      "citations": [],
                                      "needs_clarification": true,
                                      "clarification_question": "Пришлите фото листа.",
                                      "escalate_to_agronomist": false
                                    }
                                    """
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GeminiClient(api_key="test", model="gemini-test", http_client=http_client)
        completion = await client.generate_structured_answer("prompt")

    assert completion.answer.confidence == "low"
    assert completion.input_tokens == 10
    assert completion.output_tokens == 20
