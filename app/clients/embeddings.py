from __future__ import annotations


import httpx
from app.logging import logger

class EmbeddingsClientError(RuntimeError):
    pass

class EmbeddingsClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-embedding-2",
        timeout_seconds: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client

    async def get_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT", output_dimensionality: int | None = None) -> list[float]:
        logger.info(f"Requesting embedding for text (len: {len(text)}) using {self.model}")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent"
        )
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
        if output_dimensionality:
            payload["outputDimensionality"] = output_dimensionality
        
        params = {"key": self.api_key}

        if self.http_client is not None:
            response = await self.http_client.post(url, params=params, json=payload)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, params=params, json=payload)

        if response.status_code >= 400:
            logger.error(f"Google Embeddings API HTTP Error {response.status_code}: {response.text}")
            raise EmbeddingsClientError(f"Google Embeddings API returned HTTP {response.status_code}: {response.text}")

        data = response.json()
        embedding = data.get("embedding", {}).get("values")
        if not embedding:
            raise EmbeddingsClientError("Google Embeddings API response did not include embedding values.")
        
        return [float(v) for v in embedding]

    async def get_embeddings_batch(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", output_dimensionality: int | None = None) -> list[list[float]]:
        logger.info(f"Requesting batch embedding for {len(texts)} texts using {self.model}")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:batchEmbedContents"
        )
        requests = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }
            for text in texts
        ]
        if output_dimensionality:
            for req in requests:
                req["outputDimensionality"] = output_dimensionality

        payload = {"requests": requests}
        params = {"key": self.api_key}

        if self.http_client is not None:
            response = await self.http_client.post(url, params=params, json=payload)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, params=params, json=payload)

        if response.status_code >= 400:
            logger.error(f"Google Embeddings API HTTP Error {response.status_code}: {response.text}")
            raise EmbeddingsClientError(f"Google Embeddings API returned HTTP {response.status_code}: {response.text}")

        data = response.json()
        embeddings = [req.get("values") for req in data.get("embeddings", [])]
        if not embeddings or any(e is None for e in embeddings):
            raise EmbeddingsClientError("Google Embeddings API response did not include all embedding values.")
        
        return [[float(v) for v in e] for e in embeddings]
