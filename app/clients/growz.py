from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.logging import logger


class GrowzClientError(RuntimeError):
    pass


class GrowzApiClient:
    """Async client for the Growz API (https://v2-api.growz.io/api).

    Authenticates via a ``token`` header (read from settings) and pages
    through list endpoints until all ``total`` records are collected.

    Requests run sequentially (single connection) and are throttled with a
    small delay between calls plus exponential backoff retries on transient
    errors, to avoid overloading the remote database.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None,
        page_size: int = 50,
        timeout_seconds: float = 45.0,
        delay_seconds: float = 0.3,
        max_retries: int = 3,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.page_size = page_size
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.http_client = http_client
        self._made_request = False

    async def fetch_treatments(self) -> list[dict[str, Any]]:
        """Fetch all treatment records (unfiltered) paging until ``total``.

        Kept for completeness; the importer uses
        :meth:`fetch_treatments_for_disease` because the API requires a
        ``disease_id`` filter in practice.
        """
        return await self._fetch_all(f"{self.base_url}/ai/treatments")

    async def fetch_treatments_for_disease(self, disease_id: str) -> list[dict[str, Any]]:
        """Fetch all treatment records for a single disease, paging until ``total``."""
        return await self._fetch_all(
            f"{self.base_url}/ai/treatments",
            extra_params={"disease_id": disease_id},
        )

    async def _fetch_all(
        self, url: str, extra_params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not self.token:
            raise GrowzClientError(
                "Growz API token is required (set GROWZ_API_TOKEN); the API returns 401 without it."
            )

        records: list[dict[str, Any]] = []
        total: int | None = None
        page = 1

        while True:
            params = {"page": page, "limit": self.page_size}
            if extra_params:
                params.update(extra_params)
            data = await self._get(url, params)
            batch = data.get("data") or []
            if total is None:
                total = int(data.get("total", len(batch)) or 0)
            records.extend(batch)
            logger.info(f"Growz fetch page={page} got={len(batch)} collected={len(records)}/{total}")

            if not batch or len(records) >= total:
                break
            page += 1

        return records

    async def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        headers = {"token": self.token or ""}

        attempt = 0
        while True:
            # Throttle: pause between requests (not before the very first one).
            if self._made_request:
                await asyncio.sleep(self.delay_seconds)
            self._made_request = True

            try:
                response = await self._request(url, params, headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.max_retries:
                    raise GrowzClientError(f"Growz API request failed after retries: {exc}") from exc
                await self._backoff(attempt, reason=str(exc))
                attempt += 1
                continue

            status = response.status_code
            # Retry on rate-limiting and server errors; surface other 4xx immediately.
            if status == 429 or status >= 500:
                if attempt >= self.max_retries:
                    logger.error(f"Growz API HTTP Error {status}: {response.text}")
                    raise GrowzClientError(
                        f"Growz API returned HTTP {status} after retries: {response.text}"
                    )
                await self._backoff(attempt, reason=f"HTTP {status}")
                attempt += 1
                continue

            if status >= 400:
                logger.error(f"Growz API HTTP Error {status}: {response.text}")
                raise GrowzClientError(f"Growz API returned HTTP {status}: {response.text}")

            return response.json()

    async def _request(
        self, url: str, params: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        if self.http_client is not None:
            return await self.http_client.get(url, params=params, headers=headers)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            return await client.get(url, params=params, headers=headers)

    async def _backoff(self, attempt: int, *, reason: str) -> None:
        backoff = min(self.delay_seconds * 2**attempt, 30.0)
        logger.warning(
            f"Growz API retry (attempt {attempt + 1}/{self.max_retries}) "
            f"after {reason}; backing off {backoff:.1f}s"
        )
        await asyncio.sleep(backoff)
