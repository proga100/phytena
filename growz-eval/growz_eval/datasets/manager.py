"""Orchestrates dataset downloads, with filtering and skip-existing logic."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from growz_eval.datasets.downloaders import (
    AuthError,
    Downloader,
    DownloadError,
    downloader_for,
)
from growz_eval.datasets.registry import DATASETS, DatasetSpec
from growz_eval.logging_utils import get_logger, step_banner

log = get_logger(__name__)


class DatasetManager:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._cache: dict[str, Downloader] = {}

    def select(
        self,
        *,
        priority: str = "high",
        crops: Iterable[str] | None = None,
    ) -> list[DatasetSpec]:
        crop_filter = set(crops) if crops else None
        selected: list[DatasetSpec] = []
        for spec in DATASETS.values():
            if priority != "all" and spec.priority != priority:
                continue
            if crop_filter and spec.crop != "mixed" and spec.crop not in crop_filter:
                continue
            selected.append(spec)
        return selected

    def download_many(self, specs: list[DatasetSpec]) -> None:
        step_banner(log, "STEP 1/2: Downloading datasets")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        if not specs:
            log.warning("No datasets matched filters.")
            return

        log.info("Will process %d datasets:", len(specs))
        for spec in specs:
            log.info("  - %-18s crop: %-6s priority: %s", spec.id, spec.crop, spec.priority)
            log.info("    %s", spec.note)

        for spec in specs:
            out_dir = self._data_dir / spec.id
            if out_dir.exists() and any(out_dir.iterdir()):
                log.info("  SKIP %s (already downloaded)", spec.id)
                continue
            self._fetch_one(spec, out_dir)

    def _fetch_one(self, spec: DatasetSpec, out_dir: Path) -> None:
        downloader = self._cache.get(spec.source)
        if downloader is None:
            downloader = downloader_for(spec.source)
            try:
                downloader.ensure_ready()
            except AuthError as exc:
                log.error("Auth error for %s downloader: %s", spec.source, exc)
                return
            except DownloadError as exc:
                log.error("Downloader unavailable for %s: %s", spec.source, exc)
                return
            self._cache[spec.source] = downloader

        try:
            downloader.fetch(spec.locator, out_dir)
        except DownloadError as exc:
            log.warning("  failed: %s — %s", spec.id, exc)
