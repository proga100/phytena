"""High-level orchestration: scan -> sample -> materialise."""

from __future__ import annotations

from pathlib import Path

from growz_eval.golden_set.models import GoldenRow, ImageCandidate
from growz_eval.golden_set.sampler import StratifiedSampler
from growz_eval.golden_set.scanner import DatasetScanner
from growz_eval.golden_set.writer import GoldenSetWriter
from growz_eval.logging_utils import get_logger, step_banner

log = get_logger(__name__)


class GoldenSetBuilder:
    def __init__(
        self,
        *,
        data_dir: Path,
        golden_dir: Path,
        seed: int,
        copy_files: bool,
    ) -> None:
        self._data_dir = data_dir
        self._scanner = DatasetScanner()
        self._sampler = StratifiedSampler(seed=seed)
        self._writer = GoldenSetWriter(golden_dir, copy_files=copy_files)

    def build(self, size: int) -> list[GoldenRow]:
        step_banner(log, f"STEP 2/2: Building golden set ({size} images)")

        if not self._data_dir.exists() or not any(self._data_dir.iterdir()):
            log.error("No data in %s. Run the download step first.", self._data_dir)
            return []

        pool = self._scanner.scan(self._data_dir)
        if not pool:
            log.error("No images found in any dataset.")
            return []

        self._scanner.log_distribution(pool)

        log.info("Sampling %d images...", size)
        picked = self._sampler.sample(pool, size)
        if not picked:
            log.error("Sampling produced no rows.")
            return []

        rows = [self._to_row(c) for c in picked]
        self._writer.write(rows)
        return rows

    def _to_row(self, candidate: ImageCandidate) -> GoldenRow:
        dst = self._writer.materialise_image(candidate.path)
        return GoldenRow(
            id=GoldenRow.hash_id(candidate.path),
            image_path=str(dst),
            crop=candidate.crop,
            source_label=candidate.label,
            source_dataset=candidate.dataset_id,
            disease_tier=candidate.tier,
            true_disease=candidate.label,
        )
