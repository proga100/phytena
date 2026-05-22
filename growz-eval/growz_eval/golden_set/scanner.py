"""Discover image files in downloaded datasets and group them by crop/tier."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from growz_eval.config import IMG_EXTS
from growz_eval.datasets.registry import DATASETS
from growz_eval.golden_set.classifier import CropClassifier, TierClassifier
from growz_eval.golden_set.models import ImageCandidate
from growz_eval.logging_utils import get_logger

log = get_logger(__name__)


CandidatePool = dict[str, dict[str, list[ImageCandidate]]]


class DatasetScanner:
    def __init__(self) -> None:
        self._crop = CropClassifier()
        self._tier = TierClassifier()

    def scan(self, data_dir: Path) -> CandidatePool:
        pool: CandidatePool = defaultdict(lambda: defaultdict(list))

        log.info("Scanning datasets...")
        for dataset_dir in sorted(data_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            ds_id = dataset_dir.name
            spec = DATASETS.get(ds_id)
            crop_hint = spec.crop if spec else "mixed"

            count = 0
            for img_path, raw_label in self._iter_images(dataset_dir):
                crop = self._crop.classify(raw_label, crop_hint)
                tier = self._tier.classify(raw_label, crop)
                pool[crop][tier].append(
                    ImageCandidate(
                        path=img_path,
                        label=raw_label,
                        dataset_id=ds_id,
                        crop=crop,
                        tier=tier,
                    )
                )
                count += 1
            log.info("  %-18s -> %d images", ds_id, count)

        return pool

    @staticmethod
    def _iter_images(root: Path) -> Iterator[tuple[Path, str]]:
        for img in root.rglob("*"):
            if not img.is_file():
                continue
            if img.suffix.lower() not in IMG_EXTS:
                continue
            label = img.parent.name.lower().replace(" ", "_")
            yield img, label

    @staticmethod
    def log_distribution(pool: CandidatePool) -> None:
        log.info("Pool by crop / tier:")
        for crop in sorted(pool.keys()):
            for tier in sorted(pool[crop].keys()):
                log.info("  %-10s %-10s %d", crop, tier, len(pool[crop][tier]))
