"""Domain types for the golden set."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImageCandidate:
    path: Path
    label: str
    dataset_id: str
    crop: str
    tier: str


@dataclass(slots=True)
class GoldenRow:
    """One row of the golden set CSV.

    Mirrors the original schema so downstream consumers don't break.
    """
    id: str
    image_path: str
    crop: str
    source_label: str
    source_dataset: str
    disease_tier: str
    true_disease: str
    verified_by_agronomist: str = ""
    photo_quality: str = ""
    region: str = ""
    growth_stage: str = ""
    severity: str = ""
    notes: str = ""
    system_diagnosis: str = ""
    system_confidence: str = ""
    judge_verdict: str = ""
    judge_reasoning: str = ""
    extras: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def hash_id(path: Path) -> str:
        return hashlib.md5(str(path).encode()).hexdigest()[:10]
