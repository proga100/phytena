"""Read/write CSVs for the verification stage."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

_BASE_COLS: tuple[str, ...] = (
    "id", "image_path", "crop", "source_label", "source_dataset",
    "disease_tier", "true_disease",
)
_VERIFY_COLS: tuple[str, ...] = (
    "photo_quality", "gemini_verdict", "gemini_proposed", "gemini_reasoning",
    "claude_verdict", "claude_proposed", "claude_reasoning", "claude_confidence",
    "final_verdict", "final_disease", "final_confidence",
)


@dataclass(slots=True)
class VerifiedRow:
    id: str
    image_path: str
    crop: str
    source_label: str
    source_dataset: str
    disease_tier: str
    true_disease: str
    photo_quality: str = ""
    gemini_verdict: str = ""
    gemini_proposed: str = ""
    gemini_reasoning: str = ""
    claude_verdict: str = ""
    claude_proposed: str = ""
    claude_reasoning: str = ""
    claude_confidence: str = ""
    final_verdict: str = ""
    final_disease: str = ""
    final_confidence: str = ""
    extras: dict[str, str] = field(default_factory=dict)


class VerificationCsvRepository:
    @staticmethod
    def load(path: Path) -> list[VerifiedRow]:
        rows: list[VerifiedRow] = []
        known = {f.name for f in fields(VerifiedRow) if f.name != "extras"}
        with path.open(encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                core = {k: raw.get(k, "") for k in known}
                extras = {k: v for k, v in raw.items() if k not in known}
                rows.append(VerifiedRow(**core, extras=extras))
        return rows

    @staticmethod
    def save(rows: list[VerifiedRow], path: Path) -> None:
        if not rows:
            return
        extras_keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.extras:
                if key not in seen:
                    seen.add(key)
                    extras_keys.append(key)

        fieldnames = list(_BASE_COLS) + list(_VERIFY_COLS) + extras_keys

        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                data = asdict(row)
                extras = data.pop("extras", {}) or {}
                data.update(extras)
                writer.writerow(data)

    @staticmethod
    def previously_processed_ids(path: Path) -> set[str]:
        if not path.exists():
            return set()
        done: set[str] = set()
        with path.open(encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                verdict = raw.get("final_verdict", "")
                if verdict and verdict != "error":
                    done.add(raw["id"])
        return done
