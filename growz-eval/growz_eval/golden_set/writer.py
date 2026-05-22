"""Materialise golden set rows on disk (images + CSV + JSONL + summary)."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

from growz_eval.golden_set.models import GoldenRow
from growz_eval.logging_utils import get_logger

log = get_logger(__name__)


class GoldenSetWriter:
    """Materialises rows to disk and emits a summary."""

    def __init__(self, golden_dir: Path, *, copy_files: bool) -> None:
        self._golden_dir = golden_dir
        self._images_dir = golden_dir / "images"
        self._copy_files = copy_files

    def write(self, rows: list[GoldenRow]) -> None:
        if not rows:
            log.error("No rows to write.")
            return
        self._golden_dir.mkdir(parents=True, exist_ok=True)
        self._images_dir.mkdir(exist_ok=True)

        self._write_csv(rows)
        self._write_jsonl(rows)
        self._write_summary(rows)

    def materialise_image(self, src: Path) -> Path:
        img_id = GoldenRow.hash_id(src)
        dst = self._images_dir / f"{img_id}{src.suffix.lower()}"
        if dst.exists():
            return dst
        if self._copy_files:
            shutil.copy2(src, dst)
            return dst
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
        return dst

    def _write_csv(self, rows: list[GoldenRow]) -> None:
        path = self._golden_dir / "golden_set.csv"
        fieldnames = [f.name for f in fields(GoldenRow) if f.name != "extras"]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                data = asdict(row)
                data.pop("extras", None)
                writer.writerow(data)
        log.info("Wrote %s", path)

    def _write_jsonl(self, rows: list[GoldenRow]) -> None:
        path = self._golden_dir / "golden_set.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                data = asdict(row)
                data.pop("extras", None)
                fh.write(json.dumps(data, ensure_ascii=False) + "\n")
        log.info("Wrote %s", path)

    def _write_summary(self, rows: list[GoldenRow]) -> None:
        by_crop = Counter(r.crop for r in rows)
        by_tier = Counter(r.disease_tier for r in rows)
        by_dataset = Counter(r.source_dataset for r in rows)

        lines = [
            "Golden Set Summary",
            "=" * 40,
            f"Total images: {len(rows)}",
            "",
            "By crop:",
        ]
        lines += [f"  {k:15s} {v}" for k, v in sorted(by_crop.items(), key=lambda x: -x[1])]
        lines += ["", "By disease tier:"]
        lines += [f"  {k:15s} {v}" for k, v in sorted(by_tier.items(), key=lambda x: -x[1])]
        lines += ["", "By source dataset:"]
        lines += [f"  {k:25s} {v}" for k, v in sorted(by_dataset.items(), key=lambda x: -x[1])]

        path = self._golden_dir / "summary.txt"
        path.write_text("\n".join(lines))
        log.info("Wrote %s", path)
        for line in lines:
            log.info(line)
