"""CLI: two-stage label verification via Gemini + Claude."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from growz_eval.config import DEFAULT_INPUT_CSV, DEFAULT_VERIFIED_CSV
from growz_eval.io_csv import VerificationCsvRepository, VerifiedRow
from growz_eval.logging_utils import get_logger, step_banner
from growz_eval.verification import VerificationPipeline
from growz_eval.verification.models import ReviewerError
from growz_eval.verification.reviewers import (
    ClaudeApiReviewer,
    ClaudeReviewer,
    GeminiApiReviewer,
    GeminiReviewer,
)

log = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage AI verification of golden set labels.",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_CSV),
                        help=f"Input CSV (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--output", default=str(DEFAULT_VERIFIED_CSV),
                        help=f"Output CSV (default: {DEFAULT_VERIFIED_CSV})")
    parser.add_argument("--limit", type=int, help="Process only N rows")
    parser.add_argument("--gemini-only", action="store_true",
                        help="Skip Claude stage (cheaper, less robust)")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Pause between API calls (sec)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already in output file")
    return parser.parse_args(argv)


def _build_reviewers(gemini_only: bool) -> tuple[GeminiReviewer, ClaudeReviewer | None]:
    try:
        gemini = GeminiApiReviewer()
    except ReviewerError as exc:
        log.error("Gemini setup failed: %s", exc)
        log.error("Get a key: https://aistudio.google.com/apikey")
        sys.exit(1)

    if gemini_only:
        return gemini, None
    try:
        claude = ClaudeApiReviewer()
    except ReviewerError as exc:
        log.error("Claude setup failed: %s", exc)
        log.error("Either set ANTHROPIC_API_KEY or pass --gemini-only.")
        sys.exit(1)
    return gemini, claude


def _print_summary(rows: list[VerifiedRow]) -> None:
    step_banner(log, "FINAL SUMMARY")
    total = len(rows)
    by_verdict = Counter(r.final_verdict for r in rows)
    by_conf = Counter(r.final_confidence for r in rows)

    log.info("Total: %d", total)
    log.info("Final verdict:")
    for kind in ("verified", "relabeled", "dataset_wrong", "ambiguous", "error"):
        n = by_verdict.get(kind, 0)
        pct = 100 * n / total if total else 0
        log.info("  %-15s %4d  (%.1f%%)", kind, n, pct)

    log.info("Final confidence:")
    for kind in ("high", "medium", "low"):
        n = by_conf.get(kind, 0)
        pct = 100 * n / total if total else 0
        log.info("  %-8s %4d  (%.1f%%)", kind, n, pct)

    usable = (by_verdict.get("verified", 0)
              + by_verdict.get("relabeled", 0)
              + by_verdict.get("dataset_wrong", 0))
    drop = by_verdict.get("ambiguous", 0) + by_verdict.get("error", 0)
    pct = 100 * usable / total if total else 0
    log.info("Usable for golden set (high+medium): %d / %d  (%.1f%%)", usable, total, pct)
    log.info("Should drop or manually review: %d", drop)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        log.error("Run build_golden_set.py first.")
        return 1

    rows = VerificationCsvRepository.load(input_path)
    if args.limit:
        rows = rows[: args.limit]

    log.info("Loaded %d rows from %s", len(rows), input_path)
    log.info("Gemini key: %s", "set" if os.getenv("GEMINI_API_KEY") else "MISSING")
    log.info("Claude key: %s", "set" if os.getenv("ANTHROPIC_API_KEY") else "MISSING")
    log.info("Output: %s", output_path)

    gemini, claude = _build_reviewers(args.gemini_only)
    skip_ids = (
        VerificationCsvRepository.previously_processed_ids(output_path)
        if args.resume else set()
    )

    pipeline = VerificationPipeline(
        gemini=gemini, claude=claude, sleep_seconds=args.sleep,
    )
    processed = pipeline.run(rows, skip_ids=skip_ids)

    VerificationCsvRepository.save(processed, output_path)
    log.info("Wrote %s", output_path)
    _print_summary(processed)
    log.info("Next: filter %s where final_confidence in (high, medium)", output_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
