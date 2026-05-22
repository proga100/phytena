"""CLI: build a balanced golden set from open-source datasets."""

from __future__ import annotations

import argparse
import sys

from growz_eval.config import DATA_DIR, GOLDEN_DIR, SEED, WORK_DIR
from growz_eval.datasets import DatasetManager
from growz_eval.golden_set import GoldenSetBuilder
from growz_eval.logging_utils import get_logger, step_banner

log = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets and build a balanced golden eval set.",
    )
    parser.add_argument("--size", type=int, default=300,
                        help="Golden set size (default: 300)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing ./growz_eval_workspace/data/")
    parser.add_argument("--priority", default="high",
                        choices=["high", "medium", "all"],
                        help="Which datasets to download (default: high)")
    parser.add_argument("--crops",
                        help="Comma-separated crops to filter (e.g. cotton,grape)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinks (portable, slower)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    WORK_DIR.mkdir(exist_ok=True)
    log.info("Workspace: %s", WORK_DIR.absolute())

    if not args.skip_download:
        crops = [c.strip() for c in args.crops.split(",")] if args.crops else None
        manager = DatasetManager(data_dir=DATA_DIR)
        specs = manager.select(priority=args.priority, crops=crops)
        manager.download_many(specs)
    else:
        log.warning("Skipping download")

    builder = GoldenSetBuilder(
        data_dir=DATA_DIR,
        golden_dir=GOLDEN_DIR,
        seed=SEED,
        copy_files=args.copy,
    )
    rows = builder.build(args.size)
    if not rows:
        return 1

    step_banner(log, "DONE")
    log.info("Golden set ready: %s", GOLDEN_DIR / "golden_set.csv")
    log.info("Next: run verify_labels.py to AI-verify the labels.")
    log.warning(
        "Note: this is a STARTER set from open data (Bangladesh, Turkey, China). "
        "Replace with real Uzbek field photos once you have them."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
