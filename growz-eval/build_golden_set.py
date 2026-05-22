#!/usr/bin/env python3
"""
build_golden_set.py — Download datasets and build a balanced golden eval set
for plant disease diagnosis (Uzbekistan crops focus).

Does two things:
  1. Downloads open-source plant disease datasets (cotton, grape, tomato, etc.)
  2. Builds a stratified golden set (300 images by default) ready for evaluation

Output: ./growz_eval_workspace/golden_set/
  - images/          all selected images with stable hash IDs
  - golden_set.csv   one row per image, columns ready for your eval pipeline
  - golden_set.jsonl same data in JSONL
  - summary.txt      distribution stats

Setup (one time):
  pip install kaggle

  # Kaggle API:
  #   1. https://www.kaggle.com/settings -> API -> Create New Token
  #   2. Put kaggle.json in ~/.kaggle/
  #   3. chmod 600 ~/.kaggle/kaggle.json

Usage:
  python build_golden_set.py                       # default: 300 images
  python build_golden_set.py --size 500            # bigger set
  python build_golden_set.py --skip-download       # already have data
  python build_golden_set.py --priority all        # download medium-priority too
  python build_golden_set.py --crops cotton,grape  # only specific crops
  python build_golden_set.py --copy                # copy files instead of symlink
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ==================== CONFIG ====================

WORK_DIR = Path("./growz_eval_workspace")
DATA_DIR = WORK_DIR / "data"
GOLDEN_DIR = WORK_DIR / "golden_set"
SEED = 42

DATASETS = {
    # CRITICAL for Uzbekistan (cotton, grape)
    "ccdphd11": {
        "source": "kaggle", "slug": "dhamur/cotton-plant-diseases",
        "crop": "cotton", "priority": "high",
        "note": "11 classes of cotton diseases/pests, ~19k field images",
    },
    "ldd_grape": {
        "source": "github", "url": "https://github.com/lab-rossi/LDD",
        "crop": "grape", "priority": "high",
        "note": "Grape diseases with instance segmentation",
    },
    "grape_original": {
        "source": "kaggle", "slug": "rm1000/grape-disease-dataset-original",
        "crop": "grape", "priority": "high",
        "note": "4 classes: ESCA, Leaf Blight, Black Rot, Healthy",
    },

    # IMPORTANT — broader coverage
    "fieldplant": {
        "source": "kaggle", "slug": "khaledelsayedibrahim/fieldplant",
        "crop": "mixed", "priority": "medium",
        "note": "5,170 field photos under plant pathologist supervision",
    },
    "plantdoc": {
        "source": "github", "url": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "crop": "mixed", "priority": "medium",
        "note": "2,598 field images across 13 species",
    },
}

# Target golden set distribution (must sum to 1.0)
CROP_DISTRIBUTION = {
    "cotton": 0.27,
    "grape":  0.27,
    "tomato": 0.20,
    "potato": 0.10,
    "wheat":  0.08,
    "other":  0.08,
}

CLASS_BALANCE = {
    "healthy":   0.20,
    "common":    0.60,
    "rare":      0.15,
    "ambiguous": 0.05,
}

HEALTHY_KEYWORDS = ["healthy", "health", "normal", "ok"]
COMMON_DISEASES = {
    "cotton": ["bacterial_blight", "leaf_curl", "fusarium", "verticillium",
               "powdery_mildew", "aphid", "target_spot"],
    "grape":  ["esca", "black_rot", "leaf_blight", "downy_mildew",
               "powdery_mildew", "dead_arm"],
    "tomato": ["late_blight", "early_blight", "leaf_mold", "bacterial_spot",
               "septoria", "yellow_leaf_curl"],
    "potato": ["late_blight", "early_blight"],
    "wheat":  ["rust", "septoria", "powdery_mildew"],
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ==================== UTILITIES ====================

def log(msg, level="INFO"):
    colors = {"INFO": "\033[36m", "OK": "\033[32m", "WARN": "\033[33m",
              "ERROR": "\033[31m", "STEP": "\033[35m\033[1m"}
    reset = "\033[0m"
    print(f"{colors.get(level, '')}[{level}]{reset} {msg}")


def step(title):
    print()
    log("=" * 60, "STEP")
    log(title, "STEP")
    log("=" * 60, "STEP")


# ==================== STEP 1: DOWNLOAD ====================

def ensure_kaggle():
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("Kaggle CLI not found. Run: pip install kaggle", "ERROR")
        return False

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        log(f"Missing {kaggle_json}", "ERROR")
        log("Get token: https://www.kaggle.com/settings -> API -> New Token", "ERROR")
        return False
    return True


def download_kaggle(slug: str, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"  downloading kaggle:{slug}")
    cmd = ["kaggle", "datasets", "download", "-d", slug,
           "-p", str(out_dir), "--unzip"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log(f"  done: {slug}", "OK")
        return True
    except subprocess.CalledProcessError as e:
        log(f"  failed: {slug} — {e.stderr[:200]}", "WARN")
        return False


def download_github(url: str, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"  cloning {url}")
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(out_dir)],
                       check=True, capture_output=True, text=True)
        log(f"  done: {url}", "OK")
        return True
    except subprocess.CalledProcessError as e:
        log(f"  failed: {url} — {e.stderr[:200]}", "WARN")
        return False


def step_download(priority_filter: str, crops_filter: list):
    step("STEP 1/2: Downloading datasets")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    selected = []
    for ds_id, ds in DATASETS.items():
        if priority_filter != "all" and ds["priority"] != priority_filter:
            continue
        if crops_filter and ds["crop"] != "mixed" and ds["crop"] not in crops_filter:
            continue
        selected.append((ds_id, ds))

    if not selected:
        log("No datasets matched filters.", "WARN")
        return

    log(f"Will process {len(selected)} datasets:")
    for ds_id, ds in selected:
        log(f"  - {ds_id:18s} crop: {ds['crop']:6s} priority: {ds['priority']}")
        log(f"    {ds['note']}")

    needs_kaggle = any(ds["source"] == "kaggle" for _, ds in selected)
    if needs_kaggle and not ensure_kaggle():
        log("Cant download Kaggle datasets. Use --skip-download if data is local.", "ERROR")
        return

    for ds_id, ds in selected:
        out_dir = DATA_DIR / ds_id
        if out_dir.exists() and any(out_dir.iterdir()):
            log(f"  SKIP {ds_id} (already downloaded)", "OK")
            continue

        if ds["source"] == "kaggle":
            download_kaggle(ds["slug"], out_dir)
        elif ds["source"] == "github":
            download_github(ds["url"], out_dir)


# ==================== STEP 2: BUILD GOLDEN SET ====================

def find_images(root: Path):
    if not root.exists():
        return
    for img in root.rglob("*"):
        if img.suffix.lower() in IMG_EXTS and img.is_file():
            yield img, img.parent.name.lower().replace(" ", "_")


def classify_tier(label: str, crop: str) -> str:
    label_l = label.lower()
    if any(k in label_l for k in HEALTHY_KEYWORDS):
        return "healthy"
    if any(c in label_l for c in COMMON_DISEASES.get(crop, [])):
        return "common"
    return "rare"


def crop_from_label(label: str) -> str:
    label_l = label.lower()
    for crop in ["cotton", "grape", "tomato", "potato", "wheat",
                 "maize", "corn", "apple", "pepper", "onion"]:
        if crop in label_l:
            return "corn" if crop == "maize" else crop
    return "other"


def hash_id(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:10]


def step_build_golden(size: int, copy_files: bool):
    step(f"STEP 2/2: Building golden set ({size} images)")

    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        log(f"No data in {DATA_DIR}. Remove --skip-download or run download first.", "ERROR")
        return None

    rng = random.Random(SEED)
    pool = defaultdict(lambda: defaultdict(list))

    log("Scanning datasets...")
    for dataset_dir in DATA_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        ds_id = dataset_dir.name
        crop_hint = DATASETS.get(ds_id, {}).get("crop", "mixed")
        count = 0

        for img_path, label in find_images(dataset_dir):
            crop = crop_from_label(label) if crop_hint == "mixed" else crop_hint
            if crop not in CROP_DISTRIBUTION:
                crop = "other"
            tier = classify_tier(label, crop)
            pool[crop][tier].append((img_path, label, ds_id))
            count += 1

        log(f"  {ds_id:18s} -> {count} images")

    if not pool:
        log("No images found.", "ERROR")
        return None

    log("\nPool by crop / tier:")
    for crop in sorted(pool.keys()):
        for tier in sorted(pool[crop].keys()):
            log(f"  {crop:10s} {tier:10s} {len(pool[crop][tier])}")

    log(f"\nSampling {size} images...")
    selected = []
    for crop, crop_share in CROP_DISTRIBUTION.items():
        crop_n = int(round(size * crop_share))
        for tier, tier_share in CLASS_BALANCE.items():
            tier_n = int(round(crop_n * tier_share))
            source_tier = "rare" if tier == "ambiguous" else tier
            candidates = pool.get(crop, {}).get(source_tier, [])
            if not candidates:
                log(f"  no {source_tier} for {crop} (wanted {tier_n})", "WARN")
                continue
            picked = rng.sample(candidates, min(tier_n, len(candidates)))
            for img_path, label, ds_id in picked:
                selected.append({
                    "crop": crop, "disease_tier": tier,
                    "source_label": label, "source_dataset": ds_id,
                    "source_path": str(img_path),
                })

    if not selected:
        log("Sampling produced nothing.", "ERROR")
        return None

    # Write out
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = GOLDEN_DIR / "images"
    img_dir.mkdir(exist_ok=True)

    rows = []
    for item in selected:
        src = Path(item["source_path"])
        img_id = hash_id(src)
        new_name = f"{img_id}{src.suffix.lower()}"
        dst = img_dir / new_name

        if not dst.exists():
            if copy_files:
                shutil.copy2(src, dst)
            else:
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst)

        rows.append({
            # Identity
            "id": img_id,
            "image_path": str(dst),
            # Source info (auto-filled from dataset structure)
            "crop": item["crop"],
            "source_label": item["source_label"],
            "source_dataset": item["source_dataset"],
            "disease_tier": item["disease_tier"],
            # Ground truth (fill these via agronomist review)
            "true_disease": item["source_label"],   # initial guess from folder name
            "verified_by_agronomist": "",            # yes/no
            "photo_quality": "",                     # good/medium/poor
            "region": "",
            "growth_stage": "",                      # early/mid/late
            "severity": "",                          # mild/moderate/severe
            "notes": "",
            # Reserved for your eval pipeline output
            "system_diagnosis": "",
            "system_confidence": "",
            "judge_verdict": "",
            "judge_reasoning": "",
        })

    # CSV
    csv_path = GOLDEN_DIR / "golden_set.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log(f"Wrote {csv_path}", "OK")

    # JSONL
    jsonl_path = GOLDEN_DIR / "golden_set.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Wrote {jsonl_path}", "OK")

    # Summary
    by_crop = Counter(r["crop"] for r in rows)
    by_tier = Counter(r["disease_tier"] for r in rows)
    by_dataset = Counter(r["source_dataset"] for r in rows)

    summary_lines = [
        f"Golden Set Summary",
        f"{'=' * 40}",
        f"Total images: {len(rows)}",
        f"",
        f"By crop:",
    ]
    for k, v in sorted(by_crop.items(), key=lambda x: -x[1]):
        summary_lines.append(f"  {k:15s} {v}")
    summary_lines += ["", "By disease tier:"]
    for k, v in sorted(by_tier.items(), key=lambda x: -x[1]):
        summary_lines.append(f"  {k:15s} {v}")
    summary_lines += ["", "By source dataset:"]
    for k, v in sorted(by_dataset.items(), key=lambda x: -x[1]):
        summary_lines.append(f"  {k:25s} {v}")

    summary_path = GOLDEN_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    log(f"Wrote {summary_path}", "OK")

    print()
    print("\n".join(summary_lines))

    return rows


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=300,
                        help="Golden set size (default: 300)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing ./growz_eval_workspace/data/")
    parser.add_argument("--priority", default="high",
                        choices=["high", "medium", "all"],
                        help="Which datasets to download (default: high)")
    parser.add_argument("--crops", help="Comma-separated crops (e.g. cotton,grape)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinks (slower but portable)")
    args = parser.parse_args()

    WORK_DIR.mkdir(exist_ok=True)
    log(f"Workspace: {WORK_DIR.absolute()}")

    crops_filter = args.crops.split(",") if args.crops else []

    if not args.skip_download:
        step_download(args.priority, crops_filter)
    else:
        log("Skipping download", "WARN")

    rows = step_build_golden(args.size, args.copy)

    if rows:
        print()
        log("=" * 60, "STEP")
        log("DONE", "STEP")
        log("=" * 60, "STEP")
        log(f"Golden set ready: {GOLDEN_DIR / 'golden_set.csv'}", "OK")
        log("")
        log("Next steps:")
        log("  1. Open golden_set.csv — review the 'true_disease' column", "INFO")
        log("     (it was auto-filled from dataset folder names — verify with agronomist)")
        log("  2. When your backend is ready, run each row through your pipeline,", "INFO")
        log("     write the result into 'system_diagnosis' column")
        log("  3. Compare system_diagnosis vs true_disease to get accuracy")
        log("")
        log("Important: this is a STARTER set from open data (Bangladesh, Turkey,", "WARN")
        log("China). Real Uzbek field photos are NOT in here. Replace with your own", "WARN")
        log("production logs once you have feedback loop running.", "WARN")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
