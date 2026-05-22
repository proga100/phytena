#!/usr/bin/env python3
"""
verify_labels.py — Two-stage label verification for golden set.

Stage 1: Gemini 3.1 Pro looks at each image, checks if the dataset label
         matches what it sees. Cheap, fast (~$2-5 for 300 images).

Stage 2: For cases where Gemini DISAGREES with the dataset label,
         Claude Opus gives a second opinion. Slower, more expensive,
         but used only on the disputed minority (~$5-10 typical).

Final verdict logic:
  - Gemini agrees with dataset label                         -> KEEP (high confidence)
  - Gemini disagrees + Claude agrees with Gemini             -> RELABEL (use Gemini's label)
  - Gemini disagrees + Claude agrees with dataset            -> KEEP (dataset was right)
  - Gemini disagrees + Claude proposes a third option        -> AMBIGUOUS (drop or flag)

Output: golden_set_verified.csv with new columns:
  - gemini_verdict        (agree | disagree | uncertain)
  - gemini_proposed       (what Gemini thinks it is)
  - claude_verdict        (only filled for disputes)
  - claude_proposed       (only filled for disputes)
  - final_verdict         (verified | relabeled | ambiguous | dataset_wrong)
  - final_disease         (the agreed-on label to use as ground truth)
  - final_confidence      (high | medium | low)

Setup:
  pip install google-genai anthropic pillow

  export GEMINI_API_KEY="..."           # from https://aistudio.google.com/apikey
  export ANTHROPIC_API_KEY="..."        # from https://console.anthropic.com/

Usage:
  python verify_labels.py                       # process all
  python verify_labels.py --limit 20            # test on 20 first
  python verify_labels.py --gemini-only         # skip Claude stage (cheaper, less accurate)
  python verify_labels.py --resume              # continue from previous run
  python verify_labels.py --input ./my.csv      # custom input file
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

GOLDEN_DIR = Path("./growz_eval_workspace/golden_set")
DEFAULT_INPUT = GOLDEN_DIR / "golden_set.csv"
DEFAULT_OUTPUT = GOLDEN_DIR / "golden_set_verified.csv"

# ==================== PROMPTS ====================

VERIFICATION_PROMPT = """You are an expert plant pathologist examining a photo for the Growz diagnosis system.

Context:
- Crop: {crop}
- Proposed label from open-source dataset: "{dataset_label}"

Your task: look at the image and decide whether the dataset label is correct.

Important rules:
- The dataset label uses underscores and abbreviations (e.g. "bacterial_blight", "esca", "leaf_curl_virus"). Treat synonyms as matching (e.g. "ggm" and "grape_gray_mold" are the same).
- If the photo quality is too poor to tell (blurry, no leaf visible, wrong subject), respond with verdict="uncertain".
- If you see a CLEARLY different disease than the label, respond with verdict="disagree" and propose what you actually see.
- "Healthy" means no disease symptoms visible.

Respond ONLY in this JSON format, no other text:
{{
  "verdict": "agree" | "disagree" | "uncertain",
  "proposed_disease": "what you actually see (use same naming style as dataset label)",
  "reasoning": "one short sentence",
  "photo_quality": "good" | "medium" | "poor"
}}"""


CLAUDE_TIEBREAK_PROMPT = """You are a senior plant pathologist resolving a disagreement between two diagnostic opinions for the Growz system.

Context:
- Crop: {crop}
- Original dataset label: "{dataset_label}"
- First reviewer (Gemini) disagreed and proposed: "{gemini_proposed}"
- First reviewer reasoning: "{gemini_reasoning}"

Your task: examine the image yourself and make the final call.

Respond ONLY in this JSON format:
{{
  "verdict": "dataset_correct" | "gemini_correct" | "neither_correct" | "uncertain",
  "proposed_disease": "the correct disease name (use underscore_style)",
  "reasoning": "one sentence explaining your decision",
  "confidence": "high" | "medium" | "low"
}}"""


# ==================== GEMINI ====================

def call_gemini(image_path: Path, crop: str, dataset_label: str) -> dict:
    """Call Gemini 3.1 Pro Vision."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {"verdict": "ERROR", "reasoning": "Run: pip install google-genai"}

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"verdict": "ERROR", "reasoning": "GEMINI_API_KEY not set"}

    client = genai.Client(api_key=api_key)
    prompt = VERIFICATION_PROMPT.format(crop=crop, dataset_label=dataset_label)

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Detect mime type
        ext = image_path.suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        text = response.text.strip()
        return json.loads(text)
    except Exception as e:
        return {"verdict": "ERROR", "reasoning": str(e)[:200]}


# ==================== CLAUDE ====================

def call_claude(image_path: Path, crop: str, dataset_label: str,
                gemini_proposed: str, gemini_reasoning: str) -> dict:
    """Call Claude Opus Vision for tiebreak."""
    try:
        import anthropic
    except ImportError:
        return {"verdict": "ERROR", "reasoning": "Run: pip install anthropic"}

    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"verdict": "ERROR", "reasoning": "ANTHROPIC_API_KEY not set"}

    client = anthropic.Anthropic()
    prompt = CLAUDE_TIEBREAK_PROMPT.format(
        crop=crop,
        dataset_label=dataset_label,
        gemini_proposed=gemini_proposed,
        gemini_reasoning=gemini_reasoning,
    )

    try:
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        ext = image_path.suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": mime, "data": image_data}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = response.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"verdict": "ERROR", "reasoning": str(e)[:200]}


# ==================== DECISION LOGIC ====================

def decide_final(row: dict) -> tuple[str, str, str]:
    """
    Returns (final_verdict, final_disease, final_confidence).

    final_verdict ∈ {verified, relabeled, ambiguous, dataset_wrong, error}
    """
    gv = row.get("gemini_verdict", "")
    cv = row.get("claude_verdict", "")

    # Errors
    if gv == "ERROR":
        return "error", row["true_disease"], "low"

    # Gemini agrees -> trust dataset
    if gv == "agree":
        return "verified", row["true_disease"], "high"

    # Gemini uncertain -> ambiguous, keep dataset label but flag
    if gv == "uncertain":
        return "ambiguous", row["true_disease"], "low"

    # Gemini disagrees -> need Claude
    if gv == "disagree":
        # If Claude not called (gemini-only mode)
        if not cv:
            return "ambiguous", row["true_disease"], "low"

        if cv == "ERROR":
            return "error", row["true_disease"], "low"

        if cv == "dataset_correct":
            return "verified", row["true_disease"], "high"

        if cv == "gemini_correct":
            return "relabeled", row.get("gemini_proposed", ""), "medium"

        if cv == "neither_correct":
            return "dataset_wrong", row.get("claude_proposed", ""), "medium"

        if cv == "uncertain":
            return "ambiguous", row["true_disease"], "low"

    return "error", row["true_disease"], "low"


# ==================== MAIN PROCESSING ====================

def process_rows(rows: list, gemini_only: bool, sleep: float, resume_path: Path):
    # Load existing results if resuming
    done_ids = set()
    if resume_path.exists():
        with open(resume_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("final_verdict") and r["final_verdict"] != "error":
                    done_ids.add(r["id"])
        print(f"[RESUME] {len(done_ids)} rows already processed, skipping them")

    # Stage 1: Gemini on all
    print(f"\n{'='*60}\nSTAGE 1: Gemini 3.1 Pro on {len(rows)} images\n{'='*60}")

    stage1_results = []
    for i, row in enumerate(rows, 1):
        if row["id"] in done_ids:
            continue

        img_path = Path(row["image_path"])
        if not img_path.exists():
            print(f"[{i}/{len(rows)}] SKIP — missing {img_path}")
            continue

        result = call_gemini(img_path, row["crop"], row["true_disease"])

        row["gemini_verdict"] = result.get("verdict", "ERROR")
        row["gemini_proposed"] = result.get("proposed_disease", "")
        row["gemini_reasoning"] = result.get("reasoning", "")
        row["photo_quality"] = result.get("photo_quality", "")
        row["claude_verdict"] = ""
        row["claude_proposed"] = ""
        row["claude_reasoning"] = ""
        row["claude_confidence"] = ""

        stage1_results.append(row)

        marker = {"agree": "OK", "disagree": "??", "uncertain": "??",
                  "ERROR": "!!"}.get(row["gemini_verdict"], "?")
        print(f"[{i:3d}/{len(rows)}] {marker} {row['crop']:8s} "
              f"label={row['true_disease'][:20]:20s} "
              f"-> {row['gemini_verdict']:10s} "
              f"{('| ' + row['gemini_proposed'][:25]) if row['gemini_verdict'] == 'disagree' else ''}")

        time.sleep(sleep)

    # Summary of Stage 1
    counts = Counter(r["gemini_verdict"] for r in stage1_results)
    print(f"\nStage 1 results:")
    for k, v in sorted(counts.items()):
        print(f"  {k:15s} {v}")

    disputes = [r for r in stage1_results if r["gemini_verdict"] == "disagree"]
    print(f"\nDisputed cases: {len(disputes)} (will go to Claude)")

    if gemini_only:
        print("\n[--gemini-only] Skipping Claude stage")
    elif disputes:
        # Stage 2: Claude on disputes only
        print(f"\n{'='*60}\nSTAGE 2: Claude Opus tiebreak on {len(disputes)} disputes\n{'='*60}")

        for i, row in enumerate(disputes, 1):
            img_path = Path(row["image_path"])
            result = call_claude(
                img_path, row["crop"], row["true_disease"],
                row["gemini_proposed"], row["gemini_reasoning"],
            )

            row["claude_verdict"] = result.get("verdict", "ERROR")
            row["claude_proposed"] = result.get("proposed_disease", "")
            row["claude_reasoning"] = result.get("reasoning", "")
            row["claude_confidence"] = result.get("confidence", "")

            print(f"[{i:3d}/{len(disputes)}] {row['crop']:8s} "
                  f"dataset={row['true_disease'][:15]:15s} "
                  f"gemini={row['gemini_proposed'][:15]:15s} "
                  f"-> claude: {row['claude_verdict']}")

            time.sleep(sleep)

    # Final decisions
    for row in stage1_results:
        verdict, disease, conf = decide_final(row)
        row["final_verdict"] = verdict
        row["final_disease"] = disease
        row["final_confidence"] = conf

    return stage1_results


def write_results(rows: list, out_path: Path):
    if not rows:
        return

    # Ensure consistent column order
    base_cols = ["id", "image_path", "crop", "source_label", "source_dataset",
                 "disease_tier", "true_disease"]
    verify_cols = ["photo_quality", "gemini_verdict", "gemini_proposed",
                   "gemini_reasoning", "claude_verdict", "claude_proposed",
                   "claude_reasoning", "claude_confidence",
                   "final_verdict", "final_disease", "final_confidence"]
    other_cols = [k for k in rows[0].keys()
                  if k not in base_cols and k not in verify_cols]

    fieldnames = base_cols + verify_cols + other_cols
    # Add any missing cols that exist in rows
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            # Fill missing keys
            for k in fieldnames:
                r.setdefault(k, "")
            writer.writerow(r)

    print(f"\n[OK] Wrote {out_path}")


def print_summary(rows: list):
    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")

    by_verdict = Counter(r["final_verdict"] for r in rows)
    by_conf = Counter(r["final_confidence"] for r in rows)

    total = len(rows)
    print(f"Total: {total}")
    print(f"\nFinal verdict:")
    for k in ["verified", "relabeled", "dataset_wrong", "ambiguous", "error"]:
        v = by_verdict.get(k, 0)
        pct = 100 * v / total if total else 0
        print(f"  {k:15s} {v:4d}  ({pct:.1f}%)")

    print(f"\nFinal confidence:")
    for k in ["high", "medium", "low"]:
        v = by_conf.get(k, 0)
        pct = 100 * v / total if total else 0
        print(f"  {k:8s} {v:4d}  ({pct:.1f}%)")

    usable = by_verdict.get("verified", 0) + by_verdict.get("relabeled", 0) \
           + by_verdict.get("dataset_wrong", 0)
    print(f"\nUsable for golden set (high+medium confidence): {usable} / {total}  "
          f"({100*usable/total:.1f}%)")
    print(f"Should drop or manually review: {by_verdict.get('ambiguous', 0) + by_verdict.get('error', 0)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help=f"Input CSV (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Output CSV (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--limit", type=int, help="Process only N rows")
    parser.add_argument("--gemini-only", action="store_true",
                        help="Skip Claude stage (cheaper, less robust)")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Pause between API calls (sec)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already in output file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        print("Run build_golden_set.py first.")
        sys.exit(1)

    rows = list(csv.DictReader(open(input_path, encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Gemini key: {'set' if os.getenv('GEMINI_API_KEY') else 'MISSING'}")
    print(f"Claude key: {'set' if os.getenv('ANTHROPIC_API_KEY') else 'MISSING'}")
    print(f"Output:     {output_path}")

    if not os.getenv("GEMINI_API_KEY"):
        print("\n[ERROR] GEMINI_API_KEY not set")
        print("Get one: https://aistudio.google.com/apikey")
        sys.exit(1)

    if not args.gemini_only and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n[ERROR] ANTHROPIC_API_KEY not set (or use --gemini-only)")
        sys.exit(1)

    resume_path = output_path if args.resume else Path("/tmp/__nonexistent__")
    results = process_rows(rows, args.gemini_only, args.sleep, resume_path)

    write_results(results, output_path)
    print_summary(results)

    print(f"\nNext: filter golden_set_verified.csv where final_confidence in (high, medium)")
    print(f"      that's your verified golden set, ready for eval pipeline")


if __name__ == "__main__":
    main()
