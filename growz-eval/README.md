# Growz Eval Pipeline

Build a verified golden evaluation set for plant disease diagnosis (Uzbekistan focus: cotton, grape, tomato).

This pipeline downloads open-source plant disease datasets, samples a balanced subset, and verifies labels using two AI models (Gemini 3.1 Pro + Claude Opus) as stand-in agronomists. The output is a clean CSV ready to evaluate any diagnosis backend.

---

## What it does

```
Open datasets (Kaggle + GitHub)
        │
        ▼
[ build_golden_set.py ]     Download + sample 300 balanced images
        │
        ▼
golden_set.csv              Raw set with dataset labels (may be noisy)
        │
        ▼
[ verify_labels.py ]        Two-stage AI verification
        │   Stage 1: Gemini 3.1 Pro reviews every image
        │   Stage 2: Claude Opus settles disputes
        ▼
golden_set_verified.csv     Each row has gemini/claude verdicts + final decision
        │
        ▼
golden_set_clean.csv        Filtered to high+medium confidence only
                            → use this as ground truth for your eval
```

---

## Why this exists

Growz is an AI farming assistant for Uzbekistan. Current LLM-first pipeline has low accuracy on real field photos. Before improving anything, we need to **measure** baseline accuracy. That requires a labeled test set.

Hiring an agronomist takes weeks. This pipeline gets you a usable golden set in **2-3 hours** using open data and AI-assisted verification. It's a starter set — replace it with real production logs once feedback loop is running.

---

## Data sources (5 datasets total)

| # | Dataset | Source | Crop | Size |
|---|---------|--------|------|------|
| 1 | CCDPHD-11 | Kaggle (`dhamur/cotton-plant-diseases`) | cotton | ~19,000 |
| 2 | LDD | GitHub (`lab-rossi/LDD`) | grape | ~1,000 |
| 3 | Grape Original | Kaggle (`rm1000/grape-disease-dataset-original`) | grape | ~4,000 |
| 4 | FieldPlant | Kaggle (`khaledelsayedibrahim/fieldplant`) | mixed | ~5,000 |
| 5 | PlantDoc | GitHub (`pratikkayal/PlantDoc-Dataset`) | 13 species | ~2,500 |

All datasets are openly published for research. We sample 300 balanced images from this pool.

**Important:** none of these are from Uzbekistan. They come from Bangladesh, Italy, Turkey, US, China. There is a domain gap. This pipeline gives you a useful **starter** baseline, not a final answer. Replace with real Uzbek field photos as soon as you have them.

---

## Project layout

```
growz-eval/
├── pyproject.toml           # deps + tool config (ruff, pytest)
├── build_golden_set.py      # CLI entrypoint (thin wrapper)
├── verify_labels.py         # CLI entrypoint (thin wrapper)
├── growz_eval/              # package
│   ├── config.py            # paths, distributions, model IDs
│   ├── logging_utils.py     # logger with API-key redaction filter
│   ├── cli/                 # argparse + orchestration
│   ├── datasets/            # source registry + Kaggle/GitHub downloaders
│   ├── golden_set/          # scanner, classifier, sampler, writer
│   ├── verification/        # Gemini + Claude reviewers + decision engine
│   └── io_csv/              # typed CSV read/write
└── tests/                   # unit tests for classifier, sampler, decision
```

The code is structured around small, single-purpose classes. Reviewers (Gemini, Claude) and downloaders (Kaggle, GitHub) are pluggable via abstract base classes — swap providers or add sources without touching the pipeline.

---

## Prerequisites

- Python 3.10+
- ~10 GB free disk space
- Kaggle account (free): https://www.kaggle.com/
- Gemini API key: https://aistudio.google.com/apikey
- Anthropic API key + $10 balance: https://console.anthropic.com/  *(optional — see `--gemini-only`)*
- ~$10 budget for API calls (300 images verification)

---

## Setup (one-time, 15 minutes)

### 1. Install Python dependencies

```bash
cd growz-eval

python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -e ".[dev]"           # installs the package + dev tools (pytest, ruff)
```

`pip install -e .` reads `pyproject.toml`, pulls pinned versions of `kaggle`, `google-genai`, `anthropic`, `tenacity`, and registers the package in editable mode. The `[dev]` extra adds `pytest`, `ruff`, `mypy`.

### 2. Configure Kaggle API

1. Go to https://www.kaggle.com/settings → API → **Create New Token**
2. The page either downloads a `kaggle.json` or shows a one-time `KGAT_...` token.

If you got a `kaggle.json`:

```bash
# Linux/Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows PowerShell
mkdir $HOME\.kaggle
move $HOME\Downloads\kaggle.json $HOME\.kaggle\
```

If you got a `KGAT_...` token instead:

```bash
mkdir -p ~/.kaggle
echo KGAT_xxxxxxxxxxxxxxxx > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Verify:
```bash
kaggle datasets list --max-size 1000
```

### 3. Set API keys

```bash
# Linux/Mac
export GEMINI_API_KEY="AIza..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows PowerShell
$env:GEMINI_API_KEY = "AIza..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

To persist them, add to `~/.bashrc` / `~/.zshrc` (Linux/Mac) or System Environment Variables (Windows).

**Important for Anthropic:** make sure you have $5+ balance on the account (Settings → Billing), otherwise the key returns errors. If you don't have an Anthropic key yet, pass `--gemini-only` and you'll still get a useful (single-reviewer) verified set.

### 4. (Optional) Run tests

```bash
pytest -q
```

22 unit tests should pass. They cover the classifier, sampler, and decision engine — no network needed.

---

## How to run

### Step 1. Build the golden set

First, a small test run (no API costs, just downloads):

```bash
python build_golden_set.py --size 50
```

This will:
1. Download all 5 datasets (~5-10 GB, takes 30-90 minutes depending on internet)
2. Scan the downloaded files
3. Sample 50 balanced images
4. Write `growz_eval_workspace/golden_set/golden_set.csv`

If everything works, do the full run:

```bash
python build_golden_set.py --skip-download --size 300
```

The `--skip-download` flag reuses already-downloaded data. Takes 1-2 minutes.

**Output:**
```
growz_eval_workspace/
├── data/                    # Raw datasets (~10 GB, can delete later)
└── golden_set/
    ├── images/              # 300 selected images with hash IDs
    ├── golden_set.csv       # Main file — one row per image
    ├── golden_set.jsonl     # Same data, JSONL format
    └── summary.txt          # Distribution stats
```

### Step 2. Verify labels (small test first)

Always test on 20 images before spending money on 300:

```bash
python verify_labels.py --limit 20
```

This costs ~$0.50 and takes ~5 minutes. Open the output CSV and check that:
- `gemini_reasoning` makes sense
- `claude_reasoning` (for disputes) makes sense
- `final_disease` has reasonable values

If you don't have an Anthropic key, run Gemini only:

```bash
python verify_labels.py --limit 20 --gemini-only
```

### Step 3. Full verification

If the test looks good:

```bash
python verify_labels.py
```

This costs ~$10 and takes 40-60 minutes. Two stages:
- **Stage 1:** Gemini 3.1 Pro reviews every image (~$3-5)
- **Stage 2:** Claude Opus settles disputes only (~$5-7)

If interrupted (network drop, laptop sleep), resume:

```bash
python verify_labels.py --resume
```

**Output:** `growz_eval_workspace/golden_set/golden_set_verified.csv` with new columns:

| Column | Meaning |
|--------|---------|
| `gemini_verdict` | agree / disagree / uncertain |
| `gemini_proposed` | What Gemini thinks the disease is |
| `claude_verdict` | dataset_correct / gemini_correct / neither_correct / uncertain (disputes only) |
| `claude_proposed` | What Claude thinks (disputes only) |
| `final_verdict` | verified / relabeled / dataset_wrong / ambiguous / error |
| `final_disease` | The agreed-on label (use this as ground truth) |
| `final_confidence` | high / medium / low |

### Step 4. Filter to clean golden set

```bash
python -c "
import csv
rows = [r for r in csv.DictReader(open('growz_eval_workspace/golden_set/golden_set_verified.csv'))
        if r['final_confidence'] in ('high', 'medium')]
with open('growz_eval_workspace/golden_set/golden_set_clean.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f'Kept {len(rows)} rows')
"
```

You now have `golden_set_clean.csv` — typically ~230-260 images with double-verified labels. Use the `final_disease` column as ground truth.

---

## How verification works internally

### Stage 1: Gemini 3.1 Pro

For each image, Gemini receives:
- The image bytes
- The crop name (e.g. "cotton")
- The dataset label (e.g. "bacterial_blight")

Prompt asks: "Does this image match the proposed label?"

Response is structured JSON:
```json
{
  "verdict": "agree" | "disagree" | "uncertain",
  "proposed_disease": "what Gemini actually sees",
  "reasoning": "one sentence",
  "photo_quality": "good" | "medium" | "poor"
}
```

### Stage 2: Claude Opus (disputes only)

Only images where Gemini said `disagree` go to Claude. Claude receives:
- The image
- The original dataset label
- Gemini's proposed label
- Gemini's reasoning

Prompt asks: "Who is right — dataset, Gemini, or neither?"

Response:
```json
{
  "verdict": "dataset_correct" | "gemini_correct" | "neither_correct" | "uncertain",
  "proposed_disease": "correct disease name",
  "reasoning": "one sentence",
  "confidence": "high" | "medium" | "low"
}
```

### Final decision logic

Implemented in [`growz_eval/verification/decision.py`](growz_eval/verification/decision.py) as a pure function (no I/O), fully unit-tested.

| Gemini | Claude | → Final verdict | Confidence |
|--------|--------|-----------------|------------|
| agree | (not called) | **verified** | high |
| disagree | dataset_correct | **verified** | high |
| disagree | gemini_correct | **relabeled** | medium |
| disagree | neither_correct | **dataset_wrong** | medium |
| disagree | uncertain | **ambiguous** | low |
| uncertain | (not called) | **ambiguous** | low |
| ERROR | — | **error** | low |

Keep `verified`, `relabeled`, `dataset_wrong`. Drop `ambiguous` and `error`.

---

## Golden set distribution

300 images sampled with this distribution:

**By crop:**
- cotton: 80 (27%)
- grape: 80 (27%)
- tomato: 60 (20%)
- potato: 30 (10%)
- wheat: 25 (8%)
- other: 25 (8%)

**Within each crop:**
- healthy baseline: 20%
- common diseases: 60%
- rare/long-tail: 15%
- ambiguous (for "I don't know" testing): 5%

To change distribution, edit `CROP_DISTRIBUTION` in [`growz_eval/config.py`](growz_eval/config.py).

---

## CSV schema

After full pipeline, each row in `golden_set_clean.csv` has:

**Identity:**
- `id` — short hash like `a3f8b21c4d`
- `image_path` — path to image file

**Source (from datasets):**
- `crop` — cotton / grape / tomato / potato / wheat / other
- `source_label` — original folder name from dataset
- `source_dataset` — which dataset (ccdphd11, ldd_grape, etc.)
- `disease_tier` — healthy / common / rare / ambiguous

**Verification:**
- `photo_quality` — good / medium / poor (from Gemini)
- `gemini_verdict`, `gemini_proposed`, `gemini_reasoning`
- `claude_verdict`, `claude_proposed`, `claude_reasoning`, `claude_confidence`

**Final ground truth (USE THESE):**
- `final_verdict` — verified / relabeled / dataset_wrong
- `final_disease` — the correct disease name ← **use as ground truth**
- `final_confidence` — high / medium / low

---

## Using the golden set for backend eval

When your backend is ready, for each row in `golden_set_clean.csv`:

1. Send `image_path` + `crop` to your diagnosis endpoint
2. Receive `system_diagnosis` + `system_confidence`
3. Compare `system_diagnosis` against `final_disease`
4. Use LLM-as-judge (or string matching) to grade: correct / partially_correct / incorrect / not_attempted
5. Aggregate metrics: overall accuracy, per-crop accuracy, false confidence rate

Save each run as `results_YYYYMMDD_HHMM.csv` so you can track improvements over time:
- `baseline_v1.csv` — current LLM pipeline
- `after_quality_gate_v2.csv` — after adding photo quality check
- `after_rag_v3.csv` — after adding structured retrieval
- etc.

The delta between runs is what matters. Absolute numbers will lie due to domain gap with Uzbek photos — but **direction of change** is reliable.

---

## Costs

| Step | Time | Cost |
|------|------|------|
| Download datasets | 30-90 min | $0 |
| Build golden set | 2 min | $0 |
| Verify 20 (test) | 5 min | ~$0.50 |
| Verify 300 (full) | 40-60 min | ~$10 |

Total: ~$10-15 and 2-3 hours for the complete pipeline.

---

## Development

```bash
pytest -q          # run tests
ruff check .       # lint
ruff check --fix . # auto-fix lint
```

The codebase is organised so that:
- **Adding a new dataset source** → write a new `Downloader` subclass in [`growz_eval/datasets/downloaders.py`](growz_eval/datasets/downloaders.py).
- **Swapping the verification model** → write a new `GeminiReviewer` / `ClaudeReviewer` subclass in [`growz_eval/verification/reviewers.py`](growz_eval/verification/reviewers.py).
- **Changing the decision rules** → edit [`growz_eval/verification/decision.py`](growz_eval/verification/decision.py) and update its unit tests.

Secrets (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, Kaggle tokens) are read only from environment / `~/.kaggle/`, never written to files or logs. The logger applies a redaction filter that masks anything matching the key prefixes (`AIza...`, `sk-ant-...`, `KGAT_...`) just in case an error message contains one.

---

## Troubleshooting

**`kaggle: command not found`** — `pip install -e .` didn't complete or venv not activated.

**`401 Unauthorized` from Kaggle** — no credentials. Place `kaggle.json` or an `access_token` file in `~/.kaggle/` (chmod 600).

**`GEMINI_API_KEY not set`** — export the variable in the same shell where you run the script. Check with `echo $GEMINI_API_KEY`.

**Anthropic returns "invalid API key" but key looks right** — most common cause: no balance on the account. Add $5-10 at https://console.anthropic.com/settings/billing. Or skip Claude with `--gemini-only`.

**Script crashes mid-way** — run again with `--resume`. It will skip already-processed rows.

**Some datasets failed to download** — partial success is fine, golden set will be built from whatever succeeded. Re-run `build_golden_set.py` to retry failed ones.

**Verified set is too small (< 200 usable)** — increase initial `--size` to 400-500 to compensate for ambiguous drops.

---

## What's next (after you have the golden set)

1. **Build your backend** following the target architecture (Quality gate → Context builder → Retrieval → LLM → Confidence router → Feedback capture).
2. **Run baseline eval** on the golden set. Get a starting number.
3. **Improve one thing at a time.** Re-run eval after each change. Track delta.
4. **Start feedback loop in production.** Every real diagnosis saves (photo, response, user reaction). After 1-2 months you'll have real Uzbek labeled data — replace this open-source golden set with that one.

---

## Limitations to remember

- **Not Uzbek data.** Bangladeshi cotton + Italian grapes don't fully predict Uzbek production. Use this as a **starter**, not a final benchmark.
- **AI labels are ~85-90% as good as expert labels.** Critical decisions should still go through a real agronomist later.
- **Healthy / common / rare classification is heuristic.** It uses folder-name keywords. Refine `COMMON_DISEASES` in [`growz_eval/config.py`](growz_eval/config.py) for your priorities.
- **Symlinks vs copies.** By default images are symlinked from `data/` to `golden_set/images/`. To make the set portable, use `--copy`.
