# Reproduction guide

This document gives two routes.

1. A minimal verification route that checks the released artifacts against the paper numbers without making new API calls.
2. A fuller rerun route that uses the vendored release-local scripts and, where needed, your own API credentials.

## Step 0. Fetch the dataset from Hugging Face

The code repository does not ship the data; the companion dataset is on Hugging Face. Fetch it into `./data/` first so every later command finds its inputs at the expected release-relative paths.

```bash
pip install -r code/requirements.txt
pip install -U huggingface_hub
export HF_DATASET_REPO=anonymous-review/ai-vs-human-rubric-companion-data
./code/bin/00_fetch_data
```

The fetch wrapper writes to `./data/` and refuses to overwrite if `./data/` is already populated; pass `--force` to re-fetch. After this step, `data/paper_release/` and `data/canonical_full/` should exist as siblings of `code/`.

## Minimal verification route

From the release root:

```bash
python3 code/validation/validate_release_closure.py
python3 code/validation/validate_release_metrics.py --section all
```

This route verifies the three findings against the staged release artifacts.

- Finding 1 is recomputed from `data/paper_release/finding1/criterion_pairs/finding1_confirmed_pairs.json`.
- Finding 2 is recomputed from `data/paper_release/rubrics/rewrite/cascade_rewrite_audit.jsonl` together with the staged original and cascade judgement trees in `data/canonical_full/answer_eval/`.
- Finding 3 is recomputed from the staged coverage summaries, top-100 direct-check summaries, raw direct-check outputs, and same-branch normative-tendency files in `data/paper_release/finding3/`.

## Full rerun route

### 1. Build AI rubrics

```bash
./code/bin/01_build_ai_rubrics --pilot_only --models gpt54_openrouter
```

### 2. Build model responses

```bash
./code/bin/02_build_model_responses --pilot_only --models gpt54_openrouter
```

### 3. Score model responses under human and model rubrics

```bash
./code/bin/03_run_human_model_scoring --pilot_only --models gpt54_openrouter
```

### 4. Finding 1 pair artifact

```bash
./code/bin/04_run_finding1_pair_pipeline
```

This validates the shipped staged pair artifact. The exact upstream pair-consolidation builder is not part of the vendored minimal closure.

### 5. Finding 2 generality pipeline

The first-pass judge can be rerun directly:

```bash
LAB_OPENROUTER_KEY=... ./code/bin/05_run_finding2_generality_pipeline --run-round1
```

The final shipped cascade rewrite remains a staged derived artifact in this anonymous package. The exact upstream cascade-consolidation builder is not part of the minimal vendored closure, so the wrapper validates the released final artifact rather than pretending to rebuild it.

### 6. Finding 3 coverage pipeline

```bash
LAB_OPENROUTER_KEY=... ./code/bin/06_run_finding3_coverage_pipeline
```

To also rerun the direct LLM check on the selected cases:

```bash
LAB_OPENROUTER_KEY=... ./code/bin/06_run_finding3_coverage_pipeline --with-direct-check
```

The shipped tree contains the two coverage subdirectories the paper-facing analyses read directly: `data/paper_release/finding3/coverage/human_model_unique_t70_all/` and `data/paper_release/finding3/coverage/global_unique_t70/`. The third subdirectory, `data/paper_release/finding3/coverage/pooled_unique_t70/`, is created by `build_pooled_unique_criteria.py` on first run of stage 6 and is not required for the minimal verification route.

### 7. Rebuild validation summaries

```bash
./code/bin/07_build_release_summaries
```

## Notes on shipped staged artifacts

Two release-local files are treated as authoritative staged derived artifacts.

- `data/paper_release/finding1/criterion_pairs/finding1_confirmed_pairs.json`
- `data/paper_release/rubrics/rewrite/human_rubric_cascade_rewritten.jsonl`

The release package still contains enough data to verify all paper-facing headline numbers against those artifacts. The machine-checkable route is the validation script, not an unavailable hidden builder.
