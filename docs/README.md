# Anonymous release package for the AI-vs-human rubric study

This directory documents the standalone anonymous release package for the paper. In the anonymous code repository, the `data/` directory is not stored in git; reviewers fetch it from the companion Hugging Face dataset with `code/bin/00_fetch_data`. After that fetch, the local release root has the layout described below. The goal is that the released data, code, manifests, and documentation are understandable and checkable without relying on the rest of the authors' working repository.

The package has four top-level parts.

- `code/` contains the runnable script closure, thin stage wrappers, validation scripts, prompt helpers, and a local `requirements.txt` snapshot.
- `data/canonical_full/` is fetched from the companion dataset and contains the full canonical rerun corpus used in the paper. This is the largest part of the package.
- `data/paper_release/` is fetched from the companion dataset and contains the paper-facing inputs, summaries, staged raw outputs, and appendix-support artifacts.
- `docs/` and `manifests/` contain the human-readable reproduction notes and the machine-readable inventory, provenance, counts, hashes, and validation reports.

Two artifacts are shipped here as authoritative staged derived artifacts rather than being rebuilt from a vendored consolidator script.

1. `data/paper_release/finding3/criterion_pairs/finding1_confirmed_pairs.json`
2. `data/paper_release/rubrics/rewrite/human_rubric_cascade_rewritten.jsonl`

In both cases, the paper-facing numbers are still machine-checkable from the fetched release data. `code/validation/validate_release_metrics.py` recomputes the Finding 1, Finding 2, and Finding 3 headline numbers directly from the staged release data, including the seven-model 100-case Finding 1 rubric-as-response capture check, and writes a validation report under `manifests/validation/`.

## Quick start

Create a fresh environment, install the local requirements snapshot, and fetch the companion dataset into `./data/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
pip install -U huggingface_hub
export HF_DATASET_REPO=forreview43/ai-vs-human-rubric-companion-data
./code/bin/00_fetch_data
```

The fetch wrapper defaults to the verified dataset commit `c165536f6b4393df4d71e3897bc5f1fb23f6e4b5`, which includes the complete Finding 2 `coverage`, `direct_check`, `normative_dimension_labels`, and `normative_tendencies` directories, plus the complete Finding 3 `criterion_pairs`, `cascade_rescoring`, and `generality_validation` directories. You can override this with `HF_REVISION=main` if you want the latest dataset state instead of the pinned review snapshot.

Minimal verification, with no new API calls:

```bash
python3 code/validation/validate_release_closure.py
python3 code/validation/validate_release_metrics.py --section all
```

Stage wrappers are provided under `code/bin/`.

- `01_build_ai_rubrics`
- `02_build_model_responses`
- `03_run_human_model_scoring`
- `04b_run_finding1_rubric_capture`
- `05_run_finding2_coverage_pipeline`
- `06a_run_finding3_pair_pipeline`
- `06b_run_finding3_generality_pipeline`
- `07_build_release_summaries`

The first three wrappers launch the release-local canonical runners directly. The Finding 1 rubric-as-response wrapper can regenerate the serialized rubric inputs without API calls and can rerun the judge with your own OpenRouter key. The Finding 2 wrapper can rerun the coverage pipeline with your own API key. The Finding 3 pair and generality wrappers validate the shipped staged artifacts because the exact upstream pair-consolidation and cascade-consolidation scripts are not part of the minimal vendored closure.

## What Is Reproducible

There are three levels of reproduction.

1. **No-API verification.** After fetching the Hugging Face dataset, the commands above recompute every paper-facing headline number for Findings 1--3 from released artifacts. This is the recommended reviewer path.
2. **API-backed reruns.** Rubric generation, response generation, judge calls, embedding-backed coverage, and the Finding 1 rubric-as-response judge can be rerun with the reviewer's own `LAB_OPENROUTER_KEY`.
3. **Staged derived artifacts.** The final matched-pair file and final cascade-rewritten human rubric are shipped as authoritative staged artifacts. The release validates and scores them, but does not vendor the older upstream consolidation scripts that originally produced those exact files.

## Data scale

After fetching the companion dataset, the local release package includes the full canonical rerun tree. In the audited release snapshot, `data/canonical_full/` contains 77,375 files and is about 4.0 GB. `data/paper_release/` contains the smaller paper-facing subset used for the main tables, figures, and appendix support.

## API-backed reruns

Some stages require model API access.

- Embedding and Finding 2 coverage builders use `LAB_OPENROUTER_KEY`.
- The release-local generation and scoring runners also expect `LAB_OPENROUTER_KEY` when they make live calls.
- All wrappers are resume-safe in the sense implemented by the underlying vendored runners.

For the minimal verification path, no API key is required.
