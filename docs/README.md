# Anonymous release package for the AI-vs-human rubric study

This directory is the standalone anonymous release package for the paper. It is organized around one requirement: the released data, code, manifests, and documentation should be understandable and checkable without relying on the rest of the working repository.

The package has four top-level parts.

- `code/` contains the runnable script closure, thin stage wrappers, validation scripts, prompt helpers, and a local `requirements.txt` snapshot.
- `data/canonical_full/` contains the full canonical rerun corpus used in the paper. This is the largest part of the package.
- `data/paper_release/` contains the paper-facing inputs, summaries, staged raw outputs, and appendix-support artifacts.
- `docs/` and `manifests/` contain the human-readable reproduction notes and the machine-readable inventory, provenance, counts, hashes, and validation reports.

Two artifacts are shipped here as authoritative staged derived artifacts rather than being rebuilt from a vendored consolidator script.

1. `data/paper_release/finding1/criterion_pairs/finding1_confirmed_pairs.json`
2. `data/paper_release/rubrics/rewrite/human_rubric_cascade_rewritten.jsonl`

In both cases, the paper-facing numbers are still machine-checkable from the release package itself. `code/validation/validate_release_metrics.py` recomputes the Finding 1, Finding 2, and Finding 3 headline numbers directly from the staged release data and writes a validation report under `manifests/validation/`.

## Quick start

Create a fresh environment and install the local requirements snapshot:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

Minimal verification, with no new API calls:

```bash
python3 code/validation/validate_release_closure.py
python3 code/validation/validate_release_metrics.py --section all
```

Stage wrappers are provided under `code/bin/`.

- `01_build_ai_rubrics`
- `02_build_model_responses`
- `03_run_human_model_scoring`
- `04_run_finding1_pair_pipeline`
- `05_run_finding2_generality_pipeline`
- `06_run_finding3_coverage_pipeline`
- `07_build_release_summaries`

The first three wrappers launch the release-local canonical runners directly. The Finding 3 wrapper can rerun the coverage pipeline with your own API key. The Finding 1 and Finding 2 wrappers validate the shipped staged artifacts because the exact upstream pair-consolidation and cascade-consolidation scripts are not part of the minimal vendored closure.

## Data scale

The release package includes the full canonical rerun tree. In the audited release snapshot, `data/canonical_full/` contains 77,389 files and is about 4.9 GB. `data/paper_release/` contains the smaller paper-facing subset used for the main tables, figures, and appendix support.

## API-backed reruns

Some stages require model API access.

- Embedding and Finding 3 coverage builders use `LAB_OPENROUTER_KEY`.
- The release-local generation and scoring runners also expect `LAB_OPENROUTER_KEY` when they make live calls.
- All wrappers are resume-safe in the sense implemented by the underlying vendored runners.

For the minimal verification path, no API key is required.
