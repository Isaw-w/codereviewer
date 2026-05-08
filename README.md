# Anonymous code release for the AI-vs-human rubric study

This repository holds the runnable code, documentation, and machine-readable manifests for an anonymous NeurIPS submission. The companion dataset (model rubrics, model responses, judgement outputs, derived analysis artifacts) is published as a separate Hugging Face dataset so that this repository stays small and easy to read.

## Repository layout

- `code/` runnable script closure, stage wrappers under `code/bin/`, validation scripts under `code/validation/`, prompt helpers, and a local `requirements.txt` snapshot.
- `docs/` reproduction notes, a data map, asset and license notes.
- `manifests/` machine-readable inventory, provenance, SHA-256 hashes, validation reports, and the expected paper-facing metrics.
- `LICENSE` MIT.
- `data/` is **not** in this repository. It comes from Hugging Face. See "Quick start" below.

## Quick start

1. Clone this repository and create a fresh Python environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r code/requirements.txt
   pip install -U huggingface_hub
   ```

2. Fetch the companion dataset from Hugging Face into `./data/`. The fetch script writes to the exact release-relative path the rest of the pipeline expects.

   ```bash
   export HF_DATASET_REPO=forreview43/ai-vs-human-rubric-companion-data
   ./code/bin/00_fetch_data
   ```

   `HF_DATASET_REPO` is the dataset repo id printed in the paper supplementary link. By default, the fetch script pins `HF_REVISION` to `c165536f6b4393df4d71e3897bc5f1fb23f6e4b5`, the verified dataset commit that includes the complete Finding 2 and Finding 3 paper-release directories. Set `HF_REVISION=main` only if you intentionally want the latest dataset state. The `huggingface-cli` does not require login for a public anonymous dataset.

3. Verify the headline numbers without any new API calls.

   ```bash
   python3 code/validation/validate_release_closure.py
   python3 code/validation/validate_release_metrics.py --section all
   ```

Both scripts write reports under `manifests/validation/`.

## Reproducing each finding

See `docs/REPRODUCE.md` for the full step-by-step rerun route, including the API-backed stages (rubric generation, response generation, scoring, the Finding 1 rubric-as-response capture check, Finding 2 coverage builder, and Finding 3 generality check).

## Anonymity

This package is intended to support double-blind review. The repository contains no author names, institutional identifiers, or external usernames. The companion Hugging Face dataset is hosted under a dedicated anonymous account.
