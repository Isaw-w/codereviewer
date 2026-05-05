# Assets and licenses

## Code license

The code shipped in this bundle inherits the repository's `LICENSE` file. In the audited release snapshot, that license is MIT.

## Benchmark inputs and released data

This bundle includes benchmark cases, human rubrics, model-generated rubrics, model responses, judgement outputs, and derived analysis artifacts that were used in the paper. These are released here as research artifacts for anonymous reproducibility.

The benchmark-side inputs originate from the MoReBench project and should be used with the same scholarly citation practice expected for the original project.

## API-backed model outputs

The release package includes staged outputs from API-backed model runs. Re-running those stages requires your own credentials and remains subject to the terms of the relevant model provider and gateway.

The release-local wrappers and scripts expect `LAB_OPENROUTER_KEY` for OpenRouter-backed calls.

## Third-party software dependencies

The local dependency snapshot is in `code/requirements.txt`.

At the audited snapshot, the explicit Python requirements are:

- `pandas`
- `anthropic`
- `openai`
- `tqdm`
- `numpy`
- `datasets`

## Non-code assets

No external image packs, web fonts, or proprietary media assets are required for the core reproduction path in this release package.
