# Tabular foundation models

## tabUM v1.1 comparison

`purple_tabum_v11` uses the same depth rows, 1,024-row training context and 256-row query batches as Kumo. Use one estimator, no fine-tuning, and native p05/p50/p95. The model stays local (CPU on macOS, CUDA when available). Only the pinned checkpoint is downloaded; well data is not sent to Hugging Face.

Use a **separate Python 3.13 environment**: upstream's pinned runtime conflicts with Kumo's pandas version. From the repository root install `./workstreams/purple_tabular_foundation_models/models/tabum_v11` and pytest in that environment. Set `PYTHONPATH=src` when running `python -m benchmark.check_model purple_tabum_v11`. Set `WMB_TEST_TABUM=1` for the real-weight pytest smoke test. Both [weights](https://huggingface.co/helmo/tabum-v1.1/tree/819c5927aeb8e574facc3d339a91e37bdec4bb9a) and [inference code](https://github.com/heldernoid/tabum/tree/d119a5d3e11d144cd9c75b8609586a076c32db1e) are pinned; retain the [MIT notice](models/tabum_v11/LICENSE).

## Kumo

`purple_kumo_tabular` uses NVIDIA's pretrained **Kumo-Tabular small regressor** locally. One row represents one depth sample: measured depth (metres) plus canonical non-target logs. There are no well identifiers, lagged features, sequence windows, or visible target values from the query well, even for gap tasks.

`fit` caches labeled examples; it does not fine-tune weights. With wells sorted by dataset and name, it allocates up to 1,024 finite-label rows round-robin, uniformly spanning each well's eligible sample indices. Only curves present in that context become features. Absent query curves are NaN; Kumo handles missingness. Prediction reuses that context for batches of 256 rows in depth order. The baseline uses one estimator, seed 2020, and returns Kumo's native 5th, 50th and 95th percentiles. CUDA is selected when available; macOS uses CPU.

## Run from the repository root

Use a separate Python 3.12 environment with the benchmark dependencies installed. Install this workstream's dependencies with `pip install ./workstreams/purple_tabular_foundation_models`. The first fit downloads the small regression checkpoint from Hugging Face to the local cache; registry discovery does not download anything.

```sh
PYTHONPATH=src python -m benchmark.check_model purple_kumo_tabular
PYTHONPATH=src python -m benchmark.run --model purple_kumo_tabular --test synthetic_rhob_missing
PYTHONPATH=src python -m pytest -q workstreams/purple_tabular_foundation_models/tests
```

Set `WMB_TEST_KUMO=1` to include the real-checkpoint smoke test. For a FORCE comparison, first validate the local dataset paths, then run `--test rhob_missing` for both Kumo and the time-series contender. Keep the test definition unchanged so both use the same target, held-out wells, mask, and metrics. The synthetic smoke benchmark verifies integration, not real-well performance.
