# Tabular foundation models

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
