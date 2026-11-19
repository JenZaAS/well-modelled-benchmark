# PROPOSED EDIT PLAN

Working with: new `workstreams/purple_tabular_foundation_models/` workstream on `main`.
Total planned edits: 7 files. No changes to the common runner or other teams' contenders.

## MAKING EDITS

Before: no tabular foundation contender. After: the registry discovers `purple_kumo_tabular`; optional inference dependencies and model weights load only when needed. Use NVIDIA's `structured-data-models` package and the regression weights from <https://huggingface.co/nvidia/Kumo-Tabular> (OpenMDW 1.1); do not substitute a different estimator on failure.

## Edit sequence

1. Workstream `pyproject.toml`: verify and pin a compatible inference package version and record runtime requirements.
2. `models/kumo_tabular/contender.py`: implement `BaseContender.fit` to freeze a numeric feature schema and build a bounded, reproducible context balanced across training wells and sampled across depth. `predict` uses the official Kumo regression API on bounded query batches and returns a depth-aligned `Prediction`. One row per depth sample: measured depth and available permitted non-target curves, no well ID, target-derived features, or sequence windows. Handle missing columns consistently; never infer preprocessing from hidden labels.
3. `models/kumo_tabular/model.yaml`: declare supported targets, curve inputs, and weight licence to match the wrapper.
4. `models/kumo_tabular/LICENSE.md`: distinguish weight and inference package terms; link upstream notices.
5. Workstream `contender.py`: expose the model through `CONTENDERS`, without loading weights at discovery.
6. Workstream `tests/test_kumo_tabular.py`: offline contract tests for schema, exclusion of target, context selection, missing inputs, chunk alignment and registry discovery; separate real-weight smoke test from mock-backed tests.
7. Workstream `README.md`: document setup, context design and limits, what `fit` does, and the same benchmark test to run against time-series contenders. Validate offline tests first, then a synthetic regression smoke run with real weights if accessible; distinguish skipped runs from successful inference.

Dependencies: confirm API in step 1 before step 2; declarations and tests follow implementation. Await approval before implementation edits.
