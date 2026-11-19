# Proposed edit plan: shared Benchmark tab

Four planned edits, in order. No changes to model inputs, scoring or the red dashboard.

1. In `benchmark/run.py`, extend the per-well prediction payload with optional display-only GR, CALI and RDEP tracks on the task depth grid when available. Do not include target values as context. Preserve existing fields.
2. In `viewer/benchmark.js`, prefer saved context and keep `/api/well` as the legacy fallback; show when context is unavailable. Use each series' own depth grid for selection and gap handling. In the same plot, pan with the wheel or drag, zoom around the pointer with modified wheel/pinch, provide zoom buttons and full-depth reset, and use sample spacing instead of a whole-well tick interval as the zoom limit.
3. Extend the runner's prediction-payload tests for optional, absent and target curves and depth alignment.
4. Extend `tests/test_benchmark_ui.js` for wheel/zoom bounds, different depth grids, context source selection and model switching. Run the viewer and runner tests; refresh existing synthetic red artifacts via the standard synthetic benchmark and verify both contenders' saved context.

Before: context depends on a separate dataset lookup and zoom stops at a tick-derived span. After: available context travels with new results and all displayed tracks share a navigable depth window without assuming identical sample indices. Older runs without a resolvable local well require regeneration for embedded context.