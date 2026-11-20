# Proposed edit plan — Benchmark model navigation (main)

Working with: the shared live viewer and its JavaScript tests. Five conceptual edits, in order. No workstream, scoring, or results changes.

1. Model metadata: extend the `modelName` and `modelFamily` helpers and add concise descriptions and input → method → output stages verified against model documentation. Before: all blue contenders are labelled GBT. After: linear and random forest have their own types; tabular and time-series foundation models are distinct. Keep raw IDs for fetching, routing, and matching. Unknown models get a readable label and an honest metadata fallback.
2. Filtering (depends on 1): add an All-by-default model type control in `build` and apply it in `renderView`, missing-score lists, and well-log selectors. Compute run-wide comparisons before filtering so hiding the baseline doesn't change normalized scores, ranks, or colors. Handle empty selections explicitly.
3. Linked highlighting (depends on 1): mark summary rows, legends, chart marks, and property-grid rows with the same model ID. Use delegated hover and keyboard-focus events to highlight matching elements without rerendering or changing scores. Keep partial-coverage indicators visible.
4. Model page (depends on 1, 3): link model labels to a distinct hash-routed detail view with a short explanation, family, and accessible architecture diagram. Support deep links, browser Back, and return to the previous benchmark selection. Preserve existing benchmark routes.
5. Verification (depends on 1–4): extend the benchmark UI tests for names, aliases, taxonomy, filtering with a hidden baseline, model identity across panels, and navigation; run existing benchmark JavaScript tests and smoke-test interaction in the live viewer where available.

Approval requested before implementing edit 1.
