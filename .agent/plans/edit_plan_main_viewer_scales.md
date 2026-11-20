# Proposed edit plan: outlier-resistant comparison charts

Scope: shared viewer presentation only; benchmark data, rankings, scoring and individual well-log tracks stay unchanged. Three implementation edits, in order:

1. In `viewer/benchmark_quality.js`, add a pure display-scale helper for non-negative metrics. Keep zero and ordinary ranges linear. When the largest value exceeds four times the next-largest positive value, reserve the last 20% of the axis for the extreme tail. Handle empty, singleton, tied, zero and nonfinite inputs explicitly.
2. In `viewer/benchmark.js`, use the mapping consistently in the accuracy and sharpness cards, by-property comparison bars, and trade-off scatter (points, ticks, baseline guides and Pareto line). Identify compressed axes visibly, retaining original numbers and units. Do not transform calibration or speed. Before: one extreme score defines the entire linear scale. After: the main cluster remains linear and readable, and the outlier remains visible at its true labeled value.
3. In `tests/test_benchmark_quality.js`, cover ordinary ranges, an isolated outlier, degenerate inputs, monotonic mapping and rendering labels. Run the viewer JavaScript tests and inspect a FORCE comparison.

Await approval before editing the shared viewer.
