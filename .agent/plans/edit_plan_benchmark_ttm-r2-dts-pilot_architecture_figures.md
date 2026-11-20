# Proposed edit plan: model architecture figures

Branch: benchmark/ttm-r2-dts-pilot. Four conceptual edits, in order.

1. Define original SVG schematics from documented contender behavior and model architecture. Use a labeled prediction-workflow schematic when only the adapter workflow is documented; never guess internal layers. Resolve historical aliases to the same figure.
2. Extend `modelDetailHtml` in the benchmark viewer to append the corresponding accessible figure and caption immediately after the existing Input, Method and Output list. Unknown IDs retain an explicit unavailable state. This depends on the figure catalog.
3. Add responsive figure styles in the live viewer. Preserve existing links, filtering and score displays. Before: the cards end the model page. After: a diagram below them shows structure, e.g. parallel trees feeding an average versus successive trees adding corrections.
4. Extend detail-page tests and the browser smoke test to check placement, accessible labeling, distinct methods, fallback, and small-screen layout. Run the benchmark viewer tests.

Pending approval before implementation.
