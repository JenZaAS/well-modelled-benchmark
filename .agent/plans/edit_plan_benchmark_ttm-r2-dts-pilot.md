# Register the scored TTM r2 variants

Scope: red workstream only. Six planned file edits on `benchmark/ttm-r2-dts-pilot`; no shared runner changes.

1. Extend the TTM contender module with two explicitly named classes: the DTS-only combined pilot and its frozen comparison. Match the names used in the published scores. Bind the frozen model to the pinned IBM revision and the tuned model to its local checkpoint (fail clearly if absent, never fall back to frozen).
2. Register the classes via a separate optional key in the red model registry without changing existing defaults.
3. Add target/input and licence metadata for the tuned variant.
4. Add matching metadata for the frozen reference.
5. Add focused tests for registry discovery, target support, and checkpoint selection including the missing-weight case.
6. Document how to activate and load the variants, which checkpoint was scored, and how to obtain or regenerate the tuned artifact outside git. Verify both registered models on the matching FORCE tests before pushing registration changes.

The three individual-well checkpoints were timing experiments, not scored benchmark entries. Do not silently publish scores for them or commit weights or predictions.
