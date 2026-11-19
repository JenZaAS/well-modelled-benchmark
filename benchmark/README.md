# Benchmark (runner + scoreboard)

**Team:** Benchmark

One runner scores every model the same way: same wells, same split, same metrics.

```
python -m benchmark.run --list                                   # which contenders exist
python -m benchmark.run --dataset synthetic --target RHOB         # smoke test, no data needed
python -m benchmark.run --dataset force2020 --target DTS          # real run (data in FH2026_DATA/force2020)
python -m benchmark.scoreboard --write                           # one table from all results
```

How it works:
1. `registry.py` imports every `workstreams/*/contender.py` and reads its `CONTENDERS` list.
   A broken workstream is skipped with a warning; it never stops the others.
2. `run.py` splits by whole wells, calls `fit(train_wells, target)`, then
   `predict(test_well_without_target, target)`, and times both (speed metric).
3. Scores go to `results/<contender>/<dataset>/<target>/scores.json` (gitignored).
4. `scoreboard.py` collects them into `results/summary/scoreboard.md` (small; may be committed).

TODO (Benchmark): gap-filling scenario with `wmb.splits.gap_mask`, lithology task with the
FORCE penalty matrix, per-well plots, and the "fill every real gap in the whole database" speed test.
Metric definitions: `docs/metrics.md`.
