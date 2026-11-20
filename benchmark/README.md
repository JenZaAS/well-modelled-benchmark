# Benchmark (runner + scoreboard)

**Team:** Benchmark

One runner scores every model the same way: same wells, same split, same tasks, same metrics.
The contract (tasks in, predictions out) is in [`docs/benchmark.md`](../docs/benchmark.md).
To add a model, see [Build your model](../docs/build-your-model.md).

```
python -m benchmark.run --list                                   # which contenders exist
python -m benchmark.check_model template_linear                  # does a model meet the contract? (no data needed)
python -m benchmark.run --list-tests                             # YAML test definitions (benchmark/test_defs/)
python -m benchmark.run --model template_linear --test synthetic_rhob_missing   # no data needed
python -m benchmark.run --model template_linear --test rhob_missing             # needs FORCE demo data
python -m benchmark.run --model template_linear --test all
python -m benchmark.new_test my_test                             # add your own test
python -m benchmark.check_test my_test
python -m benchmark.scoreboard --write                           # one table from all results
```

The older flags `--dataset`, `--scenario` and `--target` still work, as does `python -m benchmark.make_tasks`.

Options of `run`: `--test` (name from `benchmark/test_defs/`, or `all`; replaces `--dataset/--scenario/--target`, which
must agree with the def if given; see [`docs/tests.md`](../docs/tests.md)), `--model` (same as `--contenders`; comma list or `all`), `--scenario`
(`missing_curve` default, `gap`, `gap_10`, `gap_50`, `gap_100`, `all`), `--target` (a core curve, or `all` =
RHOB, DTC, DTS; gap scenarios need one of those), `--cache` (task cache), `--dataset`, `--seed`.

How it works:
1. `registry.py` imports every `workstreams/*/contender.py` and reads its `CONTENDERS` list.
   A broken workstream is skipped with a warning; it never stops the others.
2. `make_tasks.py` turns a committed spec (`task_specs/<dataset>.yaml`: seed, targets, gap lengths, wells,
   gap windows) into `wmb.contract.Task` files cached outside git (`--cache`, `$WMB_TASK_CACHE`,
   `<cache_dir>/tasks`, or `<repo>/cache/tasks`). Listed gap windows are never re-rolled (`--reroll` does).
   `--write-spec` writes the resolved wells and windows to `<dataset>.windows.yaml`.
   `benchmark.load_tasks(dataset, cache=..., scenario=..., target=...)` reads them.
3. `run.py` calls `fit(train_wells, target)`, then `predict_task(task)` per test task, validates each
   prediction with `wmb.contract.validate_prediction`, and times load, fit and predict (per well and batched).
   A model that raises or returns an invalid prediction is recorded in `failure.json`; the run continues.
4. Scores go to `results/<contender>/<dataset>/<target>/<scenario>/scores.json` (shared through git; see
   `results/README.md`). For
   `missing_curve` a copy also goes to `results/<contender>/<dataset>/<target>/scores.json`.
   Each run also writes `results/runs/<timestamp>_<dataset>.json` and `.csv`. Each file has a `test` field (see `test_def.py`).
5. `scoreboard.py` collects them into `results/summary/scoreboard.md` (small; may be committed).

TODO (Benchmark): lithology task with the FORCE penalty matrix, per-well plots, and the "fill every real gap
in the whole database" speed test.
Metric definitions: `docs/metrics.md`.
