# Benchmark

Every team provides a **model**. This repo's benchmark gives every model
the same tasks, hides the same depths, and scores the answers the same way.

Start with the short guide: [**Build your model**](build-your-model.md).
You implement the fixed input/output contract, then run:

```bash
python -m benchmark.check_model template_linear
python -m benchmark.run --model template_linear --test all
python -m benchmark.scoreboard
```

The Python types are in `wmb.contract` and `wmb.contender`. See also
[`src/wmb/INTERFACE.md`](../src/wmb/INTERFACE.md), [`tests.md`](tests.md)
and [`metrics.md`](metrics.md). The older `--dataset` / `--scenario` /
`--target` flags still work.

## What a model receives and what it must return

The runner builds a **Task** and calls `predict_task(task)` (or, for older
models, `predict(well, target)`).

A Task contains:

| Field | Meaning |
|---|---|
| `well_id` | Well name |
| `depth` | Measured depth in metres, increasing, one value per sample |
| `inputs` | pandas DataFrame of **canonical** curves (and a `DEPTH` column). Values may be NaN. The target curve is **not** a column, except that in the gap scenario the observed target is given separately (see below). |
| `target` | Canonical name: `RHOB`, `DTC` or `DTS` |
| `predict_mask` | Boolean array, same length as `depth`. `True` where the model must predict. |
| `target_observed` | Same length as `depth`. NaN on `predict_mask`; elsewhere the measured target (gap scenario only). `None` for a whole missing curve. |
| `scenario` | `"missing_curve"` or `"gap"` |
| `gap_length_m` | `10`, `50` or `100` for a gap task; `None` otherwise |
| `dataset`, `split`, `meta` | Provenance (e.g. FORCE `leaderboard` / `hidden`) |

A **Prediction** must use the same depth index (same length as `task.depth`):

| Field | Required? | Meaning |
|---|---|---|
| `p50` (or `mean`, they are aliases) | Yes | Point prediction, in the canonical unit of the target |
| `p05` and `p95` | Optional | 5th and 95th percentiles (a 90% interval) |
| `std` | Optional | One-sigma uncertainty. If you do not give `p05`/`p95`, the runner treats the interval as Gaussian: 90% is `p50 ± 1.64485·std`, 50% is `p50 ± 0.67449·std`. |
| `quantiles` | Optional | Dict such as `{0.05: ..., 0.50: ..., 0.95: ...}` |
| `bin_edges` and `bin_probs` | Optional, together | A binned forecast: `K-1` increasing internal edges, and an `[n_depth, K]` array of probabilities. Bin `k` covers `[edges[k-1], edges[k])`, the two end bins are open, and a value on an edge belongs to the higher bin. A row is either `K` probabilities that sum to 1 or all NaN (no forecast at that depth). It is scored as `RPS_bins`, and the viewer draws it as a heatmap. |

Units are the canonical ones from `wmb.datasets.canonical_curves.yaml`
and `wmb.curves.UNITS`: **RHOB** in g/cm3 (g/cc), **DTC** and **DTS** in
us/ft. Do not convert units yourself; the loader already has.

The runner validates the prediction before scoring: same length as
depth, `p05 ≤ p50 ≤ p95` where all are finite, `std ≥ 0` where finite. Bin probabilities
must lie in [0, 1], and each finite row must sum to 1 within 0.001; the runner then renormalises it.
NaN is allowed where you cannot predict. Clear `ValueError` messages
explain a failed check. A broken model is skipped and recorded; it does
not stop the other models.

## Targets

For this first cut we predict three curves:

| Target | Unit | Notes |
|---|---|---|
| RHOB | g/cm3 | Bulk density |
| DTC | us/ft | Compressional slowness |
| DTS | us/ft | Shear slowness (scarcer) |

Allowed **inputs** are the other canonical curves that are present, plus
depth: GR, RDEP, RMED, CALI, NPHI, PEF, and RHOB / DTC / DTS when they
are not the target. A model must tolerate missing (NaN) inputs, and
must declare `required_inputs` and `optional_inputs` (the older
`inputs` list is treated as required). If a required curve is absent
from a well, still return a Prediction (use NaN if you cannot predict);
do not crash.

## Scenarios

**A. Whole missing curve (`missing_curve`).** The target is hidden on
the entire held-out well. `target_observed` is `None`. Score only at
depths where a measured target value exists.

**B. Gap filling (`gap`).** Inside sections where the target is
measured, the benchmark hides one contiguous interval of **10 m**, **50 m**
or **100 m**. The rest of the measured target is visible in
`target_observed`. Each length is a separate task, so scores can be
compared by gap size. Gaps are placed with a documented seed, fully
inside a measured interval; they never start in a real hole.

Lithology classification and a "fill every real gap" speed test are
later work; they are not in this cut.

Gap tests still show the target above and below the gap, so a univariate
gap-filler can score well. Whole-well tests hide the target in the entire
well, so the model needs other logs or a prior. Do not compare scores
across those two scenario types.

## Splits

Always split by **whole wells**. No sample from a test well may appear
in training.

For FORCE 2020 we use the official split: the 98 **train** wells for
`fit`, the **leaderboard** and **hidden** wells for testing. The exact
well names, seeds and gap windows are in the committed files
`benchmark/task_specs/force2020.yaml` and `force2020.windows.yaml`. Everyone uses those files, so
every model sees the same tasks.

```bash
python -m benchmark.make_tasks --dataset force2020
```

writes the actual Task objects (arrays included) into the local cache
(`cache_dir/tasks/…`, gitignored). If the yaml already lists wells and
gap windows, that list is used and is not re-rolled. If a listed well
is missing from your data, the command fails clearly.

Score only where measured truth is finite **and** the model returned a
finite `p50`. The number of scored samples is reported as `n`.

## Metrics

Defined in `wmb.scoring` and summarised in [`metrics.md`](metrics.md).
All accuracy numbers are in the target's canonical unit.

| Family | What we report | Better |
|---|---|---|
| **Accuracy** | MAE, RMSE, R² of `p50` vs truth | lower MAE/RMSE, higher R² |
| **Uncertainty** | Coverage of the 90% interval, and of the 50% interval | close to 0.90 and 0.50 |
| **Sharpness** | Mean width of the 90% interval (`p95 − p05`, or the Gaussian equivalent) | narrower, **only if coverage is close to 90%** |
| **Optional** | CRPS (Gaussian when only `std` is given; otherwise from quantiles when available) and pinball at 5/50/95% | lower |
| **Binned forecasts** | `RPS_bins`, the ranked probability score: the mean over the bin edges of (P(y < edge) − 1[y < edge])², averaged over the scored samples that have a binned forecast. `n_bins` counts those samples, `bin_cover` is their share of `n`, and `bin_edges` records the edges. Compare `RPS_bins` only between rows with the same edges. | lower |
| **Speed** | `fit_s`, `load_s` (constructing the model), `predict_s` (all test wells), `predict_s_per_well`, `predict_s_batch` (one batch of N wells), `samples_per_s`. Hardware (`machine`, Python, CPU count) is recorded. | lower time |

Point-only models (no `p05`/`p95` and no `std`) still get accuracy and
speed. Uncertainty and sharpness are left blank and the scoreboard
marks them as point-only (`has_uncertainty: false`). The viewer's Benchmark tab
turns these into good / ok / poor with the rules in
[benchmark_quality.md](benchmark_quality.md). Each row also records the git commit
(`git_commit`, `git_commit_full`) when git is available.

Older names `cover_1sigma` and `cover_2sigma` (mean ±1σ / ±2σ, ideals
0.683 and 0.954) are still computed when `std` is present, so earlier
results stay comparable.

## How to run

```bash
python -m benchmark.run --list
python -m benchmark.run --list-tests
python -m benchmark.run --model template_linear --test synthetic_rhob_missing
python -m benchmark.run --model template_linear --test rhob_missing
python -m benchmark.run --model template_linear --test all
python -m benchmark.scoreboard
python -m benchmark.scoreboard --write
```

`--model` and the older `--contenders` flag are the same thing (a comma
separated list, or `all`). `--test` is the usual way to choose what to
run (see [tests.md](tests.md)). The older `--dataset` / `--scenario` /
`--target` flags still work; the default dataset is `synthetic` (no real
data needed) and the default scenario is `missing_curve`.

The runner builds a fresh model and times `fit` for every (model, target,
scenario), so nothing seen in one gap length can leak into another, then
predicts every matching task. Gaps are depth-continuous (no depth hole
inside the interval). If one model raises, it is recorded as a failure and the
next model runs. A workstream whose `contender.py` cannot be imported
is skipped with a warning, as before.

## Make your own test

A test is one YAML file in `benchmark/test_defs/` (dataset, target, scenario, wells). Create, check and run one:

```bash
python -m benchmark.new_test my_test
python -m benchmark.check_test my_test
python -m benchmark.run --model template_linear --test my_test     # or --test all
```

See [`tests.md`](tests.md) for the schema and the shipped tests.

## Where results go

Per-model files (scores are shared through git, prediction files only for FORCE 2020 and synthetic; see
[results/README.md](../results/README.md)):

```
results/<model>/<dataset>/<target>/<scenario>/scores.json
```

With `--test`, these files go one level deeper, under `<scenario>/<test>/`. Each also carries a `test` field
(the test definition name, or a generated one such as `synthetic_rhob_missing` for the old flags).
For `gap`, scenario is `gap_10`, `gap_50` or `gap_100`. Each run also
writes a combined `results/runs/<timestamp>_<dataset>.json` (shared) and `.csv`
(local only). `python -m benchmark.scoreboard` prints one table from
all `scores.json` files, and `--write` updates
`results/summary/scoreboard.md` (small; may be committed).

Next to `scores.json`, each scored run writes one **prediction file** for the viewer's Benchmark tab (not used for
scoring): per test well the depth, measured target, p50, the 90% band and the hidden intervals, rounded to 4 decimals.
A model that returns a binned forecast also gets `"bins": {"edges", "rows", "probs"}`: the internal edges, the depth
indices that have a forecast, and one row of probabilities per such depth (3 decimals). The Benchmark tab's Well logs
view then offers **Uncertainty: Bin heatmap**. Its darkest colour is the well's largest bin probability (stated in
the legend), and hovering shows the median, the 80 % range and the likeliest bin at that depth.
Two formats hold the same information, and the viewer shows either one identically:

| Format | File | Use |
|---|---|---|
| `json` (default) | `predictions.json` | readable in any editor or tool |
| `npz` | `predictions.npz` | about 3x smaller and about 3x faster to write than JSON; the viewer reads one well without loading the whole run (needs NumPy) |

Choose with `--predictions json|npz`, the environment variable `WMB_PREDICTIONS_FORMAT`, or `predictions_format=`
on `benchmark.run.run_scenario` / `run_model` / `run_tests`. Writing one format removes the other from that folder.
Layout and rounding: `src/wmb/prediction_files.py`.

## Adding a model

Follow [**Build your model**](build-your-model.md). In short: copy
[`workstreams/_template_model/`](../workstreams/_template_model/) to
`workstreams/<team>/models/<name>/`, fill `model.yaml` and `LICENSE.md`,
register the class in your team's `CONTENDERS`, then:

```bash
python -m benchmark.check_model <name>
python -m benchmark.run --model <name> --test all
```

Implement `fit(train_wells, target)` and `predict_task(task)` (the older
`predict(well, target)` still works; the runner builds a `Well` from the
Task, including `target_observed` for gaps). Do not edit `src/wmb/` or
`benchmark/` to make your model pass; ask in the team chat if the
contract needs a change. The optional `models/` layout is also in
[moving-models.md](moving-models.md).
