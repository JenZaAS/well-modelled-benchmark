# Benchmark tests (YAML test definitions)

To add a model and run these tests, see [Build your model](build-your-model.md).

A **test** is one YAML file in `benchmark/test_defs/` describing one run: a dataset, a target curve, a scenario
and which test wells to score. Define your own instead of writing a one-off script; the runner scores every
model on it the same way.

## Schema (version 1)

```yaml
version: 1
name: rhob_gap50              # must equal the file name without .yaml
description: >                # plain words, for humans and agents
  Predict bulk density (RHOB) inside one hidden 50 m gap per well.
dataset: force2020            # a known dataset key, or one with a task spec or data folder
target: RHOB                  # RHOB | DTC | DTS (wmb.contract.TARGET_CURVES)
scenario:
  type: gap                   # missing_curve | gap
  gap_length_m: 50            # required (> 0) for gap; omit or null for missing_curve
  n_gaps: 1                   # optional, default 1; >1 places non-overlapping gaps per well
  seed: 2020                  # optional; default = the dataset task-spec seed
wells:
  split: test                 # test | leaderboard | list
  names: []                   # non-empty well ids when split is list
inputs_policy: model          # the model picks its inputs from model.yaml / required_inputs
metrics: [MAE, RMSE, R2, cover_90, cover_50, interval_width_90, has_uncertainty,
          load_s, fit_s, predict_s, samples_per_s]
```

- `wells.split`: `test` = the dataset's official test wells (FORCE: leaderboard + hidden; synthetic: the task-spec
  test list). `leaderboard` = FORCE wells with split `leaderboard` only (an error for synthetic). `list` = the
  wells in `names`, which must be test wells of the dataset. Train wells always stay the official train set.
- `inputs_policy: model` means the model chooses its inputs. Do not put per-model curves in a test file.
- `n_gaps: 1` equals the standard gap placement. For `n_gaps > 1`, each further gap uses a seed derived from
  seed, well, target, length and gap index. Placement retries with new seeds if a set of gaps does not fit.
  Gaps never overlap, and one task per well carries all gaps. The scenario seed does not change the well split.
- `metrics` names the standard scores of interest. The runner still writes all available scores for compatibility.
- Unknown keys, unknown metric names and a `gap_length_m` on `missing_curve` are errors.

## Create, check, run

```
python -m benchmark.new_test my_test                 # copies benchmark/test_defs/_template.yaml, then validates
python -m benchmark.check_test my_test               # validate only (also: all, or a path; new_test --check works too)
python -m benchmark.run --model template_linear --test my_test
python -m benchmark.run --model template_linear --test all
python -m benchmark.run --list-tests
```

`check_test` always runs the schema checks. When the dataset can be loaded it also checks that wells exist and that
enough test wells have the target (finite samples, or a placeable gap of that length): zero is an error, fewer than
three a warning. Without data (for example no `data_paths.local.yaml`), it prints `skipped well-presence check`.
`new_test` reports data-check problems as warnings; its exit status reflects schema validity.

`--test all` runs every valid def in name order and re-fits for each. Both `--test all` and `--list-tests` warn and skip
invalid files; `check_test all` reports them as errors. A valid def whose data cannot load gives a row with `error`
and a `failure.json`; the other tests still run. Optional datasets (`nlog_dts`,
`kansas_kgs_dts`) are **skipped** under `--test all` when the aligned folder is
not present; an explicit `--test nlog_dts_missing` still fails if the data is missing.
See [data-alignment.md](data-alignment.md). With `--test`, `--dataset`, `--target` and `--scenario` are optional;
if given they must agree with the def (for example `--scenario gap_12.5`). Files whose names start with `_` and
`TEMPLATE` are examples and never run; `all` is reserved for the CLI.

## Shipped tests

| name | dataset | target | scenario |
|---|---|---|---|
| `rhob_missing` | force2020 | RHOB | curve hidden throughout the well |
| `rhob_gap50` | force2020 | RHOB | one 50 m gap per well |
| `dtc_missing` | force2020 | DTC | curve hidden throughout the well |
| `dts_missing` | force2020 | DTS | curve hidden throughout the well |
| `dts_gap50` | force2020 | DTS | one 50 m gap per well |
| `synthetic_rhob_missing` | synthetic | RHOB | curve hidden; runs without any data |

## Old CLI

`--dataset/--target/--scenario` still work. Each combination maps to a def: the shipped def with the same dataset,
target and scenario (official test wells, one gap) if there is one (`force2020 RHOB missing_curve` is
`rhob_missing`), else a generated name such as `synthetic_rhob_missing` or `synthetic_dtc_gap_50`.

## Results

`--test` writes `scores.json`, `failure.json` and `predictions.json` under
`results/<model>/<dataset>/<target>/<scenario>/<test>/`, so named tests can share a scenario without overwriting one
another. The legacy flags still use `results/<model>/<dataset>/<target>/<scenario>/` (plus the older
`<target>/scores.json` copy for `missing_curve`). Both layouts carry the `test` field.

The runner reuses cached predictions for an unchanged model, training set and task sequence. Use `--rerun` (or
`--force`) to recompute predictions and record fresh fit and prediction timings.

## Testing

Do not use FORCE 2020 in pytest; use `synthetic` or temporary defs (see `tests/test_test_defs.py`).
