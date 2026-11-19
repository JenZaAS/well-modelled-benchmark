# Build your model

A short path from a fresh checkout to a model the benchmark can score. Details live in [models.md](models.md) (inputs and `model.yaml`), [benchmark.md](benchmark.md) (the input/output contract and metrics), [tests.md](tests.md) (how tests are defined) and [moving-models.md](moving-models.md) (optional `models/` folder layout).

## 1. Get the repo and the demo data

```bash
git pull --rebase
python -m wmb.onboarding
```

Onboarding writes `data_paths.local.yaml` (gitignored) and checks the FORCE 2020 demo files. If you already have a data folder, see [SETUP_DATA_PATHS.md](SETUP_DATA_PATHS.md). Never commit that yaml file.

## 2. Copy the template

Copy [`workstreams/_template_model/`](../workstreams/_template_model/) into your team's models folder, one model per subfolder:

```text
workstreams/<team>/models/<model_name>/
```

In that copy of `contender.py`, rename the class and set `name` (unique in the repo) and `workstream` (your team folder name, for example `blue_boosted_trees`). The runner does not scan `models/` on its own; you register the class in the next steps.

## 3. Declare inputs and the licence

Fill `model.yaml` in the model folder: `name`, `licence`, and for each target the `required` and `optional` input curves (canonical names such as `GR`, `DTC`, `RHOB`; do not list `DEPTH`). See [models.md](models.md).

Add `LICENSE.md` in the model folder (next to the wrapper and `model.yaml`) with the licence name, URL, and a one-line summary of restrictions. `python -m benchmark.check_model` looks there (`LICENSE.md`, `LICENSE` or `LICENSE.txt`). A file only in the team folder still passes, with a warning asking you to move it next to the model. Set `licence_name`, `licence_url` and `licence_restrictions` on the class. Check that the licence fits your use before you run or share the model ([THIRD_PARTY.md](../THIRD_PARTY.md)).

## 4. Register it

In `workstreams/<team>/contender.py`, import the class and add it to the `CONTENDERS` list. That list is how the runner finds models. Do not remove a working entry until the new import runs.

## 5. Check the contract

```bash
python -m benchmark.check_model <name>
```

This runs small synthetic tasks (no FORCE files needed) and checks the input/output contract, `model.yaml`, and licence metadata.

## 6. Run the tests

```bash
python -m benchmark.run --model <name> --test all
python -m benchmark.run --model <name> --test rhob_missing
python -m benchmark.run --list-tests
```

`--test all` runs every shipped test (and any you have added). FORCE tests such as `rhob_missing` need the demo data from step 1; `synthetic_rhob_missing` runs without it.

To define your own test: `python -m benchmark.new_test my_test`, edit `benchmark/test_defs/my_test.yaml`, then run `--test my_test`. See [tests.md](tests.md).

The older flags `--dataset`, `--scenario` and `--target` still work.

## 7. Look at the results

Double-click `Launch viewer.bat` (Windows), `launch_viewer.command` (macOS) or `launch_viewer.sh` (Linux). Use the **Benchmark** tab for scored runs and the **Prediction** tab to try a model on a well.

With `--test`, scores are written to `results/<model>/<dataset>/<target>/<scenario>/<test>/scores.json`
(gitignored). `python -m benchmark.scoreboard` prints the table.
