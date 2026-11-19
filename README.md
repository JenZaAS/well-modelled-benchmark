# well-modelled-benchmark

**To open the well map:** double-click `Launch viewer.bat` (Windows), `launch_viewer.command` (macOS) or `launch_viewer.sh` (Linux). It starts a local viewer that reads `data_paths.local.yaml` and does not embed well data in the repo.

Open benchmark for predicting missing well logs (physics, gradient boosted trees, time-series foundation models), scored on accuracy and honest uncertainty. FORCE Hackathon 2026.

**The idea in one line:** hide one log, predict it from the others, say how sure we are, and score every
approach the same way: same wells, same split, same metrics (accuracy, coverage, sharpness, speed).

## Quick start with the demo data

FORCE 2020 is the official demo dataset (118 public wells). You do not need a
shared team folder.

1. Download the files listed in [datasets/force2020/README.md](datasets/force2020/README.md)
   from [Zenodo 4351156](https://doi.org/10.5281/zenodo.4351156) and the official
   [FORCE 2020 GitHub](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition)
   (`lithology_competition/data/`). Unzip `train.zip` to get `train.csv`.
2. Put them in a folder named `FORCE2020` under your local data folder.
3. Run onboarding (creates gitignored `data_paths.local.yaml`):

   ```bash
   python -m wmb.onboarding
   ```

   Or set the path yourself from [docs/SETUP_DATA_PATHS.md](docs/SETUP_DATA_PATHS.md).
4. Verify the files:

   ```bash
   python -m wmb.datasets --check force2020
   python -m wmb.onboarding --status
   ```

Onboarding is safe to run again later: it stores progress in gitignored
`.wmb/onboarding_state.json`, shows what is new when the onboarding version
is bumped, skips finished steps unless you pass `--redo`, and re-verifies
datasets you chose to use.

Licence and credit: [datasets/force2020/README.md](datasets/force2020/README.md)
(CC BY 4.0 / NLOD 2.0; Bormann, P., Aursand, P., Dilib, F., Manral, S. and
Dischington, P. (2020). FORCE 2020 Well well log and lithofacies dataset for
machine learning competition. Zenodo. Also credit FORCE and the Norwegian
Offshore Directorate (Sodir).).

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt        # core only; ML extras: see tools/laptop_setup/
pip install -e .                       # makes `import wmb` work everywhere

python -m benchmark.run --list                              # all contenders
python -m benchmark.run --dataset synthetic --target RHOB   # runs with fake wells, no data needed
python -m benchmark.scoreboard
```

Real data: start with the demo dataset above, or copy [data_paths.example.yaml](data_paths.example.yaml)
to `data_paths.local.yaml` (gitignored) and set `data_root` to your local data folder.
A coding agent can do this from [docs/SETUP_DATA_PATHS.md](docs/SETUP_DATA_PATHS.md).
Then run `python -m wmb.paths --check` or `python -m wmb.onboarding --status`, and e.g.
`python -m benchmark.run --dataset force2020 --target DTS`.

## Folder map: which team works where

Edit **only your own folder**. Shared folders change only after a heads-up in the team chat.

| Folder | What | Team |
|---|---|---|
| `src/wmb/` | Shared package: load wells, curve names, scoring, splits, the `Contender` interface ([INTERFACE.md](src/wmb/INTERFACE.md)) | Shared: announce changes |
| `benchmark/` | Runner + scoreboard; finds every team's models automatically | Benchmark |
| `workstreams/data_prep/` | Datasets, curve labelling, QC, Diskos | Data preparation |
| `workstreams/red_foundation_models/` | Chronos-2, MOMENT, TTM | Red team |
| `workstreams/blue_boosted_trees/` | XGBoost, LightGBM | Blue team |
| `workstreams/black_physics/` | Gardner, Greenberg-Castagna, other rules | Black team |
| `viewer/` | Interactive well map and log plots (any dataset with a `wells.json`) | Shared |
| `tools/` | Inventory, open-data fetchers, NLOG pipeline, laptop setup | Data preparation |
| `experiments/` | Earlier proofs of concept, kept for reference | Blue team |
| `docs/` | Benchmark definition, metrics, data licences, models | Benchmark; anyone may add |
| `data/` | **Only a README.** Data lives outside git | Data preparation |
| `results/` | Outputs (gitignored); `results/summary/` small tables may be committed | Benchmark |
| `tests/` | Smoke tests that run without data | Shared |

Roles are flexible: people may take several, and teams may merge (e.g. blue and black).

## Adding a model (any team)

1. In your `workstreams/<team>/contender.py`, write a class based on `wmb.contender.BaseContender`
   with `fit(train_wells, target)` and `predict(well, target) -> Prediction(mean, std)`.
2. Add it to the `CONTENDERS` list in the same file.
3. `python -m benchmark.run --list` shows it; `python -m benchmark.run --contenders <name>` scores it.

## Data rule

Public open data only. **Never commit data, model weights or big outputs** (the `.gitignore` blocks the usual
file types). Data is shared through the team's shared data folder; Diskos data, if we get access, stays local and only
scores and aggregates are published. Licences and credit lines: [docs/data-licences.md](docs/data-licences.md).

## Licence

MIT (see `LICENSE`). Don't bring in code, models or text that can't be released under MIT.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the golden rules.
