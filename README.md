# well-modelled-benchmark

Open benchmark for predicting missing well logs (physics, gradient boosted trees, time-series foundation models), scored on accuracy and honest uncertainty. FORCE Hackathon 2026.

**The idea in one line:** hide one log, predict it from the others, say how sure we are, and score every
approach the same way: same wells, same split, same metrics (accuracy, coverage, sharpness, speed).

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

Real data: copy [data_paths.example.yaml](data_paths.example.yaml) to `data_paths.local.yaml`
(gitignored) and set `data_root` to your local copy of the shared data folder.
A coding agent can do this from [docs/SETUP_DATA_PATHS.md](docs/SETUP_DATA_PATHS.md).
Then run `python -m wmb.paths --check`, and e.g. `python -m benchmark.run --dataset force2020 --target DTS`.

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
