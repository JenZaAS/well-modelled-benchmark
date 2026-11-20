# Blue team: gradient-boosted trees

**Team:** Blue team · **Folder:** `workstreams/blue_boosted_trees/`

## Scope
XGBoost / LightGBM, a common choice in FORCE 2020. One depth sample = one row; window and gradient features welcome.

## Inputs
`list[Well]` from `wmb`.

## Outputs (contract)
Contender classes in `contender.py` returning `Prediction(mean, std)`. Saved models (`*.json`, `*.pkl`, `*.joblib`) go to `FH2026_WORK`, not git.

## What's here
`contender.py` is the registration file: the runner finds the blue models through its `CONTENDERS` list. Each model has its
own folder under `models/` (wrapper, `model.yaml`, `LICENSE.md`, `README.md`), following
[docs/moving-models.md](../../docs/moving-models.md):

| Registered name(s) | Folder | What it is |
|---|---|---|
| `blue_gbt` | `models/xgboost_quantile/` | the starter: XGBoost (or scikit-learn fallback) with quantile-based uncertainty |
| `blue_ml_gbt`, `blue_ml_gbt_trend` | `models/xgboost_gbt/` | XGBoost, point predictions, without / with trend features |
| `blue_ml_linear`, `blue_ml_linear_trend` | `models/linear_regression/` | multilinear regression |
| `blue_ml_rf` | `models/random_forest/` | random forest (base features only: with trend a fit takes about 7 min) |

The registered ML-study variants have no stratigraphy, because benchmark wells carry no formation labels; the variants with
stratigraphy and the context features are in the study and the app (below). The code that the three ML-study models share
(features, estimators, saved-model store, new-well prediction, the app) stays in `ml_study/`, so the saved models
(pickles that record that module path) keep loading; the model folders only hold thin wrappers. Earlier PoC: `experiments/poc-ml/`.

Run and check them:
```
python -m benchmark.check_model blue_ml_gbt                  # the contract check (synthetic data, no FORCE files needed)
python -m benchmark.run --model blue_ml_gbt --test rhob_missing
python -m benchmark.run --dataset synthetic --contenders blue_gbt
```

### ML study: gradient boosting vs linear regression vs random forest (`ml_study/`)
Predicts every core log (GR, RDEP, RMED, CALI, RHOB, NPHI, PEF, DTC, DTS) from the other logs + depth, each model
with and without stratigraphy, and with three feature sets (`base`, `ctx`, `trend`), on the official FORCE 2020 split
(98 training wells, 20 held-out wells). Variant ids: `gbt`, `gbt_strat`, `gbt_ctx`, `gbt_ctx_strat`, `gbt_trend`,
`gbt_trend_strat`, and the same for `linear` and `rf`. The results currently hold all `base` variants, and `ctx` and `trend`
for gradient boosting and linear regression (not random forest).

Decisions that affect comparisons:
- **Context features** (`ml_study/features.py`): a row-by-row model cannot see the rock above or below. With `ctx`, every input
  log also gets its value 5 m shallower and 5 m deeper, and 5 m and 20 m centred moving averages (4 extra columns per log).
  Windows are in metres (the FORCE 2020 sample spacing is 0.152 m). The log being predicted gets none, since it is unknown at prediction time.
  Result: no consistent gain; it helps GR a little and hurts RDEP and PEF for gradient boosting.
- **Trend features** (`trend`): per input log, the mean of the 10 m strictly above and below each sample, their difference
  (below minus above: positive = the log increases with depth) and the local standard deviation over 10 m (4 columns per log).
  Trees cannot form such differences themselves. Result: a small gain for gradient boosting on GR, CALI and RHOB
  (R² +0.03, +0.03, +0.01), no gain or a loss on RDEP and PEF; linear regression gains at most +0.02.
  Random forest with either feature set takes about 7 min per fit (about 2 h for all logs), so it has not been run yet.
- **Stratigraphy** = the `GROUP` and `FORMATION` columns already in the FORCE 2020 CSVs (14 groups, 69 formations; present in
  train and test files). `ml_study/data.py` has its own loader and keeps them in `Well.meta`, so no shared code changed.
  Trees get them as ordinal codes ranked by median depth; the linear model gets one-hot columns. Unseen labels are NaN / all zeros.
- **Models** (`ml_study/models.py`): XGBoost (200 trees, depth 6), `LinearRegression` (winsorised inputs at the 0.5/99.5
  percentiles, median imputation with missing flags, standardised; the imputer is a memory-light equivalent of scikit-learn's),
  random forest (60 trees, `min_samples_leaf=5`). All are `BaseContender`s, point predictions only (no uncertainty yet).
- **Training rows** are thinned evenly to 250,000 per fit (`--max-train-rows 0` = all) to keep the random forest tractable on a laptop.
- **Resistivity** (RDEP, RMED) is trained in log10 and scored in log10(ohm.m). Other logs are scored in their own unit.
- **Statistics** (`ml_study/metrics.py`): MAE, RMSE, R², bias (mean of predicted - measured), median abs error,
  MAE / std of the measured log, Pearson r.

### Quick start for a teammate (start here)
**With an AI agent (easiest):** pull the repo, open it in VS Code with Copilot Chat in agent mode, and run the prompt
[`.github/prompts/run-blue-app.prompt.md`](../../.github/prompts/run-blue-app.prompt.md) (type `/run-blue-app` in chat). It
checks what is already set up, asks you for the data folder and the two zip files, installs the packages, unpacks the zips
and starts the app. With another agent, tell it: "follow `.github/prompts/run-blue-app.prompt.md`". The manual steps follow.

From the repo root, in a virtual environment (Python 3.12 recommended):

1. **Install** the core packages and the app packages:
   ```
   pip install -r workstreams/blue_boosted_trees/requirements-app.txt
   ```
2. **Point the code at the FORCE 2020 data** (not in git). Create `data_paths.local.yaml` in the repo root
   (see [docs/SETUP_DATA_PATHS.md](../../docs/SETUP_DATA_PATHS.md)), then check it. You should see `force2020  found`:
   ```
   python -m wmb.paths --check
   ```
3. **Get the study results.** Either:
   - **Fast (recommended):** ask the owner for `blue_ml_study.zip` (in the team Dropbox) and unzip it into the repo's `results/`
     folder, so the files end up in `results/blue_ml_study/` (`metrics.csv` and `predictions/*.parquet`).
     `results/` is gitignored, so these are never committed. Or
   - **Recompute** (about 35 min on 4 cores for the 6 `base` variants, the default). Add feature sets with
     `--features base,ctx,trend`: each of `ctx` and `trend` adds about 12 min for gradient boosting and linear
     regression and about 2 h for random forest:
     `python -m workstreams.blue_boosted_trees.ml_study.run_study`
4. **Start the app:**
   ```
   streamlit run workstreams/blue_boosted_trees/ml_study/app.py
   ```
   It opens at http://localhost:8501. The first load takes about a minute (it reads the 20 test wells). To show it to
   people on the same network, add `--server.address 0.0.0.0` and share `http://<your-ip>:8501`.

Other commands:
```
python -m workstreams.blue_boosted_trees.ml_study.run_study --targets RHOB --models gbt,linear   # a subset
python -m workstreams.blue_boosted_trees.ml_study.run_study --models gbt,linear --features ctx,trend   # only add those variants
python -m pytest workstreams/blue_boosted_trees/tests                                            # tests need no data
```
Results are merged per (log, variant), so a partial run keeps everything it did not recompute. The app shows whichever variants are in the results.
Troubleshooting:
- "No results in ..." in the app: step 3 is missing, or the sidebar *Results folder* points to the wrong place.
- "FORCE 2020 data not found": step 2 is missing.
- "No module named streamlit/plotly": step 1 was run in a different environment.
- On Windows, set `PYTHONUTF8=1` if the console output contains garbled or non-ASCII characters (`benchmark.run` needs it).

The app shows only the 20 held-out wells on the first two tabs (the models never saw them). Zip `results/blue_ml_study` to share your own results.

### Predicting a new well (third tab, "Predict a new well")
Upload a LAS file, optionally a formation-tops table, pick a log and a model, and get the predicted curve next to the
other logs (and next to the measured curve if the file has one), with a CSV download (`<LOG>_predicted`, `<LOG>_measured`,
`<LOG>_filled` = measured where it exists, predicted elsewhere).

1. **Get the saved models** (gradient boosting and linear regression, every feature set, about 20 MB in total):
   unzip the shared `blue_ml_study_models.zip` into the work folder (`FH2026_WORK`, default `~/fh2026-cache/work/blue_ml_study/models`),
   or train them (about 35 min): `python -m workstreams.blue_boosted_trees.ml_study.train_models`.
   They are trained on **all 118** FORCE 2020 wells (no held-out wells, so no accuracy figures from them); the accuracy
   on the other tabs comes from separate 98-well models. Saved models are joblib pickles, so only load files you trust and use
   the same library versions (`manifest.json` records them and the app warns on a mismatch).
2. **LAS file:** curve mnemonics are mapped to the 9 standard logs automatically (`wmb.curves`), depth is converted to
   metres. The app warns if a log's median is outside the training range (wrong unit, for example kg/m3 instead of g/cm3,
   % instead of v/v, us/m instead of us/ft). Units are not converted for you.
3. **Tops (optional):** the stratigraphy variants need `GROUP`/`FORMATION` per depth. Without tops only the variants without
   stratigraphy are offered. Tops use the NPD layout: a name column (`Surface` or `HorizonName`, e.g. `VIKING GP. Top`,
   `Draupne Fm. Top`) and a depth column (`MD`), in csv or xlsx, optionally with a `Well identifier` column (the rows of the
   well named in the LAS file are used). Names ending in `GP.` are groups, all others formations; a top applies down to the next
   top of its level, and the part above the first top is unknown. On the FORCE wells this reproduces the dataset's own labels
   exactly. Names that the models have not seen are listed and treated as unknown.
4. Sample spacing: context and trend windows are defined in metres, so wells with another spacing work, but the models were trained on 0.152 m.
5. **Check the input coverage.** The FORCE 2020 CSVs only contain the interval with lithology labels, but a LAS file usually
   holds the whole logged section, including shallow depths where only GR and resistivity were run. The models were rarely
   trained on depths with so few inputs (under 10% of training rows had 4 or fewer input logs) and predict them badly: on the
   FORCE well 15/9-23, GR inside the CSV interval scored R² 0.84 but -6.6 in the extra shallow section. The tab therefore has
   a *minimum number of input logs* setting (default 5) and shows no prediction below it. DTC is the exception: it held up
   with 3 inputs on the held-out wells.

## Rules for this folder
- Only this team edits files here. Others may read and import.
- Put personal experiments in `scratch/` (gitignored) or a notebook named `<yourname>_*.ipynb` (clear outputs before committing).
- No data, weights or big outputs here: data in `FH2026_DATA`, weights in the cache, results in `results/` (gitignored).
- Need a change in `src/wmb/` or `benchmark/`? Ask in the team chat; don't edit it yourself.
