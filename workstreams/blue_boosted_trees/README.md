# Blue team: gradient-boosted trees

**Team:** Blue team · **Folder:** `workstreams/blue_boosted_trees/`

## Scope
XGBoost / LightGBM, a common choice in FORCE 2020. One depth sample = one row; window and gradient features welcome.

## Inputs
`list[Well]` from `wmb`.

## Outputs (contract)
Contender classes in `contender.py` returning `Prediction(mean, std)`. Saved models (`*.json`, `*.pkl`, `*.joblib`) go to `FH2026_WORK`, not git.

## What's here
`contender.py` with `blue_gbt` (XGBoost or sklearn fallback, quantile sigma). Earlier PoC: `experiments/poc-ml/`.

Run it:
```
python -m benchmark.run --dataset synthetic --contenders blue_gbt
```

### ML study: gradient boosting vs linear regression vs random forest (`ml_study/`)
Predicts every core log (GR, RDEP, RMED, CALI, RHOB, NPHI, PEF, DTC, DTS) from the other logs + depth, each model
with and without stratigraphy, on the official FORCE 2020 split (98 training wells, 20 held-out wells).

Decisions that affect comparisons:
- **Stratigraphy** = the `GROUP` and `FORMATION` columns already in the FORCE 2020 CSVs (14 groups, 69 formations; present in
  train and test files). `ml_study/data.py` has its own loader and keeps them in `Well.meta`, so no shared code changed.
  Trees get them as ordinal codes ranked by median depth; the linear model gets one-hot columns. Unseen labels are NaN / all zeros.
- **Models** (`ml_study/models.py`): XGBoost (200 trees, depth 6), `LinearRegression` (winsorised inputs at the 0.5/99.5
  percentiles, median imputation with missing flags, standardised), random forest (60 trees, `min_samples_leaf=5`). All are
  `BaseContender`s, point predictions only (no uncertainty yet).
- **Training rows** are thinned evenly to 250,000 per fit (`--max-train-rows 0` = all) to keep the random forest tractable on a laptop.
- **Resistivity** (RDEP, RMED) is trained in log10 and scored in log10(ohm.m). Other logs are scored in their own unit.
- **Statistics** (`ml_study/metrics.py`): MAE, RMSE, R², bias (mean of predicted - measured), median abs error,
  MAE / std of the measured log, Pearson r.

### Quick start for a teammate (start here)
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
   - **Recompute** (about 35 min on 4 cores): `python -m workstreams.blue_boosted_trees.ml_study.run_study`
4. **Start the app:**
   ```
   streamlit run workstreams/blue_boosted_trees/ml_study/app.py
   ```
   It opens at http://localhost:8501. The first load takes about a minute (it reads the 20 test wells). To show it to
   people on the same network, add `--server.address 0.0.0.0` and share `http://<your-ip>:8501`.

Other commands:
```
python -m workstreams.blue_boosted_trees.ml_study.run_study --targets RHOB --models gbt,linear   # a subset; other logs keep their results
python -m pytest workstreams/blue_boosted_trees/tests                                            # tests need no data
```
Troubleshooting:
- "No results in ..." in the app: step 3 is missing, or the sidebar *Results folder* points to the wrong place.
- "FORCE 2020 data not found": step 2 is missing.
- "No module named streamlit/plotly": step 1 was run in a different environment.
- On Windows, set `PYTHONUTF8=1` if the console output contains garbled or non-ASCII characters (`benchmark.run` needs it).

The app shows only the 20 held-out wells (the models never saw them). Zip `results/blue_ml_study` to share your own results.

## Rules for this folder
- Only this team edits files here. Others may read and import.
- Put personal experiments in `scratch/` (gitignored) or a notebook named `<yourname>_*.ipynb` (clear outputs before committing).
- No data, weights or big outputs here: data in `FH2026_DATA`, weights in the cache, results in `results/` (gitignored).
- Need a change in `src/wmb/` or `benchmark/`? Ask in the team chat; don't edit it yourself.
