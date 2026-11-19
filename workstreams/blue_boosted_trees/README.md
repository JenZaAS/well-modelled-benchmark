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

## Rules for this folder
- Only this team edits files here. Others may read and import.
- Put personal experiments in `scratch/` (gitignored) or a notebook named `<yourname>_*.ipynb` (clear outputs before committing).
- No data, weights or big outputs here: data in `FH2026_DATA`, weights in the cache, results in `results/` (gitignored).
- Need a change in `src/wmb/` or `benchmark/`? Ask in the team chat; don't edit it yourself.
