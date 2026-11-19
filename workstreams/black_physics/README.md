# Black team: physics and empirical models

**Team:** Black team · **Folder:** `workstreams/black_physics/`

## Scope
Gardner, weighted density, Greenberg-Castagna and similar rules: fast, explainable floors to beat. Geoscience know-how wanted. Don't compute porosity, Vshale or Sw as targets; use independent CPI curves instead.

## Inputs
`list[Well]` from `wmb`.

## Outputs (contract)
Contender classes in `contender.py` returning `Prediction(mean, std)`.

## What's here
`contender.py` with `black_gardner` (RHOB from DTC) and `black_castagna_dts` (DTS from DTC, mudrock line).

Run it:
```
python -m benchmark.run --dataset synthetic --target RHOB --contenders black_gardner
```

## Rules for this folder
- Only this team edits files here. Others may read and import.
- Put personal experiments in `scratch/` (gitignored) or a notebook named `<yourname>_*.ipynb` (clear outputs before committing).
- No data, weights or big outputs here: data in `FH2026_DATA`, weights in the cache, results in `results/` (gitignored).
- Need a change in `src/wmb/` or `benchmark/`? Ask in the team chat; don't edit it yourself.
