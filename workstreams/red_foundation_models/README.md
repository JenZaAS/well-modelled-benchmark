# Red team: time-series foundation models

**Team:** Red team · **Folder:** `workstreams/red_foundation_models/`

## Scope
Chronos-2, MOMENT and TTM: zero-shot, a small trained head, or a fine-tune. Read each log as a curve over depth. Stretch: a head with calibrated bin probabilities.

## Inputs
`list[Well]` from `wmb` (train wells with the target; test wells without it).

## Outputs (contract)
Contender classes in `contender.py` returning `Prediction(mean, std)`. Weights stay in the Hugging Face cache under `FH2026_CACHE`, never in git.

## What's here
`contender.py` with a placeholder floor (`red_floor_train_mean`). Replace or add real models next to it.

Run it:
```
python -m benchmark.run --dataset synthetic --contenders red_floor_train_mean
```

## Rules for this folder
- Only this team edits files here. Others may read and import.
- Put personal experiments in `scratch/` (gitignored) or a notebook named `<yourname>_*.ipynb` (clear outputs before committing).
- No data, weights or big outputs here: data in `FH2026_DATA`, weights in the cache, results in `results/` (gitignored).
- Need a change in `src/wmb/` or `benchmark/`? Ask in the team chat; don't edit it yourself.
