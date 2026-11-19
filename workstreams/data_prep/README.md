# Data preparation

**Team:** Data preparation · **Folder:** `workstreams/data_prep/`

## Scope
Compile datasets and label which log is which: FORCE 2020 ready, missing-curve check, Diskos curves + QC (ideally someone with Diskos experience). Volve, NLOG and the other open sets go through the same `wells.json` manifest.

## Inputs
Raw files in the team's shared data folder, copied to the local cache (`FH2026_DATA`).

## Outputs (contract)
For every dataset: a valid `wells.json` (schema: `viewer/manifest.schema.json`) with `curve_map` to the 9 core curves, completeness per well, licence and credit. Cleaned copies in `FH2026_WORK/prepared/<dataset>/`. Diskos raw data never leaves the laptop.

## What's here
`prepare.py` (prints completeness through `wmb`). Existing helpers: `tools/inventory/`, `tools/nlog/`, `tools/fetch_open_data/`.

Run it:
```
python -m workstreams.data_prep.prepare --dataset synthetic
```

## Rules for this folder
- Only this team edits files here. Others may read and import.
- Put personal experiments in `scratch/` (gitignored) or a notebook named `<yourname>_*.ipynb` (clear outputs before committing).
- No data, weights or big outputs here: data in `FH2026_DATA`, weights in the cache, results in `results/` (gitignored).
- Need a change in `src/wmb/` or `benchmark/`? Ask in the team chat; don't edit it yourself.
