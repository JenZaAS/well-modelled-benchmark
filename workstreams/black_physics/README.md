# Black team: physics and empirical models

**Team:** Black team · **Folder:** `workstreams/black_physics/`

## Scope
Gardner, weighted density, Greenberg-Castagna and similar rules: fast, explainable floors to beat. Geoscience know-how wanted. Don't compute porosity, Vshale or Sw as targets; use independent CPI curves instead.

## Inputs
`list[Well]` from `wmb`.

## Outputs (contract)
Contender classes in `contender.py` returning `Prediction(mean, std)`.

## What's here

Seven registered physics contenders. Each is a pointwise cross-log predictor for whole missing curves and
gaps. Choose by the input logs you have; each model card lists units, assumptions, fit requirements,
abstention rules and examples.

| ID | Target ← inputs | Model card |
| --- | --- | --- |
| `black_gardner` | RHOB ← DTC | [Gardner, refitted a and b](models/gardner/README.md) |
| `black_gardner_fixed` | RHOB ← DTC | [Gardner, published 0.31 · Vp^0.25](models/gardner/README.md) |
| `black_castagna_dts` | DTS ← DTC | [Castagna mudrock line](models/castagna_mudrock/README.md) |
| `black_gc_brine_gr` | DTS ← DTC, GR | [Greenberg–Castagna, brine sand/shale from GR](models/greenberg_castagna/README.md) |
| `black_wyllie_nphi` | DTC ← NPHI | [Wyllie time average, NPHI as porosity](models/wyllie/README.md) |
| `black_density_nphi` | RHOB ← NPHI | [Volumetric density, NPHI as porosity](models/volumetric_density/README.md) |
| `black_vpvs_ratio` | DTS ← DTC | [One fitted DTS/DTC ratio (control)](models/vpvs_ratio/README.md) |

Opt-in gap rows (not in default runs; [granite/README.md](granite/README.md) has the 50 m density-gap experiment,
its environments and results):

| ID | Target ← inputs | Switch | Model card |
| --- | --- | --- | --- |
| `black_ttm_r3_frozen`, `black_ttm_r3_tuned` | RHOB ← RHOB flanks only | `WMB_BLACK_GRANITE=1` | [Granite gap rows](models/granite/README.md) |
| `black_tspulse_frozen`, `black_tspulse_tuned` | RHOB ← RHOB flanks only | `WMB_BLACK_GRANITE=1` | [Granite gap rows](models/granite/README.md) |
| `black_linear_gap` | RHOB, DTC, DTS ← own flanks | `WMB_BLACK_GRANITE=1` | [Granite gap rows](models/granite/README.md) |
| `black_gardner_flank_resid` | RHOB ← DTC + RHOB flanks | `WMB_BLACK_GRANITE=1` | [Granite gap rows](models/granite/README.md) |
| `black_timestral_target_only` | RHOB ← RHOB flanks only | `WMB_BLACK_TIMESTRAL=1` | [licence note](models/timestral_target_only/LICENSE.md) |

- `contender.py`: explicit registration (`CONTENDERS`); `Gardner` and `MudrockDTS` stay importable from it.
- `models/<family>/`: `equation.py` (plain NumPy), `wrapper.py` (station wrapper), `README.md` (model card),
  `LICENSE.md` and one `<id>.model.yaml` per registered ID (required inputs per target, read by the viewer and
  `check_model`; see `docs/models.md`). `models/_shared/` has slowness conversion, curve access and
  uncertainty calibration.
- Each model returns NaN, not an error, wherever an input is missing or outside the equation's domain. It
  returns one Gaussian σ from equal-well-weight residuals (out-of-fold when parameters are fitted), or
  `std=None` when fewer than two wells contribute.
- [`VALIDATION.md`](VALIDATION.md): FORCE 2020 results (error, fill fraction, common-support comparisons,
  interval coverage, timing) and how to run a black model next to your own on the same depths.
- `validate_force.py`: the aggregate-only script behind that note, including a gap-only straight-line
  interpolation reference.
- `tests/`: focused tests (`python -m pytest -q workstreams/black_physics/tests`).

Run it:
```
python -m benchmark.check_model black_gc_brine_gr
python -m benchmark.run --dataset synthetic --model black_gardner,black_gc_brine_gr --target all --scenario all
python -m benchmark.run --dataset force2020 --model black_gc_brine_gr,template_linear --target DTS --scenario gap_50
python -m benchmark.run --model black_gc_brine_gr,black_castagna_dts --test dts_gap50     # same windows via a test definition
python -m workstreams.black_physics.validate_force --compare template_linear --check-results
```

## Rules for this folder
- Only this team edits files here. Others may read and import.
- Put personal experiments in `scratch/` (gitignored) or a notebook named `<yourname>_*.ipynb` (clear outputs before committing).
- Implementation plans, research briefs and pilots live in `notes/`, which is gitignored by this folder's `.gitignore` and never committed.
- No data, weights or big outputs here: data in `FH2026_DATA`, weights in the cache, results in `results/` (gitignored).
- Need a change in `src/wmb/` or `benchmark/`? Ask in the team chat; don't edit it yourself.
