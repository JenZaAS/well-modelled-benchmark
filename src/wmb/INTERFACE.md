# `wmb` shared package: stable interface

Shared by all teams. Everyone imports it; nobody changes it silently.
To change anything below: say so in the team chat first, keep the change small,
and keep old names working until everyone has pulled.

| Module | What you use | Promise |
|---|---|---|
| `wmb.io` | `Well(name, depth, curves, dataset, split, meta)`, `read_las`, `read_force_csv`, `load_manifest`, `iter_manifest_wells` | Depth in metres MD, curves keyed by the 9 core names, NaN = missing |
| `wmb.curves` | `CORE`, `UNITS`, `LOG_SCALE`, `match_property(mnemonic)` | Core names never change: GR, RDEP, RMED, CALI, RHOB, NPHI, PEF, DTC, DTS (+ LITH) |
| `wmb.contender` | `BaseContender`, `Contender` protocol, `Prediction(mean, std, quantiles)` | `fit(train_wells, target)`, `predict(well_without_target, target) -> Prediction` |
| `wmb.scoring` | `regression_scores`, `classification_scores`, `crps_gaussian`, `timer` | Metric names and definitions as in `docs/metrics.md` |
| `wmb.splits` | `well_split`, `official_split`, `gap_mask` | Split by whole wells; same seed, same split |
| `wmb.paths` | `data_root()`, `get_dataset_path(name)`, `work_root()`, `results_root()` | Data from `data_paths.local.yaml` (see `docs/SETUP_DATA_PATHS.md`) |
| `wmb.synthetic` | `make_wells()` | Fake wells for smoke tests only |
