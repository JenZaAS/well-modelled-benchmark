# Data (not in git)

This folder only holds this README. **Data is never committed** (`.gitignore` blocks `data/*`, `*.las`, `*.csv`, ...).

## Where the data lives

1. **Shared copy:** the team's shared data folder (ask the team for access). One subfolder per dataset, e.g. `FORCE2020/`, `Volve/`, `NLOG/`, each with a README
   (source, licence, credit) and, where ready, a `wells.json` manifest.
2. **Your local cache:** copy the datasets you need to `%USERPROFILE%\fh2026-cache\data\` (Windows) or
   `~/fh2026-cache/data/` and point the code at it:

   ```
   set FH2026_DATA=%USERPROFILE%\fh2026-cache\data        (cmd)
   $env:FH2026_DATA = "$env:USERPROFILE\fh2026-cache\data"  (PowerShell)
   export FH2026_DATA=~/fh2026-cache/data                  (bash)
   ```

   Other variables: `FH2026_WORK` (intermediate files, default `~/fh2026-cache/work`), `FH2026_CACHE`
   (venv, model weights, Hugging Face cache), `WMB_RESULTS` (default `<repo>/results`).
   Never put the cache inside the repo folder.

Expected layout under `FH2026_DATA`:

```
force2020/        train.csv, leaderboard_test_features.csv, leaderboard_test_target.csv, hidden_test.csv, penalty_matrix.npy, wells.json
sodir_factpages/  wellbore_*_all.csv (coordinates, tops)
Volve/            LAS files per well (+ wells.json when ready)
nlog/             <well>/*.las, wells.json, README.md
...               any other dataset: a folder with LAS files + wells.json
```

## The `wells.json` contract (manifest_version 1)

Every dataset folder has a `wells.json` next to its files. Paths inside are relative to that folder.
Full schema: [`viewer/manifest.schema.json`](../viewer/manifest.schema.json); details: [`viewer/README.md`](../viewer/README.md).

- `dataset`: `id`, `name`, `licence`, `source_url`, `credit`, `depth_units` (`m`/`ft`), default `curve_map`, optional `files`.
- `wells[]`: `name`, optional `split` (`train` / `leaderboard` / `hidden` / ...), `lat`/`lon` (WGS84) **or**
  `x`/`y` + `xy_crs`, `curve_map` (core name -> mnemonic in the file), `files[]` (`path`, `format`: `las` / `csv` / `csv-target`),
  optional `completeness`.
- Core curve names: `GR, RDEP, RMED, CALI, RHOB, NPHI, PEF, DTC, DTS` (+ `LITH` for lithology).

Load any of it in Python with `wmb.io.iter_manifest_wells(<folder>)` (LAS) or `wmb.io.read_force_csv(<file>)` (FORCE CSV).

## Rules

- Public open data only; respect each dataset's licence and credit line ([docs/data-licences.md](../docs/data-licences.md)).
- Diskos raw data (if we get access) stays on the local machine. Only scores and aggregates may be published.
- Don't use computed porosity, Vshale or Sw as targets we compute ourselves; use independent CPI curves.
