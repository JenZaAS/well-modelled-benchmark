# Data (not in git)

This folder only holds this README. **Data is never committed** (`.gitignore` blocks `data/*`, `*.las`, `*.csv`, ...).

## Where the data lives

Point the code at your local copy of the shared team data folder with a
gitignored file at the repo root: `data_paths.local.yaml`.

1. Copy [`data_paths.example.yaml`](../data_paths.example.yaml) to `data_paths.local.yaml`
   (or point a coding agent at [`docs/SETUP_DATA_PATHS.md`](../docs/SETUP_DATA_PATHS.md);
   it will ask you for the folder and write the file).
2. Set `data_root` to the shared team data folder. One subfolder per dataset, e.g.
   `FORCE2020/`, `Sodir-FactPages/`, `NLOG-DTS/`, `Kansas-KGS-DTS/`, `Volve/`,
   each with a README (source, licence, credit) and, where ready, a `wells.json` manifest.
   For NLOG and Kansas, point at the *aligned* copy (standard names, metres MD,
   one LAS per well; see [docs/data-alignment.md](../docs/data-alignment.md)).
3. Run `python -m wmb.paths --check` and confirm the datasets you have are found.
4. **Never commit** `data_paths.local.yaml`.

Fallback: if that file is missing, the environment variable `FH2026_DATA` is still
accepted as `data_root`. Other optional variables: `FH2026_WORK` (intermediate files),
`FH2026_CACHE` (venv, model weights, Hugging Face cache; also `cache_dir` in the yaml),
`WMB_RESULTS` (default `<repo>/results`). Never put the cache inside the repo folder.

Expected layout under `data_root`:

```
FORCE2020/        train.csv, leaderboard_test_features.csv, leaderboard_test_target.csv, hidden_test.csv, penalty_matrix.npy, wells.json
Sodir-FactPages/  wellbore_*_all.csv (coordinates, tops)
NLOG-DTS/         <well>/*.las, wells.json, README.md
Kansas-KGS-DTS/   <well>/*.las, wells.json, README.md
Volve/            LAS files per well (+ wells.json when ready)
...               any other dataset: a folder with LAS or CSV files + wells.json
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
Resolve the folder with `wmb.paths.get_dataset_path("force2020")` (also `sodir_factpages`, `nlog_dts`, `kansas_kgs_dts`, `volve`).

## Rules

- Public open data only; respect each dataset's licence and credit line ([docs/data-licences.md](../docs/data-licences.md)).
- Diskos raw data (if we get access) stays on the local machine. Only scores and aggregates may be published.
- Don't use computed porosity, Vshale or Sw as targets we compute ourselves; use independent CPI curves.
