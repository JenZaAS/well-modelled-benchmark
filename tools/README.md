# Tools (data plumbing, not models)

Mostly used by Data preparation. Data in/out via env vars (`FH2026_DATA`,
`FH2026_WORK`, `FH2026_DL`), never the repo.

| Folder | What |
|---|---|
| `inventory/` | Per-well curve completeness for FORCE 2020 and Volve; mnemonic mapping notes; FORCE vs Volve duplicates note |
| `fetch_open_data/` | `fh_logs.py` (LAS header matching, completeness, manifests), `screen_kgs.py` (screen Kansas LAS headers), `extract_and_build.py` (KGS / Utah FORGE / GLADE / USGS CWLDB to `wells.json`) |
| `nlog/` | NLOG (Netherlands) fetch, completeness and manifest writer |
| `laptop_setup/` | Windows CPU setup: venv, data, model weights, offline smoke test |

Completeness CSVs and manifests are outputs: they go to the data/work folders, not git (for now).
