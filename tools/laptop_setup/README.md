# FH2026 laptop setup

Sets up a Windows laptop for the hackathon so everything works offline: a Python environment, the FORCE 2020 data, the Sodir FactPages snapshot, offline docs and the ML model weights. Then it runs a smoke test.

Written 17 Nov 2026. Tested on Linux with Python 3.12 (packages, downloads, smoke test all PASS). Not yet run on Windows, so watch the first run.

## Files

- `setup-fh2026.ps1`: the script you run.
- `requirements.txt`: Python packages (torch and momentfm are installed separately by the script).
- `fh2026_fetch.py`: downloads data, docs and model weights, with size and checksum checks.
- `fh2026_smoke_test.py`: offline test of every package and model. Prints the hardware line, timings and PASS/FAIL per step.

## Before you start

- **Python 3.12** (3.11 or 3.13 also work). Python 3.14 is too new for the TTM package, and 3.10 is too old.
  Check with `py -0p`. If 3.12 is missing, install it for your user only (no admin):
  `winget install --id Python.Python.3.12 --scope user`, or the python.org installer with "for all users" unticked.
- **About 3.8 GB free** on the drive with the cache (plus 2 GB margin). The script checks this.
- A good connection: about 1.7 GB to download in total.
- Copy this `laptop_setup` folder to the laptop if the project share isn't there. Any folder works.

## Run it

Open a normal PowerShell window (not the ISE), go to this folder and run:

```powershell
cd "<repo>\tools\laptop_setup"
powershell -ExecutionPolicy Bypass -File .\setup-fh2026.ps1 -WithSodir
```

It takes about 10-20 minutes the first time. Everything goes to `%USERPROFILE%\fh2026-cache`. Use `-CacheRoot D:\fh2026-cache` to put it somewhere else. The cache must not be inside a git repo (the script refuses).

At the end you get a summary table. **ALL STAGES PASSED** means you're ready. If something fails, fix it and run the same command again: it skips what's already done.

## Options

| Option | What it does |
|---|---|
| `-CacheRoot <path>` | Where the venv, data and weights go (default `%USERPROFILE%\fh2026-cache`) |
| `-Python <path>` | Use this python.exe instead of auto-detecting |
| `-WithSodir` | Also download the Sodir FactPages CSVs (~10 MB) |
| `-RefreshSodir` | Download the Sodir CSVs again even if they're present |
| `-SkipInstall` | Don't create or update the Python environment |
| `-SkipData` | Don't download the FORCE 2020 data |
| `-SkipModels` | Don't download model weights, and skip the model tests |
| `-SkipDocs` | Don't download the offline docs |
| `-SkipSmoke` | Don't run the smoke test |
| `-SmokeOnly` | Only run the smoke test |
| `-Force` | Carry on even if disk space looks low |

## What ends up in the cache

```
fh2026-cache\
  venv\                    Python env (CPU-only torch 2.11)
  pip-cache\               downloaded wheels (lets you rebuild the venv offline)
  data\force2020\          train.csv/zip, leaderboard + hidden test CSVs, penalty_matrix.npy,
                           Zenodo LAS zip (all 118 wells), NPD tops/casing xlsx
  data\sodir_factpages\    wellbores, lithostrat tops, fields, licences (CSV) + SOURCE.txt
  docs\                    DuckDB docs (PDF + MD), lasio, pdfplumber, MOMENT/Chronos/TTM
                           READMEs and notebooks, FORCE retrospective, NLOD 2.0 text
  hf\                      Hugging Face model weights (MOMENT, Chronos, TTM)
  fh2026-env.ps1           activates the env in a new window
  pip-freeze.txt           exact package versions installed
```

## Later, at the hackathon

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. "$env:USERPROFILE\fh2026-cache\fh2026-env.ps1"
python -c "import duckdb, lasio; print('ok')"
```

This activates the venv, sets `HF_HOME` and offline mode (`HF_HUB_OFFLINE=1`), and sets `FH2026_DATA` to the data folder. The repo code should read the data path from config or `FH2026_DATA`, never from a path inside the repo.

To re-check everything offline: `powershell -ExecutionPolicy Bypass -File .\setup-fh2026.ps1 -SmokeOnly`

## Notes

- **momentfm** 0.1.4 pins numpy 1.25, transformers 4.33 and huggingface-hub 0.24, which don't install on Python 3.12. The script installs it with `--no-deps`. `pip check` then lists those three pins as conflicts. That's expected. MOMENT loads and gives identical outputs with transformers 4.57 and 5.23 (checked).
- **torch** is pinned to 2.11 (CPU build) because granite-tsfm 0.3.12 needs torch >=2.10 and <2.12.
- **PyMuPDF** is left out on purpose: it's AGPL, which doesn't fit an MIT repo. `pdfplumber` (MIT) covers PDF tables and text.
- **Licences:** MOMENT-1-small is MIT. Chronos-Bolt and Chronos-2 are Apache-2.0. TTM r2 is Apache-2.0. Don't commit model weights. The FORCE 2020 Zenodo record is CC BY 4.0, and the original well logs are NLOD 2.0. Sodir FactPages are NLOD 2.0 (credit Sodir).
- First run on Windows: Hugging Face may warn about symlinks. That's harmless (the warning is switched off, and it just copies files).
