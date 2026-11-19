# poc-ml: XGBoost baselines on FORCE 2020 (proof of concept)

Small CPU-only check from 17 Nov 2026: lithology (accuracy + FORCE penalty score) and density (RHOB:
regression, quantile regression and binned classification with uncertainty).

Needs the FORCE 2020 data in the cache from `tools/laptop_setup`. Then, with the venv active:

```
set FH2026_CACHE=%USERPROFILE%\fh2026-cache   (PowerShell: $env:FH2026_CACHE = ...)
python poc_run.py              # lithology + density, a few minutes on 8 cores
python poc_run.py --only litho
python make_tables.py          # Markdown tables from results/poc-results.json
```

Data stays in `$FH2026_CACHE`; `results/` here is gitignored. Open FORCE 2020 data only. No company or client code, no client data.
