---
description: Set up and start the blue-team ML app (model comparison, well explorer, predict a new well)
mode: agent
---

# Run the blue-team ML app

Goal: get the Streamlit app in `workstreams/blue_boosted_trees/ml_study/app.py` running on this machine, then tell the
user the address. Work from the repository root. Check each step first and skip it if it is already done. Ask the user
instead of guessing whenever a step says **ask**.

## Rules

- Never commit or `git add` data, results, model files, zips or `data_paths.local.yaml`. In particular never run
  `git add -A` or `git add .`: `results/blue_ml_study/` is not ignored by git.
- Do not train or recompute anything unless the user asks (it takes 35 minutes or more). Use the two zip files instead.
- Run long commands such as the app server in the background and verify they respond.
- On Windows use PowerShell and set `$env:PYTHONUTF8="1"`. On macOS and Linux use `export PYTHONUTF8=1`.

## 1. Python environment

1. Find a Python 3.12 interpreter (3.11 or 3.13 also work). If the user already has a virtual environment for this
   repo, use it. Otherwise create one **outside** the repository, for example `~/wmb-venv`.
2. Install: `python -m pip install -r workstreams/blue_boosted_trees/requirements-app.txt`
3. Check: `python -c "import streamlit, plotly, pyarrow, xgboost, sklearn; print('ok')"`

## 2. Data location

1. Check that `data_paths.local.yaml` exists in the repository root. If not, **ask** the user for the folder that holds
   the FORCE 2020 data (a folder with `FORCE2020/` in it) and follow [docs/SETUP_DATA_PATHS.md](../../docs/SETUP_DATA_PATHS.md).
2. Check it. `wmb` is not installed as a package, so put `src` on the path:
   - PowerShell: `$env:PYTHONPATH="src"; python -m wmb.paths --check`
   - bash: `PYTHONPATH=src python -m wmb.paths --check`

   The line `force2020  found` must appear. If it does not, stop and tell the user.

## 3. The two zip files (from the team's Dropbox)

You cannot download from Dropbox without a link, so **ask** the user where these two files are on their machine
(for example `Downloads`), or for a direct-download link (a Dropbox link ending in `?dl=1`):

- `blue_ml_study.zip`: the study results, for the first two tabs.
- `blue_ml_study_models.zip`: the saved models, for the "Predict a new well" tab.

Find the two folders to unpack into:

- Results: `<repo>/results/` (the zip already contains a top-level `blue_ml_study` folder).
- Models: the work folder, from `PYTHONPATH=src python -c "from wmb.paths import work_root; print(work_root() / 'blue_ml_study' / 'models')"`.
  Create it if needed. This zip has **no** top-level folder, so unpack its contents straight into it.

Unpack, then check:

- `<repo>/results/blue_ml_study/metrics.csv` exists and there is a `predictions/` folder next to it.
- `<models folder>/manifest.json` exists and there is one subfolder per log (`GR`, `RDEP`, ...).

If the user has only one of the two zips, continue: without the results the first two tabs show "No results"; without the
models the third tab shows "No saved models". Tell the user which tab will not work.

## 4. Start the app

Run in the background from the repository root:

```
python -m streamlit run workstreams/blue_boosted_trees/ml_study/app.py --server.headless true --server.port 8501
```

If port 8501 is busy, use the next free port. Then check that it responds: `GET http://localhost:<port>/_stcore/health`
returns `ok`. The first page load takes about a minute (it reads the 20 test wells).

## 5. Tell the user

- The address, for example `http://localhost:8501`, and how to stop it (Ctrl+C in that terminal).
- What they should see: the sidebar says "9 logs, 14 model variants"; the tabs are Model comparison, Well explorer and
  Predict a new well.
- To show it to people on the same network, restart with `--server.address 0.0.0.0` and share `http://<their-ip>:<port>`.

## If something fails

| Message | Cause and fix |
|---|---|
| No results in ... | Step 3 is missing, or the sidebar "Results folder" points elsewhere. |
| FORCE 2020 data not found | Step 2: fix `data_paths.local.yaml`. |
| No saved models in ... | Step 3 (models zip), or type the folder into the "Models folder" box on the third tab. |
| No module named streamlit / plotly / wmb | Step 1 was done in another environment; for `wmb` add `PYTHONPATH=src`. |
| Version warning about scikit-learn or xgboost | The saved models are pickles made with other library versions. Install the versions named in the warning (`python -m pip install scikit-learn==<version>`). |

More background: [workstreams/blue_boosted_trees/README.md](../../workstreams/blue_boosted_trees/README.md).
