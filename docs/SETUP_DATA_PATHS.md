# Set up local data paths

Well data is **not** in git. Each person keeps a local copy of the shared team
data folder, often in a different place. A gitignored file at the repo root,
`data_paths.local.yaml`, tells the code where that copy is on this machine.

For a walk-through that you can run again later (including the official
FORCE 2020 demo dataset), use python -m wmb.onboarding. Re-run it with
--status or --redo. Progress is stored in gitignored .wmb/onboarding_state.json.
Dataset download instructions live under [datasets/](../datasets/README.md).

This page is written so a user can point a coding agent (GitHub Copilot, Cursor,
Claude, or similar) at it and the agent can finish the setup.

## What the agent should do

1. Check whether `data_paths.local.yaml` already exists at the repository root.
   If it does, run `python -m wmb.paths --check` and stop unless the user wants
   to change the folder.
2. If the file is missing, ask the user for the path to their **shared team
   data folder**. That folder should have one subfolder per dataset. The
   expected names are:
   - `FORCE2020`
   - `Sodir-FactPages`
   - `NLOG-DTS`
   - `Kansas-KGS-DTS`
   - `Volve`
   Not every dataset has to be present. A missing subfolder is fine; the check
   command will list it as not found.
3. Verify the path exists and is a directory. For each of the subfolders that
   are present, confirm it contains `.las` / `.LAS` and/or `.csv` / `.CSV`
   files (they may be in nested well folders). Tell the user if a present
   subfolder looks empty.
4. Write `data_paths.local.yaml` at the repository root (see the template
   below, or copy `data_paths.example.yaml`). Use forward slashes in the path,
   and quote it if it contains spaces. Only add the optional per-dataset keys
   when a dataset lives *outside* `data_root`.
5. **Never commit** `data_paths.local.yaml`. It is listed in `.gitignore`.
   Do not `git add` it, and do not include it in a commit.
6. Run `python -m wmb.paths --check` from the repository root and show the
   user which datasets were found. If PyYAML is missing, install it with
   `pip install pyyaml` (or `pip install -r requirements.txt`) and rerun.

## Template

```yaml
# Local data locations for this machine. Do not commit this file.
data_root: "/path/the/user/gave"

# Optional overrides and cache — uncomment only if needed:
# force2020: "/path/to/FORCE2020"
# sodir_factpages: "/path/to/Sodir-FactPages"
# nlog_dts: "/path/to/NLOG-DTS"
# kansas_kgs_dts: "/path/to/Kansas-KGS-DTS"
# volve: "/path/to/Volve"
# cache_dir: "/path/to/fh2026-cache"
```

## How the code finds the file

1. The environment variable `WMB_DATA_PATHS` (a path to the yaml), if set.
2. Otherwise `data_paths.local.yaml` at the repository root.

If neither is present, `FH2026_DATA` is still accepted as a fallback for
`data_root`. If that is also unset, the code stops with an error that points
here. Other optional env vars: `FH2026_WORK`, `FH2026_CACHE`, `WMB_RESULTS`.

In Python: `from wmb.paths import data_root, get_dataset_path` then
`get_dataset_path("force2020")` (also `sodir_factpages`, `nlog_dts`,
`kansas_kgs_dts`, `volve`).
