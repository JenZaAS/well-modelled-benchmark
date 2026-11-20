# MOMENT setup (`red_moment`)

How to make `red_moment` (AutonLab `MOMENT-1-small`) runnable next to the other
foundation models without breaking them. Items are marked **Tested** or **Untested**.

## Why `momentfm` needs special handling

`momentfm==0.1.4` declares pins `transformers==4.33.3`, `huggingface-hub==0.24.0` and
`numpy==1.25.2`. Applying them would break Chronos-2 and other shared models.
`tools/laptop_setup/README.md` and `setup-fh2026.ps1` also note that these pins are too
old for Python 3.12.

- Never install `momentfm` with dependency resolution in a shared venv.
- Never downgrade `torch`, `transformers` or `scikit-learn`.
- Do not disable SAC.

## Install

**Tested** (the only supported command):

```
python -m pip install --no-deps momentfm==0.1.4
```

Evidence:

- `models/moment/README.md`, `requirements.txt` and `backend.py` give this command and state it
  was "verified on the box with transformers 4.57.6 and torch 2.11.0+cpu".
- `tools/laptop_setup/setup-fh2026.ps1` sets `$MomentSpec = 'momentfm==0.1.4'` and runs
  `& $VenvPy -m pip install $MomentSpec --no-deps`. Afterwards `pip check` lists the three
  pins above as conflicts; the laptop README says this is expected and harmless.
  "MOMENT loads and gives identical outputs with transformers 4.57 and 5.23 (checked)."
- Measured 2026-11-19 on the shared Linux box (Europe/Oslo, CET, UTC+1): an isolated
  venv with `momentfm 0.1.4` installed `--no-deps` on `torch 2.11.0+cpu` /
  `transformers 4.57.6` completed FORCE `rhob_gap50` (20 test wells). The `momentfm` pins
  were not applied, and the other shared venvs were left untouched. Prefer an isolated
  venv and keep the model opt-in.
- Consistent, not a new inference test: read-only metadata on the Windows laptop on
  2026-11-20 showed `momentfm 0.1.4` already present in the Python 3.12.10 user
  site-packages alongside `torch 2.11.0+cpu`, `transformers 4.57.6`, `numpy 2.5.5` and
  `huggingface_hub 0.36.2`, consistent with the pins not having been applied.
  No run was performed to write this file.

**Untested / do not do:**

- `pip install momentfm` without `--no-deps`.
- Installing the declared `transformers 4.33.3`, `huggingface-hub 0.24.0`, `numpy 1.25.2` pins.
- Downgrading `torch`, `transformers` or `scikit-learn`.
- Copying weights to any shared or synced folder.
- `MOMENT-1-large` (heavier; not the default).

## Weights

The benchmark wrapper (`backend.py`) does **not** download at runtime. It searches the
[workstream weight cache](../../weights/README.md) under `hf/` and the configured
`wmb.paths.cache_dir()/hf` (typically `%USERPROFILE%\fh2026-cache\hf` on Windows) for
`AutonLab/MOMENT-1-small`. It raises an error ("no runtime download") if the snapshot
is missing.

Prefetch is a separate step: `tools/laptop_setup/setup-fh2026.ps1` / `fh2026_fetch.py`
sets `HF_HOME` to `<cache-root>\hf` and calls
`huggingface_hub.snapshot_download(repo_id="AutonLab/MOMENT-1-small", revision="main")`.
The wrapper then loads through an explicit local config plus the safetensors file, because
newer hub mixins drop the config when building the pipeline (see the comment in `backend.py`).

The workstream weights folder is git-ignored except its README. Do not commit weights and
do not copy them into synced folders.

## Quick checks

**Import only** (lightest; from the import list of `tools/laptop_setup/fh2026_smoke_test.py`):

**Untested** (listed for completeness; this file's author did not run it):

```
python -c "import momentfm"
```

**Load the model** (not run for this file): this snippet uses the upstream loader, not
the benchmark wrapper. The laptop setup smoke test uses this loader with
`HF_HUB_OFFLINE=1` and `HF_HOME` pointing at the prefetched cache.

```python
from momentfm import MOMENTPipeline
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-small", model_kwargs={"task_name": "reconstruction"})
model.init()
model.eval()
```

**Wrapper check** (loads the weights; not run for this file):

```
WMB_RED_MODELS=moment python -m benchmark.check_model red_moment
```

On Windows PowerShell set the variable first: `$env:WMB_RED_MODELS = "moment"`.
If `momentfm` is missing, use the tested install command above. A full FORCE
benchmark run is not part of setup.
