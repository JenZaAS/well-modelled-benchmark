# red_ttm — IBM Granite TTM R2

Zero-shot, point fine-tuned, or quantile fine-tuned wrapper around
[`ibm-granite/granite-timeseries-ttm-r2`](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2).
Part of the red foundation-model workstream; enable it with `WMB_RED_MODELS=ttm`
(opt-in, so core-only laptops never load the backbone). Licence: [Apache-2.0](LICENSE.md).

- **Zero-shot / point fine-tune:** a point forecast; p10/p90 are a Gaussian interval
  around p50 using the training-set first-difference standard deviation. The
  benchmark's 90% band is then the Gaussian rule from `std`.
- **`--probabilistic` fine-tune:** TTM's native multi-quantile head is trained with
  pinball loss at q05/q50/q95. Optional split-conformal widening (CQR) uses a
  held-out set of *training* wells only (never test wells). The backend returns
  q50 as the prediction and q05/q95 as the scored 90% band (`cover_90` / `interval_width_90`).

Default window: context 512, horizon 96 (the R2 main variant), rolled to cover longer gaps.
Base checkpoints and fine-tunes live in `workstreams/red_foundation_models/weights/`
(git-ignored except its README; override with `WMB_RED_CACHE`, legacy `WMB_WHITE_CACHE` still works).
The depth adapter is [`tsfm_depth_model.py`](../../tsfm_depth_model.py).

```bash
export WMB_RED_MODELS=ttm    # legacy WMB_WHITE_MODELS=ttm also works
python -m benchmark.check_model red_ttm
python -m benchmark.run --model red_ttm --test rhob_gap50

# Point fine-tune (weights/ttm/<TARGET>/):
python -m workstreams.red_foundation_models.models.ttm.finetune_ttm --target RHOB --epochs 1

# Quantile fine-tune + conformal (weights/ttm/<TARGET>-probabilistic/):
python -m workstreams.red_foundation_models.models.ttm.finetune_ttm --target RHOB --probabilistic \
    --epochs 3 --max-wells 0 --max-windows 0 --stride 64 --batch-size 32 --max-seconds 540
```

`red_ttm` prefers a probabilistic checkpoint when present, then a point fine-tune, otherwise zero-shot.
Checkpoints written by earlier runs keep their `white_meta.json` metadata file name.

## Published DTS pilot (TTM r2)

The paired FORCE scores are under `results/{red_ttm_r2_dts_pilot_combined,red_ttm_r2_frozen}/force2020/DTS/`, one `scores.json` per named test. Reading aggregate metrics needs neither checkpoint. Per-depth predictions and trained weights are not included with these scores.

To repeat inference, obtain the combined pilot's `model.safetensors` and `config.json` from the run owner via approved team storage, **not git**. Point `WMB_TTM_DTS_PILOT` at their directory, or install them under `weights/ttm-pilots/20261120-102336/combined/` in this workstream. Both files are checked against the scored artifact; a single-well or freshly retrained model is not a substitute. The frozen control downloads the pinned IBM snapshot on first use and needs no team artifact.

From the repository root with the TTM dependencies and FORCE data configured:

```sh
WMB_RED_MODELS=ttm_r2_dts_pilot PYTHONPATH=src python -m benchmark.run \
  --model red_ttm_r2_dts_pilot_combined,red_ttm_r2_frozen --test dts_gap50,dts_missing --rerun
```

Install `granite-tsfm>=0.3.12` without downgrading `torch` or `transformers`
(see [requirements.txt](requirements.txt)). The brief for a further fine-tuning experiment is in
[FINETUNE_PROMPT.md](FINETUNE_PROMPT.md).

Earlier results were recorded under the id `white_ttm`; that id is a deprecated alias
for `red_ttm` (see `src/wmb/aliases.py`), and `results/white_ttm/` is still read.

Measured on the shared Linux box, 2026-11-19, Europe/Oslo (CET, UTC+1), FORCE `rhob_gap50`
(20 test wells, 6600 samples):

| Variant | MAE | R² | cover90 | interval_width_90 | s/well |
| --- | --- | --- | --- | --- | --- |
| Zero-shot (Gaussian band) | 0.0508 | 0.867 | 0.63 | n/a | 0.062 |
| Quantile fine-tune + CQR (67.73 s CPU train, 78+20 train/calib wells) | 0.0466 | 0.878 | 0.93 | 0.243 | 0.062 |
