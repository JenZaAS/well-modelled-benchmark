# Red team: foundation models for missing well logs

`red_chronos2` (Amazon Chronos-2) and `red_timestral` (Mistral Timestral) use the
same depth-window adapter. `red_floor_train_mean` remains a no-model baseline.
Run from the repository root with the project installed (`pip install -e .`).
Use local inference only; never upload wells or commit data, weights, or predictions.

## Backbones and weights

- **Chronos-2:** install `chronos-forecasting==2.3.2` and `transformers>=4.57.6,<5`
  in an ML environment. Pre-cache `amazon/chronos-2` via `hf download amazon/chronos-2`;
  use the same `HF_HOME` at inference. The existing laptop forward-pass check is
  `check_chronos2` in `tools/laptop_setup/fh2026_smoke_test.py`.
- **Timestral-5M:** the [upstream inference code](https://github.com/mistralai/timestral)
  is Apache-2.0, but the [checkpoint licence](Timestral-5M/LICENCE)
  ([pinned upstream copy](https://huggingface.co/mistralai/Timestral-5M/blob/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/LICENSE))
  is **Modified MIT**, with an organization revenue restriction. Do not fetch,
  redistribute, or use those weights unless your organization has confirmed
  authorization and benchmark eligibility. The optional local adapter does not
  bundle the weights. Its [module docstring](timestral_backend.py) gives the
  pinned install and cache commands for an authorized run.

## Run two different tasks

1. **Missing section:** mask reproducible 20-50 m intervals on held-out wells;
   score only originally observed, deliberately hidden samples. The red-only
   experiment uses the shared loader, split, gap mask and metrics. It does not
   publish gap scores into the whole-curve scoreboard.

   ```sh
   python -m wmb.paths --check
   python -m workstreams.red_foundation_models.experiment --model chronos2 --dataset force2020 --target RHOB --scenario gap --max-wells 1
   ```

2. **Entire curve missing:** use the unchanged shared benchmark. Opt in only
   to installed backbones so core-only runs do not need either ML stack.

   ```sh
   WMB_RED_MODELS=timestral python -m benchmark.run --dataset force2020 --target RHOB --contenders red_timestral
   ```

The same commands accept `synthetic` for a wiring check, or a supported LAS
manifest dataset. FORCE uses its official split. CSV and LAS are input formats,
not independent samples: do not add a LAS copy of a held-out FORCE well to training.
Paths come from `wmb.paths` and the untracked `data_paths.local.yaml`.

## Representation and experiments

- **Depth, not clock time.** One native depth sample is one model step; measured
  depth in metres is also a numeric covariate. Windows stay in one monotonic
  well segment and break at jumps over three times the well's median spacing.
  We do not interpolate the target to a regular grid. Smaller spacing variations
  remain an approximation; compare fixed-metre resampling before pooling datasets
  with materially different sampling densities.
- **Section vs. whole curve.** For a gap, forecast from up to 256 observed
  samples on each available flank, reversing the deeper flank. Roll forward in
  blocks of 32 using predicted medians, never hidden targets. Blend the two
  forecasts by inverse depth distance. With no flank, prompt the frozen model
  with one contiguous labelled window per selected *training* well (first usable
  wells by name) and average the forecasts. This is a demonstration baseline,
  not a claim that different wells are geologically adjacent.
- **Continuous plus discrete.** Available non-target core logs are numeric
  covariates over the gap. `--facies` adds one binary channel per training facies
  and an unknown indicator; facies codes have no numeric order. Absent `LITH`
  is allowed. Only enable this if facies would independently be known at
  prediction time: interpretations inferred from the hidden target would leak it.
- **Point and uncertainty.** Return depth-aligned P10/P50/P90. The benchmark
  point estimate is P50; `(P90 - P10) / 2.563` approximates one Gaussian sigma.
  Blended intervals are heuristics: check coverage alongside MAE, not just width.
- **Compare adaptations.** Repeat the gap command with `--model timestral`, then
  with/without `--facies` using the same seed, target and wells. Check
  `wells_with_facies`, scored `n` and `masked_samples` before comparing MAE.
  For `--scenario whole`, compare `--shots 1` and `--shots 4`; `--shots 0`
  intentionally abstains when no target context exists. For bounded gaps with
  visible flanks no demonstrations are needed. Neither model updates weights.
  For a fine-tuning experiment, partition the training wells into fit and
  validation wells before cutting windows. Train on simulated depth gaps in
  fit wells, select window length/checkpoint on validation wells, and evaluate
  once on the untouched test split against the frozen baseline.

## FORCE 2020 density-gap validation

From the repository root, run the command below twice, adding `--facies` on the
second run. `--split validation` uses a seeded 20% holdout of training wells,
not the leaderboard or hidden wells. Linear depth interpolation uses the same
visible target flanks and is scored on the same samples as the model.

```sh
python -m workstreams.red_foundation_models.experiment --model chronos2 --dataset force2020 --target RHOB --scenario gap --split validation --shots 0 --seed 0
```

For Timestral, replace `--model chronos2` with `--model timestral` in its authorized local environment.
Frozen checkpoints, no weight updates; 20 validation wells, 12,001 masked
RHOB samples, all scored. Errors are pooled over samples, in g/cm³.

| Input | MAE | RMSE | Within one predicted sigma |
| --- | ---: | ---: | ---: |
| Linear depth interpolation | 0.0731 | 0.1034 | n/a |
| Chronos-2, continuous logs | 0.0479 | 0.0920 | 41.0% |
| Chronos-2, logs + interpreted facies | 0.0500 | 0.0901 | 46.2% |
| Timestral-5M, continuous logs | 0.0474 | 0.0702 | 37.4% |

Keep facies off by default: RMSE improved slightly but MAE worsened, and
interpreted labels may not be independently available. All model runs were
overconfident relative to nominal 68% one-sigma coverage. This is one target
and one mask seed; it is not evidence about whole-curve recovery or official test performance.

## Local Timestral dashboard

From the repository root, activate the authorized Timestral environment and set
`HF_HOME` to the same external cache used to download the checkpoint. On macOS/Linux:

```sh
python -m pip install streamlit==1.55.0 plotly==6.5.0 lasio==0.32
export PYTHONPATH="$PWD/src:$PWD" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
python -m wmb.paths --check
python -m streamlit run workstreams/red_foundation_models/dashboard.py --server.address=127.0.0.1 --server.headless=true --browser.gatherUsageStats=false
```

Open the localhost URL printed by Streamlit; stop it with Ctrl+C. Do not expose
the app to the network: the new-well page reads local files. Neither view saves
log tracks or predictions to disk. The dashboard does not change model discovery.

- **Benchmark:** choose a reference dataset and target, then run validation. The
  inventory includes the unused test split. Metrics are pooled over the same
  masked samples for both methods. Select a well to compare predictions to the
  deliberately hidden observations. These are frozen-model results, not training losses.
- **New well:** enter a local LAS or semicolon-separated FORCE CSV path (columns
  `WELL`, `DEPTH_MD`, and recognized log columns). Read wells, select one, and
  predict missing samples. The model uses visible target flanks or one training
  demonstration if an entire segment lacks the target. Observations are not
  scored as predictions. A blank forecast means the model abstained.
- Depth increases downward. P10/P90 are model quantiles, not guaranteed coverage;
  use the validation coverage statistics to judge uncertainty before relying on them.
