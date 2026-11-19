# TimesFM 2.5 + XReg for well logs

**Why this third contender:** a thin bed can appear as a sharp transition in
several logs at the same measured depth. [TimesFM 2.5 with XReg](https://github.com/google-research/timesfm#update---oct-29-2025)
can use those visible channels *at the missing samples*. In each direction we
fit upstream ridge regression (penalty 1) to the supplied context, evaluate it
on the known covariates, and let the frozen 200M model forecast the residual.
The covariate contribution is added at native depth resolution, not averaged
over an input patch. There is no clock-frequency label, resampling, or change
to the shared 256-sample context, 32-sample rollout, or training-well split.
P10/P50/P90 come from model deciles, not a Gaussian fitted to the point estimate.

| Alternative considered | Trade-off for this benchmark |
|---|---|
| [Moirai 1.1 small](https://huggingface.co/Salesforce/moirai-1.1-R-small) | Its 8-sample patch option and probabilistic multivariate design are attractive, but weights are CC BY-NC 4.0. |
| [Granite TTM R2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) | Lightweight, with high-resolution pretraining; its published zero-shot path is point forecasting and its exogenous recipes involve fine-tuning. |
| [TimesFM 3.0](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights) | Native covariates, but downloaded weights are restricted to non-commercial, non-production use. Version 2.5 is Apache-2.0. |

This is a candidate for **correlated** fine structure, not a way to infer an
unobserved, unpredictable spike. A relationship estimated on one flank or a
training well may break at a formation boundary. Quantiles describe residual
forecast uncertainty, not uncertainty in that relationship.

## Local setup and verification

From the repo root, in a separate Python 3.11 environment:

```sh
python -m pip install -e . -r workstreams/red_foundation_models/models/timesfm25/requirements.txt
python -m wmb.paths --check
export HF_HOME="$(python -c 'from wmb.paths import cache_dir; print(cache_dir() / "hf")')"
hf download google/timesfm-2.5-200m-pytorch config.json model.safetensors --revision 1d952420fba87f3c6dee4f240de0f1a0fbc790e3
export HF_HUB_OFFLINE=1 WMB_RED_MODELS=timesfm25 OMP_NUM_THREADS=2
python -m benchmark.check_model red_timesfm25
python -m benchmark.run --model red_timesfm25 --test synthetic_rhob_missing
python -m workstreams.red_foundation_models.experiment --model timesfm25 --dataset force2020 --target RHOB --scenario gap --split validation --shots 0 --seed 0
```

The last command uses only a holdout of training wells and reports paired depth
interpolation on the same samples. Keep it separate from official test scores.
