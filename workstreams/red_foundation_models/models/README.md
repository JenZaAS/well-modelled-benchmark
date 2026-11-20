# Red foundation models

| Model | Enable with `WMB_RED_MODELS` | Benchmark name | Licence |
| --- | --- | --- | --- |
| [Chronos-2](chronos2/) | `chronos2` | `red_chronos2` | [Apache-2.0](chronos2/LICENSE.md) |
| [Timestral-5M](timestral/) | `timestral` | `red_timestral` | [Modified MIT weights; Apache-2.0 inference code](timestral/LICENSE.md) |
| [TimesFM 2.5 + XReg](timesfm25/) | `timesfm25` | `red_timesfm25` | [Apache-2.0](timesfm25/LICENSE.md) |
| [TTM r2](ttm/) | `ttm` | `red_ttm` | [Apache-2.0](ttm/LICENSE.md) |
| [MOMENT-1-small](moment/) | `moment` | `red_moment` | [MIT](moment/LICENSE.md) |
| [IBM Granite TTM-R3 lite and TSPulse R1, density gaps](granite/) | `granite` | `red_ttm_r3_frozen`, `red_ttm_r3_tuned`, `red_ttm_r3_ft`, `red_tspulse_frozen`, `red_tspulse_tuned`, `red_tspulse_ft` | [Apache-2.0](granite/LICENSE.md) |
| [Timestral-5M target-only control](timestral_target_only/) | `timestral_target_only` | `red_timestral_target_only` | [as Timestral](timestral_target_only/LICENSE.md) |

Chronos-2, Timestral and TimesFM use the [shared depth adapter](../depth_model.py); TTM and
MOMENT use the [TSFM depth adapter](../tsfm_depth_model.py), which adds quantile-band support
for fine-tuned TTM. All are registered explicitly in [`../contender.py`](../contender.py), not
by scanning this directory, and all are opt-in so a core-only laptop never loads a backbone.
Combine keys with commas, e.g. `WMB_RED_MODELS=ttm,moment`. The older `WMB_WHITE_MODELS`
variable (keys `ttm`, `moment`) is still honoured.

Their per-target inputs are declared in
[`../red_chronos2.model.yaml`](../red_chronos2.model.yaml),
[`../red_timestral.model.yaml`](../red_timestral.model.yaml),
[`../red_timesfm25.model.yaml`](../red_timesfm25.model.yaml),
[`../red_ttm.model.yaml`](../red_ttm.model.yaml), and
[`../red_moment.model.yaml`](../red_moment.model.yaml): no log is required
(depth plus visible target or training demonstrations), and every other core log
is an optional covariate (TTM and MOMENT accept but do not use them). Check a model in its own environment, for example:

```sh
WMB_RED_MODELS=timestral python -m benchmark.check_model red_timestral
WMB_RED_MODELS=timestral python -m benchmark.run --model red_timestral --test all
```

The Granite rows and the Timestral target-only control come from the 50 m density-gap experiment in
[black_physics/granite](../../black_physics/granite/README.md), which trains, calibrates and scores them.
The Granite rows need that experiment's Granite environment and frozen artifacts. They only reconstruct
RHOB gaps (target-only) and abstain on whole-curve tasks. The control is `red_timestral` with the other
logs removed. They moved here from black_physics: the flags `WMB_BLACK_GRANITE` / `WMB_BLACK_TIMESTRAL` still
enable them, and the old `black_*` ids are aliases.

`white_ttm` and `white_moment` are deprecated aliases for `red_ttm` and `red_moment`
(see `src/wmb/aliases.py`): old commands still resolve and old results still display
under the new ids. MOMENT needs special install handling: see [moment/SETUP.md](moment/SETUP.md).

See the [workstream README](../README.md) for setup and run commands and
[moving-models.md](../../../docs/moving-models.md) for the layout.
