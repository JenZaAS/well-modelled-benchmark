# Red foundation models

| Model | Enable with `WMB_RED_MODELS` | Benchmark name | Licence |
| --- | --- | --- | --- |
| [Chronos-2](chronos2/) | `chronos2` | `red_chronos2` | [Apache-2.0](chronos2/LICENSE.md) |
| [Timestral-5M](timestral/) | `timestral` | `red_timestral` | [Modified MIT weights; Apache-2.0 inference code](timestral/LICENSE.md) |
| [TimesFM 2.5 + XReg](timesfm25/) | `timesfm25` | `red_timesfm25` | [Apache-2.0](timesfm25/LICENSE.md) |

All use the [shared depth adapter](../depth_model.py) and are registered
explicitly in [`../contender.py`](../contender.py), not by scanning this directory.
Their per-target inputs are declared in
[`../red_chronos2.model.yaml`](../red_chronos2.model.yaml),
[`../red_timestral.model.yaml`](../red_timestral.model.yaml), and
[`../red_timesfm25.model.yaml`](../red_timesfm25.model.yaml): no log is required
(depth plus visible target or training demonstrations), and every other core log
is an optional covariate. Check a model in its own environment, for example:

```sh
WMB_RED_MODELS=timestral python -m benchmark.check_model red_timestral
WMB_RED_MODELS=timestral python -m benchmark.run --model red_timestral --test all
```

See the [workstream README](../README.md) for setup and run commands and
[moving-models.md](../../../docs/moving-models.md) for the layout.
