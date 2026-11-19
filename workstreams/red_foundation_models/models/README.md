# Red foundation models

| Model | Enable with `WMB_RED_MODELS` | Benchmark name | Licence |
|---|---|---|---|
| [Chronos-2](chronos2/) | `chronos2` | `red_chronos2` | [Apache-2.0](chronos2/LICENSE.md) |
| [Timestral-5M](timestral/) | `timestral` | `red_timestral` | [Modified MIT weights; Apache-2.0 inference code](timestral/LICENSE.md) |

Both use the [shared depth adapter](../depth_model.py) and are registered
explicitly in [`../contender.py`](../contender.py), not by scanning this directory.
See the [workstream README](../README.md) for setup and run commands and
[moving-models.md](../../../docs/moving-models.md) for the layout.
