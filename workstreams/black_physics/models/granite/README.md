# Granite-experiment baselines (model card)

Opt-in (`WMB_BLACK_GRANITE=1`, NumPy only) station rows from the 50 m density-gap experiment
([../../granite/README.md](../../granite/README.md)). The experiment's foundation-model rows (TTM-R3, TSPulse,
Timestral target-only) are registered by the red workstream:
[red_foundation_models/models/granite](../../../red_foundation_models/models/granite/README.md).

| ID | What it does | Inputs |
| --- | --- | --- |
| `black_linear_gap` | straight line in depth between the nearest measured values above and below (the station's reference rule) | own flanks |
| `black_gardner_flank_resid` | fixed Gardner (0.31·Vp^0.25) from DTC, plus the RHOB − Gardner residual on the flanks interpolated linearly across the gap | DTC + RHOB flanks |

**Units.** RHOB in g/cm³; depth in metres; DTC in µs/ft (converted to Vp).

**Predictions.** A full-depth array, NaN outside `Task.predict_mask`; no interval. Whole-curve (`missing_curve`)
tasks have no flank context, so both rows abstain on them.

**Licence:** MIT (these wrappers; Gardner 1974 is a published equation).

```
WMB_BLACK_GRANITE=1 python -m benchmark.check_model black_gardner_flank_resid
WMB_BLACK_GRANITE=1 python -m benchmark.run --model black_linear_gap,black_gardner_flank_resid --test rhob_gap50
```
