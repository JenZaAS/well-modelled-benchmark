# Two-sided TTM: a learned bridge between the two forecasts (negative result)

**Result in one line.** A learned bridge between TTM's two one-sided forecasts did not help. On the D and test
wells, which were not used to pick it, it is no better than pretrained TTM. TSPulse's whole-model fine-tune remains
the best density-only row.

## Why we tried it

TTM is a forecaster. To fill a 50 m gap it makes two separate forecasts:
- down from the measured density above the gap;
- up from the reversed measured density below it.

It then blends the two with a fixed weight by position in the gap. Neither forecast sees the other side.

The stronger fine-tune ([finetune.py](finetune.py)) showed what that costs. On the D and test gaps:
- gaps with measurements on both sides (92) were unchanged (+0.5 %);
- only the gaps filled from one side (16) improved (−7 %);
- TSPulse, which sees both sides in one window, improved 7 % on its two-sided gaps.

## What was built

[two_sided.py](two_sided.py) replaces the fixed blend on two-sided gaps with a small network, the **bridge**
(5,058 weights). At each depth in the gap it reads:
- the fixed blend weight (position in the gap);
- both forecasts' medians and P10–P90 widths;
- the nearest measured density above and below;
- how far each forecast misses the measured value on the far side.

It outputs a correction to the mixing weight and an additive shift, applied to all nine quantiles. Density inputs
are taken relative to the fixed blend, so the bridge doesn't depend on the absolute density level.

Its last layer starts at zero: at step 0 the output is exactly the fixed blend. Gaps with only one usable side keep
their single forecast.

Training matches the stronger fine-tune:
- the same 13,824 A gaps;
- pinball loss on the bridged output;
- AdamW, warm-up and cosine schedule, batch 32;
- a check on the B wells every 100 updates, patience 10, at most 12 minutes.

Three settings were tried, and the best B check after at least one update won:

| Setting | Trained | Examples | B at start → best (update) | Minutes |
|---|---|---|---|---|
| `bridge` | bridge only, on pretrained TTM | 11,698 two-sided | 0.0799 → 0.0774 (300) | 6.2 |
| `bridge_on_ft` | bridge only, on the stronger fine-tune's TTM | 11,698 two-sided | 0.0774 → 0.0762 (400) | 6.3 |
| **`heads+bridge`** (selected) | bridge + TTM prediction heads (1.08 M), from pretrained | 13,806 | 0.0799 → 0.0761 (1200) | 11.7 |

Bands are calibrated on D, and the test gaps were scored once, after the checkpoint and bands were saved. The
tests ([test_granite_two_sided.py](../tests/test_granite_two_sided.py)) check that:
- the bridge starts as the fixed blend;
- it ignores the absolute density level and reacts to the far side;
- the training loss scores exactly what inference produces.

## Results

Mean per-well MAE (g/cm³). Paired differences are two-sided TTM minus each other row, with a 95 % well-level
bootstrap interval.

| Row | B (selection) | D (88 gaps) | Test (20 gaps) |
|---|---|---|---|
| TTM-R3 pretrained | 0.0799 | 0.0786 | 0.0481 |
| TTM-R3 fine-tune (heads) | 0.0774 | 0.0777 | 0.0471 |
| **TTM-R3 two-sided** | **0.0761** | **0.0795** | **0.0474** |
| TSPulse fine-tune (whole model) | 0.0748 | 0.0713 | 0.0460 |
| Timestral target-only | – | – | 0.0469 |

| Two-sided TTM − | B | D | Test |
|---|---|---|---|
| TTM pretrained | −0.0037 [−0.0064, −0.0013] | +0.0009 [−0.0014, +0.0033] | −0.0007 [−0.0062, +0.0040] |
| TTM fine-tune (heads) | −0.0013 [−0.0028, +0.0001] | +0.0019 [−0.0001, +0.0044] | +0.0003 [−0.0019, +0.0025] |
| TSPulse fine-tune | +0.0013 [−0.0007, +0.0032] | +0.0082 [+0.0035, +0.0145] | +0.0014 [−0.0036, +0.0060] |

On the gaps it was built for, those with measurements on both sides (92 D + test gaps), the mean gap MAE is 0.0677,
against 0.0661 for pretrained TTM and 0.0665 for the TTM fine-tune. One-sided gaps (16): 0.1013, against 0.1059
pretrained and 0.0984 fine-tune.

**How to read the B gain.**
- B moves by about ±1.5 % between consecutive checks of the same run.
- The winner is the best of about 50 checks across three settings.
- Even on B it was not clearly ahead of the TTM fine-tune.
- The gain disappears on D and test.

## Why it did not work, and what would

The bridge can only reweight and shift two forecasts that each saw one side. If a forecast carries no information
about the other side of the gap, nothing downstream can add it, and the fixed distance blend is already close to the
best mixing rule.

A truly two-sided TTM needs the other side *inside* the model's input, for example:
- the reversed deep flank as a second channel, with TTM's channel mixing switched on; or
- the other side's forecast as a known future input through TTM's forecast-channel mixer.

Both add new weights trained from scratch on about 12,000 gaps. Neither was tried. TSPulse already sees both sides,
so for density-only gap filling the practical choice is TSPulse.

## Station row and reproduction

`red_ttm_r3_twosided` (opt-in `WMB_RED_MODELS=granite`, Granite environment) loads the checkpoint and D bands
through `manifest_twosided.json`. The manifest checks the official-well identity and the checkpoint's SHA-256. The
row reproduces 0.0474 on `rhob_gap50`. It lives on the branch `granite-ttm-two-sided` and is not on main.

```
<granite-python> -m workstreams.black_physics.granite.two_sided train     # three settings on A, select on B (~25 min)
<granite-python> -m workstreams.black_physics.granite.two_sided score     # D bands, B/D rows, test once
<granite-python> -m workstreams.black_physics.granite.two_sided compare   # the tables above
WMB_RED_MODELS=granite <granite-python> -m benchmark.run --model red_ttm_r3_twosided --test rhob_gap50
<granite-python> -m pytest -q workstreams/black_physics/tests/test_granite_two_sided.py
```

Local outputs, never committed, under `<work_root>/black_granite/`:
- `checkpoints/ttm_twosided_best.pt`
- `twosided_ttm.json` (every run's history)
- `dev/{B,D}_black_ttm_r3_twosided.json`
- `final/twosided.json`, `final/twosided_predictions.npz`, `final/twosided_compare.json`
- `manifest_twosided.json`
