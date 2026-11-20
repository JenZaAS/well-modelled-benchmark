# Benchmark tab: good / ok / poor rules

The live viewer's Benchmark tab colours every score as good, ok or poor. All rules and colours live in one module,
[`viewer/benchmark_quality.js`](../viewer/benchmark_quality.js); change them there (the Node tests in
`tests/test_benchmark_ui.js` and `tests/test_benchmark_quality.js` pin the boundaries). Boundaries are inclusive.

| Measure | Direction | good | ok | poor |
| --- | --- | --- | --- | --- |
| **Accuracy** (`MAE`) | lower | within 10% of the best model in the same column | 10-30% worse | more than 30% worse |
| **Calibration** (`cover_90`) | closest to 0.90 | within 3 points of 0.90 (0.87-0.93) | within 8 points (0.82-0.98) | otherwise |
| **Sharpness** (`interval_width_90`) | narrower | within 15% of the narrowest *calibrated* model | 15-45% wider | wider still |
| **Speed** (`predict_s_per_well`) | lower | real-time, up to 0.05 s per well | batch-OK, up to 0.5 s | slow, above 0.5 s |

- Calibration below 0.90 reads as *overconfident*, above as *too cautious*.
- Sharpness only counts for a model whose calibration is good or ok. A poorly calibrated model's sharpness is greyed out
  as "doesn't count".
- The speed dial is a log scale from 0.001 to 100 s per well: `position = clamp((log10(t) + 3) / 5, 0, 1)`.

## Trade-off scatter

The summary scatter defaults to accuracy (y) vs sharpness (x), with each point filled by **calibration** of the 90%
band. Pickers above the plot choose the X axis, Y axis and colour from accuracy, sharpness, calibration (points from
90%) and speed; colour "None" restores the model-colour fill with the calibration ring. Picking the other axis's
measure swaps the axes. All measures are lower-is-better, so bottom-left is always best; the Linear-centred quadrants
in All mode only appear when both axes are accuracy / sharpness.

The fill runs red (worse) → amber → green (better) via `goodness()` in `benchmark_quality.js`, anchored on the rules
above: 0 off / the best model = full green, the good boundary = 0.75, the ok boundary = 0.4, twice the ok boundary
or beyond = red. Accuracy and sharpness are measured against the lowest value shown; speed uses a log scale
(5 ms green … 5 s red).

## All mode

"All" never averages raw values across properties (the units differ). Using `template_linear` as the baseline:

- Accuracy = mean over properties of `MAE / baseline MAE`; sharpness = mean of `interval_width_90 / baseline width`.
  1.0 is the baseline, lower is better. The 10/30% and 15/45% rules apply to these relative scores.
- Calibration = mean of `|cover_90 - 0.90|` (shown in points); the 3 / 8 point rules apply to that mean.
- Speed = mean of `predict_s_per_well`.

A property missing for the baseline or for the model is left out of that model's mean. Without `template_linear` in
the run, All is disabled.

## Runs and header

One run is one dataset + scenario (`gap_50`, `missing_curve`). Rows from different properties are merged into one run; the
test name only splits a run when the same model and property were scored under several tests. The header shows
what the scores carry: test wells and samples, run time (Europe/Oslo), git commit (`git_commit`, written by
`benchmark.run`) and hardware. Missing fields are omitted. Results written before `git_commit` existed still load.
