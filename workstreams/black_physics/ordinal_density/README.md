# Where is a reconstructed density log uncertain? (15-bin probabilities)

## What we did, in plain words

A well log is a curve of one rock property measured down a borehole. Bulk density (RHOB, in g/cm³) is one of them.
Logs often have holes. The station test `rhob_gap50` hides one real 50 m stretch of RHOB in each of 20 FORCE 2020
test wells and asks a model to fill it.

Two models fill the same hidden stretch:

- **TSPulse fine-tune** (`red_tspulse_finetune`). A small IBM time-series model. It sees only the density measured
  above and below the hole.
- **Timestral-5M + other logs** (`red_timestral`). Mistral's small time-series model, frozen. It also sees the
  *other logs* at every depth, including inside the hole. These are other measurements made in the same borehole at
  the same depths:
  - gamma ray, deep and medium resistivity, calliper and compressional sonic, all present in every test gap;
  - shear sonic, neutron porosity and photoelectric factor, present in most.

  Only density is hidden. The full list is in [../granite/RESULTS.md](../granite/RESULTS.md).

A fill is one number per depth. This module turns each fill into a **forecast over 15 fixed density ranges**:

| Bin | 1 | 2 | 3 | … | 14 | 15 |
|---|---|---|---|---|---|---|
| RHOB | below 1.60 | 1.60–1.70 | 1.70–1.80 | … | 2.80–2.90 | 2.90 and above |

So each depth gets 15 probabilities that add to 1. A value exactly on an edge belongs to the higher bin. Plotted down
the hole, the probabilities make a heatmap that shows where the fill is sure and where it is not. The pitch shows the
two heatmaps first, then reveals the measured curve.

**What turns a fill into probabilities (the "heads").** The base models are never retrained. Every head is fitted on
other wells (group C), and its one calibration number on further wells (group D). The test wells are touched only
for the final scores.

- **Empirical errors (pooled).** Take the fill's errors (measured − filled) on the 20 C wells. Shift them by the new
  fill and count how many land in each bin. Every well counts equally.
- **Empirical errors by flank variability.** The same, with one error set for each third of *flank variability*.
  Flank variability is how much the measured density wiggles just above and below the hole. A wiggly neighbourhood
  gets a wider forecast. A depth with no measured neighbour uses the pooled set.
- **Numeric head.** A small learned classifier (one linear layer, 15 outputs, softmax). It reads a few numbers per
  depth: the fill; the position in the hole; the distance to, and value of, the nearest measured density above and
  below; and the flank variability. For Timestral it also reads Timestral's own P10–P90 width. TSPulse's band is
  left out because it is a constant width set on D.
- **Numeric head + sonic check.** The same, plus one physics feature: how far the fill is from the density that the
  fixed Gardner equation predicts from the sonic log, and whether sonic is present.
- **Decima-small.** A small pretrained decision model (`amyrmahdy/decima-small`, INT8 ONNX on CPU, revision pinned).
  It reads the same depth state as text and answers *"At the marked depth, which range contains the measured bulk
  density?"*, with the 15 ranges as ordered options. It is not trained here.
- **Temperature.** Each head gets one number, fitted on the 12 D wells, that sharpens or flattens its
  probabilities. Raw and calibrated versions are both scored. On D, each well's calibrated score uses a temperature
  fitted on the other 11 D wells.

**How it is scored.**

- **Ranked probability score (RPS).** At each of the 14 edges, compare the forecast chance of being below the edge
  with what happened (0 or 1), and average the squared gaps. 0 is a perfect sure forecast. A forecast one bin off
  costs more than a spread-out forecast, and two bins off costs more than one. Scores are averaged over depths in a
  gap, then gaps in a well, then wells.
- **Coverage.** How often the measured value falls inside the central 80 % and 90 % ranges, and how wide those ranges
  are.
- **"Below 2.40" event.** The chance that density is below 2.40 g/cm³: the sum of bins 1–9. It is scored with the
  Brier score and a reliability plot.
- **Heatmap choice.** Each model's heatmap head is the one with the lowest RPS on D, not on test.

## Results (run `rhob_gap50-5a53a2cd66`, 2026-11-20)

**Read these limits first.**

- **This is a new comparison on a known benchmark, not a fresh blind test.** These 20 test gaps were already used
  to report the base models' point errors. The heads and their selection never saw them, but we did know before
  starting that Timestral's fills are better there.
- **20 test gaps are 20 pieces of evidence, not 6,600.** Depths inside a gap move together, so every interval below
  comes from resampling wells. Apart from Decima, the differences between heads are small: 0.0004–0.003 in RPS, about 2–11 %.
- **The test gaps are easier than the development gaps.** TSPulse's point error is 0.046 g/cm³ on test against 0.071
  on D. So 80 % ranges that cover about 80 % on D cover 89–94 % on test, and the heads look over-cautious there.
- **The heatmap choice made on D did not hold up on test for TSPulse.**
  - D picked the numeric head with the sonic check (D RPS 0.0383 against 0.0391 for empirical errors by flank
    variability).
  - On test that head scores 0.0276 against 0.0259.
  - We keep the D choice, as planned, and report this.
- **The heads widen the forecast but do not move it.** Every head is centred on the base fill. A sharp feature that
  the fill misses stays missed: see the low-density streak in the pitch figure.

**Test scoreboard (20 wells, all 6,600 hidden depths).** RPS is averaged depth → gap → well; lower is better.
◆ marks the heatmap head, chosen on D.

| Head (raw) | TSPulse fine-tune | Timestral + other logs |
|---|---|---|
| Empirical errors, pooled | 0.0266 | 0.0194 |
| Empirical errors by flank variability | **0.0259** | **0.0189** ◆ |
| Numeric head | 0.0287 | 0.0214 |
| Numeric head + sonic check | 0.0276 ◆ | 0.0214 |
| Heatmap head: 80 % range covers (mean width) | 89 % (0.23 g/cm³) | 94 % (0.17 g/cm³) |
| Heatmap head: 90 % range covers (mean width) | 95 % (0.29) | 97 % (0.25) |
| Heatmap head: same on D, 80 % range | 82 % (0.23) | 85 % (0.19) |
| Model's own band, for reference | 90 % band covers 96 % (0.34)¹ | 80 % band covers **39 %** (0.04) |
| Brier score of P(RHOB < 2.40), heatmap head | 0.045 | 0.032 |
| Point error of the fill / of the binned median | 0.046 / 0.049 | 0.033 / 0.035 |

¹ TSPulse's band was calibrated on D, so its 90 % coverage on D is in-sample.

All rows, the calibrated versions and the paired intervals are in `scoreboard.md` in the run folder.

**What the numbers say.**

1. **The sharper base model gives the better forecast.** Timestral's heatmap beats TSPulse's in 18 of 20 test wells:
   RPS difference +0.0086 [+0.0051, +0.0125] in Timestral's favour. On D it wins in 11 of 12 wells (+0.0087
   [+0.0055, +0.0116]). Its 80 % ranges are also narrower (0.17 against 0.23 g/cm³).
2. **Timestral's own band is badly overconfident; the heads fix it.** Its nominal 80 % band covers 39 % of the hidden
   values on test (38 % on D) at a width of 0.04 g/cm³. Its heatmap head's 80 % range covers 94 % on test and 85 % on
   D, about 4× wider.
3. **Simply counting past errors is the best head.**
   - Empirical errors by flank variability beat the pooled errors by a small margin:
     - TSPulse −0.0007 [−0.0014, +0.0000], 13 of 20 wells;
     - Timestral −0.0004 [−0.0008, −0.0000], 14 of 20 wells.
   - The learned numeric head is worse by about +0.002:
     - TSPulse +0.0021 [−0.0000, +0.0041];
     - Timestral +0.0020 [+0.0004, +0.0035].
   - Its penalty was chosen by grouped 4-fold cross-validation inside C. All four numeric heads picked the smallest
     value, 1e-5; the fixed 1e-3 we started with was clearly worse.
   - Likely reason: with logits linear in the fill, a linear softmax can only slide a Gaussian-shaped bump. The fill
     errors on C have much heavier tails than a Gaussian (excess kurtosis 7.4 for TSPulse, 10.6 for Timestral),
     and counting the errors keeps their real shape.
4. **The sonic check helps only the model that does not already read sonic.** Adding the
   Gardner-from-sonic disagreement to TSPulse's numeric head:
   - D: −0.0024 [−0.0041, −0.0006], 10 of 12 wells;
   - test: −0.0011 [−0.0025, +0.0005], 16 of 20 wells.

   For Timestral, which already reads DTC: test +0.0000 [−0.0006, +0.0008], D +0.0031 [−0.0008, +0.0104]. DTC gave
   a usable Gardner value at 100 % of test depths, 96 % of D and 95 % of C.
5. **Decima-small is far behind.** Matched on 8 sampled depths per gap of the TSPulse fills (704 D and 160 test
   states, the same depths for every head):

   | | Decima-small | Empirical errors by flank variability |
   |---|---|---|
   | Test RPS, raw | 0.089 | 0.026 |
   | D RPS, raw | 0.099 | 0.039 |
   | Wells where the empirical head is lower | | 19 of 20, Δ −0.063 [−0.089, −0.039] |
   | Raw 80 % coverage (width) | 52 % (0.36 g/cm³) | 88 % (0.20) |
   | Run time | 143 ms per state | about 2 µs per depth |

   - Decima's forecast barely moves. Its mean stays between 2.27 and 2.32 g/cm³ while the density estimates it
     reads range from 1.85 to 2.67: correlation 0.10 on D and −0.05 on test.
   - Its most likely bin is always 2.20–2.30 or 2.30–2.40, so it answers with roughly the same prior every time.
   - Its D temperature (2.13) flattens it further, giving 88 % coverage at 0.60 g/cm³ width.
   - Its raw D RPS (0.099) matches the earlier pilot (0.101 on 176 depths).
6. **Temperature calibration did not help.**
   - Every fitted temperature was above 1 (1.05–1.55), so D's log loss asked for flatter forecasts.
   - Leave-one-well-out on D, calibrated and raw RPS are nearly equal.
   - On the easier test gaps, the flatter forecasts score slightly worse.
7. **The "below 2.40 g/cm³" event is reliable on D for both models.** On test the middle probabilities have few
   depths (see the counts in the reliability figure). There, TSPulse says 0.2–0.5 where the observed share is lower.
8. **Bin resolution.**
   - Predicting each hidden value's own bin centre would give an MAE of 0.026 g/cm³. That is the floor this 0.10
     g/cm³ grid can resolve.
   - 3 test depths fall in the top open bin (≥ 2.90) and none in the bottom one.
   - Neither tail needed the declared one-bin extension: C has 50 values below 1.60 and 22 at 2.90 or above.

**Stretch: the two-scale hybrid did not apply.** We split each fill with a 2 m Gaussian smoother into a broad part
and a fine remainder. Timestral had the lower C error at both scales: broad 0.034 against 0.042 g/cm³, fine 0.026
against 0.030. So "broad from one model, fine from the other" is Timestral alone, and the test was not scored. On D,
Timestral alone (0.054) also beats the equal average of the two fills (0.060) and TSPulse (0.071).

**Figures** from run `rhob_gap50-5a53a2cd66`, committed in [figures/](figures/). They show FORCE 2020 well data,
which is shareable with the credit line each figure carries. The run's scoreboard, scores, scales, config and
pitch-gap choice are in [report/](report/); the frozen heads and per-depth probabilities stay local under
`<work_root>/black_ordinal_density/`, so re-running the station rows needs that folder.

- `fig_pitch_frame1.png`, `fig_pitch_frame2.png`: the two pitch frames.
- `fig_scoreboard.png`: test RPS of every head, raw and calibrated, with the matched Decima panel.
- `fig_reliability_below_2.40.png`: the below-2.40 event on D and test, with depth counts.

**The pitch gap was chosen without the hidden values.** It is the test gap with the median flank variability among
gaps measured on both sides: well 25/5-3, 1933–1983 m. On that gap:

| | TSPulse | Timestral |
|---|---|---|
| Gap RPS | 0.040 | 0.026 |
| 80 % range covers (width) | 84 % (0.27 g/cm³) | 87 % (0.19 g/cm³) |

The measured curve has a low-density streak at 1960–1972 m: 56 % of its depths are below 2.10 g/cm³. Neither
forecast catches it. Both give "below 2.10" only 13–16 % there, and both 80 % ranges miss about half of the streak.

**In the station and the viewer.** The two selected heads are also registered as station rows,
`black_ordinal_timestral` and `black_ordinal_tspulse_finetune` (opt-in, `WMB_BLACK_ORDINAL=1`; model card:
[../models/ordinal_density/README.md](../models/ordinal_density/README.md)). Each row runs its base model unchanged,
adds the probabilities per hidden depth (`Prediction.bin_edges` / `bin_probs`) and is scored as `RPS_bins`.
TSPulse reports the 15 bins. Timestral's empirical head reports the same forecast on 0.02 g/cm³ bins
(`bins.fine_edges(0.02)`, 67 bins; `heads.EmpiricalHead.predict(..., edges)`). They add up exactly to its 15
(`bins.coarsen`), and its quantiles come from them (`bins.quantile_on`). A softmax head or a temperature is fitted
on the 15 bins, so `infer.Head.predict` refuses finer bins for those. On `rhob_gap50` the station reproduces this
README's 15-bin test scores exactly: 0.018946 for Timestral (its 67 bins added up) and 0.027582 for TSPulse. The
viewer's Benchmark tab draws the probabilities as a heatmap (Well logs → Uncertainty: Bin heatmap), with the darkest
colour at the well's largest probability. Hovering over a depth shows the forecast's median, 80 % range and
likeliest range.

**Not done, on purpose.** Decima on Timestral's fills: the plan's budget ran Decima on the TSPulse fills only.

## The thirty-second pitch

> Two models fill the same hidden 50 metres of density in well 25/5-3. Instead of one line, each gives 15
> probabilities per depth, one for each tenth of a g/cm³ — that's the shading. TSPulse sees only the density above
> and below the hole, and its band is wide. Timestral also reads the other logs through the hole, and its band is
> narrower. Now the measured curve. Across all 20 test wells the narrower forecast is also the better one: ranked
> probability score 0.019 against 0.028, better in 18 of 20 wells, with 80 % ranges about 0.17 g/cm³ wide that hold
> the truth 94 % of the time. Timestral's own 80 % band held it only 39 % of the time. The best way to get these
> probabilities was the simplest: count the model's past errors. A pretrained decision model, Decima-small, scored
> three times worse. And the streak at 1960 metres? Neither model saw it coming. That's exactly where you send a
> person.

## How to run

The heads read saved fills, so no model has to run again. They need the cached fills from
`python -m workstreams.black_physics.granite.decide_any predict --model <id>` (see
[../granite/README.md](../granite/README.md)), which live under
`<work_root>/black_granite/decision_any/preds_<model>.npz`. Run from the repository root with `src` on the path:

```bash
export PYTHONPATH="$PWD/src:$PWD"
python -m workstreams.black_physics.ordinal_density prepare --test rhob_gap50   # join fills to tasks, build depth states
python -m workstreams.black_physics.ordinal_density fit --test rhob_gap50       # heads on C, temperatures on D
python -m workstreams.black_physics.ordinal_density score --test rhob_gap50 --include-decima
python -m workstreams.black_physics.ordinal_density plot --test rhob_gap50
python -m workstreams.black_physics.ordinal_density scales --test rhob_gap50    # stretch: two-scale hybrid
python -m workstreams.black_physics.ordinal_density freeze --test rhob_gap50    # selected heads -> station rows
```

`freeze` copies each model's selected head to `<work_root>/black_ordinal_density/frozen/<model>.json`. It also
writes the heads' and the inference code's sha256 to
[../models/ordinal_density/frozen_heads.yaml](../models/ordinal_density/frozen_heads.yaml), which is committed.
The station rows refuse a head that does not match, so after re-fitting, re-run `freeze` and then the benchmark.

- **Models.** `--models` defaults to `red_tspulse_finetune,red_timestral`. Older ids such as `black_tspulse_ft`
  resolve through `wmb.aliases`.
- **Where outputs go.** `<work_root>/black_ordinal_density/<run-id>/`. The run id is the test name plus a hash of
  the frozen configuration and the cached fills, so every stage finds the same folder. Each stage adds its settings
  to `config.json` and prints the folder when it ends.
- **Decima.** `--include-decima` runs the pinned checkpoint in its own environment (`WMB_DECIMA_PYTHON`, default
  `%LOCALAPPDATA%/wmb-envs/decima-cpu`; set up as in [../granite/decima_score.py](../granite/decima_score.py)). Its
  answers are cached by state, model, revision, precision, question and options in
  `<work_root>/black_ordinal_density/decima_cache.jsonl`.
- **Run time.** `prepare` takes about 15 s, `fit` about 10 min (mostly the grouped cross-validation of the numeric
  heads), `score` about 15 s plus about 3 min for Decima the first time, and `plot` about 15 s.
- **Tests.** `python -m pytest workstreams/black_physics/tests/test_ordinal_density.py
  workstreams/black_physics/tests/test_ordinal_rows.py` (synthetic, NumPy and SciPy only).

**Using a head on new fills.** [infer.py](infer.py) is the reusable entry point:

```python
from workstreams.black_physics.ordinal_density.infer import Head
head = Head.load(run_dir, "red_timestral")      # the head chosen on D, with its temperature
head, rec = Head.load_frozen("red_timestral", sha256)   # or the frozen copy, checked against its hash
out = head.predict(task, p50, lo, hi)            # masked Task + the base fill; returns probs [n_hidden, 15]
```

It reads only the masked task and the fill, and needs NumPy only (no SciPy), so it runs inside a model's own
environment. A head fitted on one model's errors can be applied to another
model's fills, but it is then uncalibrated.

## Files

| File | What it holds |
|---|---|
| [bins.py](bins.py) | The 15 fixed bins and the edge rule; mean, median and interval readouts; RPS, log loss and P(< 2.40). |
| [features.py](features.py) | The per-depth state, built from the masked task and the fill only. |
| [heads.py](heads.py) | Empirical-error heads, the softmax head with grouped-CV penalty, temperature calibration and row weights. |
| [scoring.py](scoring.py) | Depth → gap → well averaging, paired well bootstrap, coverage, reliability and the bin-resolution floor. |
| [decima.py](decima.py), [decima_runner.py](decima_runner.py) | The matched Decima-small comparison: state text, the 8-per-gap sample, the keyed cache, and the runner for Decima's own environment. |
| [cli.py](cli.py) | The stages, the keyed join to the tasks, validation, the scoreboard and the sidecar. |
| [plots.py](plots.py) | Scoreboard, reliability plot and the two pitch frames. |
| [scales.py](scales.py) | Stretch: the two-scale hybrid. |
| [infer.py](infer.py) | Reusable inference on any masked task and fill; frozen heads and their hash checks. |

**Outputs in the run folder.** None of these are committed:

- `truth.npz` (hidden values, used only for labels and scores) and `prepared_<model>.npz` (depth states);
- `heads_<model>.json` (fitted heads, tails, penalties and temperatures) and `probs_<model>.npz`;
- `sidecar_<model>_test.npz`: the heatmap head's `[6600, 15]` probabilities with well, gap, index, depth, the base
  p50, the probability-weighted mean (the pitch estimate) and the binned median (p50) stored separately;
- `scores.json`, `scoreboard.md`, `decima_<model>.npz`, `scales.json`, `pitch_gap.json`, the figures and `config.json`.
