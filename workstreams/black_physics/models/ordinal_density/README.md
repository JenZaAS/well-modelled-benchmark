# Density-range probabilities for RHOB gaps (model card)

**What these rows do, in plain words.** A gap-filling model draws one line through a hidden stretch of density log.
These two rows keep that line and add, at every hidden depth, a probability for each of 15 density ranges: below
1.60, 1.60–1.70, …, 2.80–2.90, and 2.90 g/cm³ and above. The viewer's Benchmark tab shows the probabilities as a
heatmap behind the line (Well logs → Uncertainty: Bin heatmap). Darker means more likely, and hovering shows the
likeliest ranges at that depth. The method, its scores and its caveats are in
[../../ordinal_density/README.md](../../ordinal_density/README.md).

**Read this first.**

- **The heads were fitted and chosen on the station's development wells,** not on the test wells: the base model's
  errors on the C wells' 50 m gaps, with the head chosen on the D wells. Their numbers on other gap lengths, other
  targets or other base models have not been checked.
- **The point estimate is not new.** p50 is the base model's own fill, so MAE and R² equal the base row's. What
  changes is the uncertainty: the 90 % and 50 % intervals and the new `RPS_bins` score.
- **RPS_bins compares only rows with the same 15 bins.** It is the ranked probability score of the binned forecast:
  0 is perfect, lower is better.
- **Neither row foresees a sharp streak that the flanks do not show.** For an example, see the pitch gap in the
  method README.

| ID | Base fill (unchanged) | Head (frozen) | Environment |
| --- | --- | --- | --- |
| `black_ordinal_timestral` | `red_timestral`: Timestral-5M, RHOB flanks + other logs | the base model's past errors, grouped by how variable the gap's flanks are | Timestral (`timestral-cpu`) |
| `black_ordinal_tspulse_finetune` | `red_tspulse_finetune`: TSPulse R1 after the stronger fine-tune, RHOB only | a softmax layer on the fill, position in the gap, flank values and the Gardner(DTC) difference | Granite (`granite-cu126`) |

**Results on `rhob_gap50`** (20 test wells, one 50 m gap each, 6,600 hidden depths):

| ID | RPS_bins | 90 % interval: coverage / width (g/cm³) | 50 % interval coverage | MAE (g/cm³) |
| --- | --- | --- | --- | --- |
| `black_ordinal_timestral` | 0.0189 | 0.97 / 0.245 | 0.74 | 0.0333 |
| `black_ordinal_tspulse_finetune` | 0.0276 | 0.95 / 0.290 | 0.72 | 0.0460 |

The station reproduces the method's own test scores for the same heads exactly: RPS 0.018946 and 0.027582. Each MAE
equals that of the cached base fills the heads were fitted on (0.033314 and 0.045959), so the base models ran
unchanged. Both intervals cover more than their nominal level, so they are somewhat wide on the test wells. The
heads were chosen on D without temperature scaling ("raw"), and no interval was tuned to a target coverage.

**Output.** `Prediction.p50` is the base fill. `p05`, `p25`, `p75` and `p95` are quantiles of the binned
distribution. Where a quantile would fall on the wrong side of p50, it is moved onto p50, and
`provenance["band_widened"]` counts these moves. `bin_edges` holds the 14 internal edges (1.60 to 2.90) and
`bin_probs` is an `[n_depth, 15]` array of probabilities. Its rows are NaN outside the gap. Whole-curve
(`missing_curve`) tasks have no gap flanks, so both rows abstain (NaN).

**Frozen heads.** The heads live in `<work_root>/black_ordinal_density/frozen/<base model>.json` and are not in
git (the Timestral head is 4 MB of past errors). [frozen_heads.yaml](frozen_heads.yaml) records the sha256 of each
head and of the inference code. A row refuses to run (`StaleHead`) if a head is missing or edited, if the code has
changed, or, for TSPulse, if the base checkpoint is not the one the head was fitted on. To rebuild the heads and
the lock:

```bash
export PYTHONPATH="$PWD/src:$PWD"
python -m workstreams.black_physics.granite.decide_any predict --model red_timestral        # Timestral env
python -m workstreams.black_physics.granite.decide_any predict --model red_tspulse_finetune # Granite env
python -m workstreams.black_physics.ordinal_density all --test rhob_gap50                  # Granite env (SciPy)
python -m workstreams.black_physics.ordinal_density freeze --test rhob_gap50
```

**Run.** Each row runs in its base model's environment. The flag registers both rows.

```bash
WMB_BLACK_ORDINAL=1 HF_HOME=<hf cache> python -m benchmark.run --model black_ordinal_timestral --test rhob_gap50
WMB_BLACK_ORDINAL=1 python -m benchmark.run --model black_ordinal_tspulse_finetune --test rhob_gap50
```

**Licence.** MIT for the wrapper and the heads. Each base model keeps its own licence: Timestral-5M weights are
Modified MIT (see [its row](../../../red_foundation_models/models/timestral/LICENSE.md)), and IBM Granite
TSPulse is Apache-2.0.
