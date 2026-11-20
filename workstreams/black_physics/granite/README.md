# Granite 50 m density-gap experiment (handoff)

**Question.** One real 50 m stretch of bulk density (RHOB) is hidden in a well. Two small IBM Granite time-series
models see only the density measured above and below the gap; Mistral's Timestral-5M runs with and without the
other logs. How well do they fill it, and can a cheap rule tell, gap by gap, whether to trust the fill?

**Read first:** [RESULTS.md](RESULTS.md). It opens with a plain-language description of the experiment: what is
hidden, what each row sees (including what the "other logs" are), how wells are split and how everything is scored.

**Status (2026-11-20).**
- All stages ran on a laptop (Quadro T1000 4 GB, Windows): prepare, train, select, calibrate, decide, freeze, the
  station runs, final, timestral, finetune, finetune_score and report.
- Aggregate results are in [RESULTS.md](RESULTS.md).
- Per-gap rows, figures and checkpoints are local under `<work_root>/black_granite/` and are not committed.

## Result in one paragraph

On the station test `rhob_gap50` (one hidden 50 m gap in each of the 20 FORCE 2020 test wells):

- **Best row: Timestral-5M + other logs** (frozen, red's adapter): mean per-well MAE 0.033 g/cm³, against 0.061
  for the straight line. It beats the line in 18 of 20 wells, and 16 of 20 gaps are within 0.05.
  - It also beats TTM-R3 tuned in 16 of 20 wells: Δ +0.014 [+0.006, +0.023] in Timestral's favour.
- **The other logs are what make the difference.** Timestral target-only gets 0.047, a tie with the density-only
  Granite rows: TTM tuned − Timestral target-only is Δ +0.0007 [−0.0023, +0.0045]. Adding the other logs is worth
  Δ −0.0135 [−0.0221, −0.0062].
- **Timestral's intervals are overconfident.** Its nominal 90 % band covers 46 % of hidden samples. The
  D-calibrated Granite bands cover 96 %, but are 5–7× wider (0.30–0.36 against 0.055 g/cm³).
- **Selected Granite row.** TSPulse tuned was chosen in advance on the B wells. Its mean per-well MAE is 0.049 g/cm³.
  - The straight line gets 0.061 and fixed Gardner from sonic gets 0.100.
  - It beats the line in 15 of 20 wells. The paired bootstrap Δ is −0.012 [−0.027, −0.000].
- **Other Granite rows.** TTM-R3 tuned did best on test at 0.0475. Its difference from TSPulse tuned is within
  noise: Δ −0.002 [−0.007, +0.003].
- **Fine-tuning: the brief's run barely trained. A proper run helps a little.**
  - The brief's head-only fine-tune stopped after 75–125 updates, in under 5 seconds. The stopping rule (3 checks
    without improvement, 25 updates apart) reacted to noise. TSPulse could only change its last 200 weights, and
    TTM was trained on something other than what it is scored on. It changed almost nothing: TTM Δ −0.0006
    [−0.0017, +0.0004]; TSPulse Δ −0.0001.
  - The stronger fine-tune ([finetune.py](finetune.py)) fixes those problems:
    - training more of the network, up to all of it;
    - 8× more training gaps (13,824);
    - TTM trained on its blended two-sided output, the thing it is scored on;
    - batch 32, a warm-up plus cosine schedule, patience 10 checks, up to 12 min;
    - three settings per model, picked on B.
    Every setting beat the pretrained model on B.
  - **TSPulse, whole model** (1.07 M weights, 2,900 updates, 2.7 min): 0.0460 on test, Δ −0.0035 [−0.0070, −0.0008]
    against pretrained, about 7 %. That holds up on test as well as B. It is now the best density-only row,
    level with Timestral target-only (Δ −0.0009 [−0.0040, +0.0022]).
  - **TTM, prediction heads** (1.08 M weights, 1,800 updates, 3.5 min) is credibly better on B (Δ −0.0024) but
    not on test: 0.0471, Δ −0.0010 [−0.0046, +0.0028]. Training the whole TTM did not beat training its heads on B.
  - Neither comes close to Timestral with the other logs (0.033). Fine-tuning buys about 0.003; the other logs buy
    about 0.014.
- **Two-line physics baseline.** Gardner + flank residual gets 0.056.
- **Decision rule.** The decision layer sits on the selected Granite row, as the brief specifies, not on
  Timestral. A numeric logistic rule decides from gap features whether the fill is within 0.05 g/cm³:
  - At p ≥ 0.80 it accepted 7 of 20 test gaps with 0 false accepts and sent 13 to review.
  - Its Brier score is 0.158, against 0.261 for the base rate.
- **Decima-small.** Reading the same facts as text, it gave p_pass of 0.47–0.53 for every gap and accepted none.

## Commands

From the repository root, with the Granite environment's Python (see *Environments*):

```
python -m workstreams.black_physics.granite fetch        # pinned TTM-R3 and TSPulse snapshots into the HF cache
python -m workstreams.black_physics.granite all          # prepare, train, select, calibrate, decide (+ freeze)
WMB_BLACK_GRANITE=1 python -m benchmark.check_test rhob_gap50
WMB_BLACK_GRANITE=1 python -m benchmark.check_model black_ttm_r3_tuned        # likewise for the other five IDs
WMB_BLACK_GRANITE=1 python -m benchmark.run --model black_ttm_r3_frozen,black_ttm_r3_tuned,black_tspulse_frozen,black_tspulse_tuned,black_linear_gap,black_gardner_flank_resid,black_gardner_fixed --test rhob_gap50 --predictions npz --rerun
python -m workstreams.black_physics.granite final        # sealed test gaps once: keyed predictions + decision sidecar
# Timestral environment (see Environments), HF_HOME=<cache_dir>/hf, HF_HUB_OFFLINE=1:
WMB_RED_MODELS=timestral WMB_BLACK_TIMESTRAL=1 <timestral-python> -m benchmark.run --model red_timestral,black_timestral_target_only --test rhob_gap50 --predictions npz --rerun
<timestral-python> -m workstreams.black_physics.granite timestral   # both Timestral rows on the same keyed gaps
python -m workstreams.black_physics.granite finetune       # stronger fine-tune sweep (extension, ~1 h GPU)
python -m workstreams.black_physics.granite finetune_score # its D bands, B/D rows and the test rows
python -m workstreams.black_physics.granite report       # tables, bootstrap, figures, 30-second text, RESULTS.md
python -m pytest -q workstreams/black_physics/tests/test_granite.py               # NumPy-only (any env)
python -m pytest -q workstreams/black_physics/tests/test_granite_torch.py         # real networks (Granite env)
```

Each stage can be run on its own: `prepare`, `train [--arm ttm|tspulse]`, `select`, `calibrate`, `decide`,
`freeze`. Every stage appends to `<work_root>/black_granite/pipeline.log`.

On this laptop, `all` takes about 2 minutes, and `final` and `report` under a minute each. The Timestral station
run takes about 3.5 minutes on CPU, and `timestral` about 2 minutes.

## Design

| Piece | Choice (all values in [config.py](config.py); `config_hash` identifies a run) |
| --- | --- |
| Wells | 98 FORCE training wells, split by SHA-256(seed 2026 + well ID) into A 54 (fit heads), B 12 (pick checkpoint and row), C 20 (fit decision rules), D 12 (calibrate bands and recalibrate rules). The 20 test wells stay sealed until `final`. |
| Gaps | Placed by the station's own `place_gap` (50 m, one acquisition segment, fully measured). A: up to 32 overlapping gaps per well (1,728). B/C/D: up to 8 non-overlapping per well (91 / 148 / 88). Test: the station test definition, one gap per well (20, 6,600 samples). |
| TTM-R3 | Forecast 512 → 336 down from the shallow flank, and up from the reversed deep flank. Blend with weight u = depth fraction across the gap. P50 is the blended q0.5; the raw band is the blended q0.1 / q0.9. |
| TSPulse | One 512-sample window, centred on the gap and clipped to the segment. The gap goes in as a user mask (the mask token replaces the values); the reconstruction is read in raw units. |
| Timestral-5M | Frozen `mistralai/Timestral-5M@bdd20cf`, red's `DepthForecaster` unchanged: ≤ 256-sample flanks, rollout in blocks of 32, deeper flank reversed, inverse-distance blend. `red_timestral` gets every log the test offers as covariates (GR, RDEP, RMED, CALI, DTC on all 20 gaps; PEF 19, NPHI 17, DTS 13). `black_timestral_target_only` passes empty covariate dictionaries. Bands are nominal (Gaussian from its P10/P90), not D-calibrated. |
| Fine-tuning | Head only. AdamW, lr 3e-4, weight decay 0.01, effective batch 8 (microbatch picked by 10 measured steps), ≤ 300 updates or 45 min, evaluated on B every 25 updates, patience 3. Each run starts with an overfit-and-reload check: 2 examples, loss down, and the reload is exact. TTM uses pinball loss over 9 quantiles, with two directed examples per A gap. TSPulse uses MAE on the hidden samples. |
| Stronger fine-tune (extension) | [finetune.py](finetune.py). Up to 256 gaps per A well (13,824). TTM: pinball on the blended two-sided output; scope head (lr 1e-3) or full (1e-4, 3e-5). TSPulse: MAE on hidden samples; scope decoder (1e-4) or full (1e-4, 3e-5). AdamW, wd 0.01, batch 32, 100-update warm-up then cosine to 10 %, clip 1.0, eval on B every 100 updates, patience 10, ≤ 4,000 updates or 12 min. Best B evaluation wins; bands refitted on D. Picked: TTM head, lr 1e-3 (B 0.0774); TSPulse full, lr 1e-4 (B 0.0748). |
| Selection | One row among the four: lowest B mean well-level MAE with full support. Result: TSPulse tuned (0.07885). The ft rows came later and do not change it. |
| Bands | Fitted on D. Width = k·s, where s is the local TTM P90 − P10 (floor 0.005) or 1 for TSPulse; k is an equal-well-weighted quantile of \|error\|/s, giving P05/P95 and P25/P75. Raw TTM 80 % coverage on D is recorded: 0.78 frozen, 0.80 tuned. |
| Decision label | Gap MAE ≤ 0.05 g/cm³ on a complete gap. |
| Decision features | Width proxy, flank-median difference, predicted range, context lengths, plus Gardner disagreement and DTC fraction. The features are written as a deterministic text state with no well name or outcome ([features.py](features.py)). |
| Decision methods | Prevalence; width-only logistic; numeric L2 logistic with and without Gardner features (all fitted on C); intercept/slope recalibration on D; Decima-small INT8 ONNX on CPU (`verify` question, p_pass = P(yes)). Accept at p ≥ 0.80. |
| Freezing | `manifest.json` records the code hash (inference files), config, split, data and gap hashes, head SHA-256 and calibrators. The station wrappers refuse stale artifacts. |

## Deviations from the brief and things to know

1. **Short TTM contexts.** The brief pads short contexts to 512 as unobserved. The frozen TTM sanity run on B
   then diverged on short flanks: forecasts up to 35 g/cm³, B MAE 0.26.
   - TTM uses the observed mask only in its scaler, so a 30-sample context padded to 512 is out of distribution.
   - I compared padding and minimum-context rules on A gaps only (500 gaps, frozen TTM) and kept padding
     unobserved.
   - New rule: a flank feeds TTM only if it has ≥ 256 contiguous measured samples (`ttm_min_context`).
     Otherwise that side is dropped (one-sided forecast) or the straight line is used.
   - On A, the worst gap MAE fell from 14.8 to 0.30 g/cm³.
   - The broken B number was seen before the rule was set. B was not used to choose the 256 value.
2. **Tuned checkpoint.** The tuned row uses the best evaluation after at least one update, even when step 0 scored
   better on B. That happened for TTM: step 0 got 0.07986 and the best tuned head 0.08042, at update 50.
   - So TTM tuned is a genuinely tuned head, and the four-row selection on B still decides which row is used.
   - `train_ttm.json` records `step0_won`.
3. **The brief's fine-tune barely ran (read this before quoting "tuned" rows).** Patience stopped TTM at 75 updates
   and TSPulse at 125, in under 5 s each. That is about 20 % of one pass over the TTM training examples. The
   45-minute budget was never used.
   - Updates cost 0.02–0.06 s, so compute was never the limit. The patience of 3 checks, 25 updates apart,
     stopped on B noise of about ±0.0005.
   - TTM-R3's prediction heads hold 1.08 M of 3.16 M parameters. The TSPulse time head holds only 200.
   - The "tuned" rows are kept as the brief's matched pair. The stronger fine-tune (*ft* rows,
     [finetune.py](finetune.py)) is the real answer to "does FORCE training help?".
   - The ft rows were designed after the head-only test results were known. All their choices (scope, learning
     rate, checkpoint, bands) use A, B and D only. They were scored on test once.
   - They are not registered station contenders yet: the pipeline scores them with the same reconstructor code
     as the station rows, which matches the station to 2e-8 elsewhere.
4. **Headline decision method.** The headline rule is the one with the lowest D Brier ([report.py](report.py)
   `headline_method`); it is the numeric logistic.
   - This rule uses development data only, but it was written after `final` had run. All eight methods are
     reported.
   - The four numeric variants give the same accept/review split on test.
5. **Test pass rate.** It is 0.65 on test, against 0.49 on C and 0.40 on D. The test gaps (station placement, one
   per well) are easier than the development gaps, so the prevalence rule is miscalibrated on test.
6. **Station check.** The pipeline's own scoring matches `benchmark.run` to within 2e-8 g/cm³ for all nine
   rows; both use the identical gap IDs.
7. **Timestral ran after the first report.** Both Timestral rows were scored after `final` and the first report,
   in their own CPU environment, with frozen weights and red's adapter unchanged. Nothing about them was tuned or
   chosen on test data.
   - The decision layer and the B selection are Granite-only, as the brief specifies, so they did not change.
   - Timestral's bands come from its own quantiles and are not D-calibrated. Its coverage is therefore not
     comparable with the Granite bands.

## Step 8: material to choose from

All files are in `<work_root>/black_granite/report/`. Figures show measured FORCE curves, so keep them local or
share them deliberately.

| File | What it shows | Good for |
| --- | --- | --- |
| `figures/pitch_median.png` | **Brief's pitch figure.** Left: the representative gap (deterministic: selected-row MAE closest to the median), with flanks, hidden truth, P50, 90 % band, Gardner and the straight line. Right: p_pass against actual gap error for all 20 test gaps, with accepted and review counts. | Main slide |
| `figures/pitch_best.png`, `pitch_worst.png` | The same layout on the easiest and the hardest test gap | Showing honest failure, Q&A |
| `figures/leaderboard.png` | Mean per-well MAE for all nine rows, with bootstrap CIs | One-glance result |
| `figures/paired_timestral.png` | Per well: Timestral with − without the other logs; density-only TTM − Timestral; TTM − Timestral with logs | "Is it the model or the inputs?" |
| `figures/gallery_test_gaps.png` | All 20 gaps sorted by error, with accept/review marks, plus the TTM and Timestral + logs fills | Backup slide; credibility |
| `figures/paired_deltas.png` | Per-well tuned − frozen (both arms) for the brief's head-only run, and selected − line | "Did the brief's fine-tune matter?" (no) |
| `figures/finetune_sweep.png` | B error over training for every setting of the stronger fine-tune, against pretrained and the brief's run | "What does real fine-tuning do?" |
| `figures/paired_finetune.png` | Per well: stronger fine-tune − pretrained, both models | "Did it help on unseen wells?" |
| `figures/decisions_risk_reliability.png` | Risk–coverage curves and 3-bin reliability for the rules and Decima | The decision story |
| `figures/coverage.png` | Observed 50 % / 90 % coverage and band width per row | Uncertainty honesty |
| `figures/decision_coefficients.png` | What the numeric + Gardner rule weighs | Explaining the rule |
| `figures/training_curves.png` | B metric and training loss over updates | Appendix |
| `figures/extension_vs_linear.png` | Per gap: Gardner + flank residual and the selected row against the line | Physics extension |
| `REPORT.md` | All tables, per-gap rows and hashes | Speaker notes |
| `results_table.json` | Replayable table: every number, the hashes behind it (code, config, split, data, gaps, heads, predictions, sidecar) and the replay commands | Audit |
| `../final/decision_sidecar.json` | Small keyed sidecar: one row per test gap, with gap ID, checkpoint, features, text state, p_pass for every method, MAE and label | Joining with the viewer |
| `../final/predictions.npz` | Full-precision keyed predictions: `<row>\|<gap_id>\|p50/p05/p25/p75/p95/raw_lo/raw_hi`, plus `truth\|` and `depth\|` | Re-scoring |
| `../final/timestral.json`, `timestral_predictions.npz` | The same, for the two Timestral rows (timing, covariates offered, package versions) | Re-scoring |
| `../final/finetune.json`, `finetune_predictions.npz`; `../finetune_<arm>.json` | The same for the two ft rows, plus every sweep run's history, D calibrators and the checkpoint hashes | Re-scoring, audit |

The 30-second text is at the top of [RESULTS.md](RESULTS.md); `report` regenerates it from the numbers.

**Possible stories:**
- (a) "The other logs matter more than the model": leaderboard plus paired_timestral. From density alone, three
  foundation models tie at about 0.048. Give Timestral the other logs and it drops to 0.033.
- (b) "Small foundation models beat interpolation and physics on density gaps, and know when they are right":
  pitch_median plus leaderboard.
- (c) "Accurate is not the same as honest": coverage. Timestral is the most accurate, but its 90 % band covers
  46 %. The calibrated Granite bands cover 96 %.
- (d) "Inputs beat training": finetune_sweep plus paired_finetune next to paired_timestral. Three minutes of
  laptop fine-tuning buys about 0.003 g/cm³ for TSPulse. Giving the model the other logs buys about 0.014.
- (e) "Physics plus flanks is a cheap strong baseline": extension_vs_linear.

## Files

| File | Role |
| --- | --- |
| [config.py](config.py) | Frozen settings, pinned revisions, config hash, work folder |
| [gaps.py](gaps.py) | Split, gap index, segment geometry, flank runs, linear fill, task construction (NumPy only) |
| [backends.py](backends.py) | TTM-R3 / TSPulse loading at pinned revisions, head selection, batching |
| [reconstruct.py](reconstruct.py) | Gap geometry and status for both arms (backend injected; unit-tested with fakes) |
| [train.py](train.py) | Examples, losses, overfit/reload, step measurement, time-boxed head training |
| [calibrate.py](calibrate.py) | Equal-well-weighted empirical bands |
| [features.py](features.py), [decision.py](decision.py) | Gap features and text state; logistic rules, recalibration, decision metrics, risk–coverage |
| [decima_score.py](decima_score.py) | Stand-alone Decima-small scorer for its own CPU environment (`python -I`) |
| [artifacts.py](artifacts.py) | Manifest identity, stale-artifact checks, variant loading for the station wrappers |
| [pipeline.py](pipeline.py), [report.py](report.py) | Stages and the step-8 report |
| [finetune.py](finetune.py) | Stronger fine-tune extension: broader unfreezing, more A gaps, blended two-sided TTM loss, schedule, sweep, scoring |
| [../models/granite/](../models/granite/README.md) | Station wrappers, model YAMLs, licences |
| [../tests/test_granite.py](../tests/test_granite.py), [../tests/test_granite_torch.py](../tests/test_granite_torch.py) | Split disjointness and reproducibility; non-overlapping measured gaps; segment breaks; flank reversal and distance weights; short flanks, fallback and abstain; TSPulse window and user mask; label-perturbation invariance; state text; nested bands and coverage; output shapes; Gardner residual and missing DTC; decision metrics and ties; tuned checkpoint changes only the head |

## Environments

These are not part of `pyproject`. The Granite and Decima environments need conflicting `transformers` and
`torch` setups, so each lives in `%LOCALAPPDATA%\wmb-envs\`.

- **Granite** (Python 3.12):
  - Packages: torch 2.10.0+cu126, transformers 4.57.6, numpy, pandas, scikit-learn 1.7.2, matplotlib, pytest,
    `pip install -e .`
  - granite-tsfm from a sparse checkout of commit `31fa2f0`. `git clone -c core.longpaths=true`, and exclude
    `notebooks/` because a plain pip git install fails on Windows path length. Then `pip install <checkout>`.
- **Decima** (Python 3.12, CPU):
  - `pip install --no-deps decima-ai==2.5.1`, then onnxruntime 1.32.0, transformers 4.57.6, tokenizers,
    huggingface_hub, numpy. No torch.
  - The pipeline finds it through `WMB_DECIMA_PYTHON`, defaulting to `%LOCALAPPDATA%\wmb-envs\decima-cpu`.
  - If it is missing, the Decima rows are marked unavailable and the run continues.
- **Timestral** (Python 3.12, CPU; `%LOCALAPPDATA%\wmb-envs\timestral-cpu`):
  - The pins from red's [backend.py](../../red_foundation_models/models/timestral/backend.py): torch 2.8.0 (CPU
    wheel), numpy 2.2.6, safetensors 0.6.2, huggingface_hub 0.35.3 and timestral at commit `a534bdb`. Then
    `pip install -e ".[dev]"`.
  - Weights: `mistralai/Timestral-5M` at revision `bdd20cf`, downloaded once into `<cache_dir>/hf`. Inference
    uses `HF_HUB_OFFLINE=1`.
  - Licence: Modified MIT, with a revenue clause for companies over USD 20 M a month (see red's LICENSE.md).
