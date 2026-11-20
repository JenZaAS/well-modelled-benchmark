# Granite 50 m density gap: results (aggregate only)

Written by `python -m workstreams.black_physics.granite report` from the frozen artifacts; no well names, depths or curve values. Test `rhob_gap50`: one hidden 50 m RHOB gap in each of the 20 FORCE 2020 test wells (6,600 samples). Code hash `a00483f7fc540fff`, config `00bafc81acb0bec0`, split `e418285e7420ef26`, data `d80d06e1233764cb`; predictions sha256 `1914285b2893fcf3`, decision sidecar `65e6a89bf4d9d88c` (local files).

## What this experiment is

**The problem.** A well log is one physical measurement recorded against depth as a tool is pulled up a borehole.
Bulk density (RHOB, g/cm³) is one such log. Logs often have holes: the tool failed, the borehole wall caved in, or
a section was never logged. Petrophysicists then fill the hole by hand or with a crude rule. We test how well
automatic methods fill a hole, and whether a method can say when its fill should not be trusted.

**The test.** FORCE 2020 is a public set of Norwegian offshore wells. Its 20 official test wells were never used for
fitting or choosing anything. In each test well we hide one real, fully measured 50 m stretch of density, about
330 samples at 0.15 m spacing; the benchmark's named test `rhob_gap50` places it. Each method fills the
stretch. We then reveal the measurements and score the fill. Rock densities here mostly lie between about 2.0 and
2.7 g/cm³, so an error of 0.05 g/cm³ is about 2 %.

**What a method may see.** Every method sees the density measured above and below the hidden stretch (the
*flanks*). Only density is hidden. The well's **other logs**, recorded by other tools at the same depths, stay
visible, including inside the hidden stretch. Some methods use them and some deliberately do not:
  - **GR**: gamma ray (natural radioactivity; high in shale); present in 20 of the 20 test gaps
  - **RDEP**: deep resistivity (how strongly the rock resists current, far from the hole; fluids, porosity); present in 20 of the 20 test gaps
  - **RMED**: medium resistivity (the same, closer to the hole); present in 20 of the 20 test gaps
  - **CALI**: caliper (hole diameter; a washed-out hole spoils density readings); present in 20 of the 20 test gaps
  - **DTC**: compressional sonic slowness (how slowly a sound wave crosses the rock); present in 20 of the 20 test gaps
  - **DTS**: shear sonic slowness; present in 13 of the 20 test gaps
  - **NPHI**: neutron porosity (hydrogen content, mostly water in pores); present in 17 of the 20 test gaps
  - **PEF**: photoelectric factor (tracks mineralogy, e.g. sandstone vs limestone); present in 19 of the 20 test gaps

**The rows.**

| Row | Sees | What it does |
|---|---|---|
| Straight line | density flanks | Draws a straight line from the last measured density above to the first below. The baseline to beat. |
| Gardner (fixed) | sonic (DTC) inside the gap | A textbook rock-physics formula, density = 0.31 × velocity^0.25, using sonic speed. Uses no density at all. |
| Gardner + flank residual | sonic + density flanks | Gardner, shifted by how far it is off on the flanks (that offset is interpolated across the gap). |
| TTM-R3 frozen / tuned | density only | IBM's small (3 M weights) time-series forecaster, pretrained on non-geological series; one depth step is treated as one time step. It forecasts down from the upper flank and up from the lower flank and blends the two. *Tuned*: the brief's run, which retrained its output layers on our training wells but stopped after 75 updates (under 5 s). |
| TSPulse frozen / tuned | density only | IBM's small (1 M weights) gap-filling model: it sees a 512-sample window with the gap blanked out and reconstructs it. *Tuned*: the brief's run, which retrained only its final 200-weight output layer for 125 updates. |
| TTM-R3 / TSPulse *head*, *decoder* or *full fine-tune* | density only | The stronger fine-tune added after the brief's run: up to the whole network is trained, on 8× more practice gaps (13,824), for minutes rather than seconds. The part trained and the learning rate are picked on the B wells; see the sweep table. |
| Timestral-5M + other logs | density flanks + all other logs + depth | Mistral's forecaster with the red team's adapter, frozen. The other logs across the flanks *and* the gap are side information. |
| Timestral-5M target-only | density only | The same model and adapter, with the other logs removed: the control that shows what the other logs are worth. |

*Frozen* means the published weights, untouched. No method sees the hidden density.

**Wells and why they are split.** The 98 training wells are split once, by a fixed hash of the well name:
A (54 wells) to fine-tune the models, B (12) to choose checkpoints and the "selected" row, C (20) to fit the trust
rule, and D (12) to calibrate the uncertainty bands and the trust rule's probabilities. Each well is used for one
job only, so no choice is checked on the wells it was made on.

**How it is scored.**
- **Gap error (MAE):** the mean of |filled − measured| over the hidden samples, in g/cm³. Each test well has one
  gap, so per-well and per-gap error are the same; the headline is the mean over the 20 wells.
- **Paired bootstrap:** to compare two methods we take their per-well differences and resample the 20 wells 2,000
  times. The 95 % interval of the mean difference is reported; if it excludes 0, the difference holds up at this
  sample size. Negative means the first method is better.
- **Uncertainty bands:** a 90 % band should contain 90 % of the hidden values. *Coverage* is how often it does;
  *width* is how wide it is. The Granite bands are calibrated on D; Timestral's come from its own quantiles.
- **Trust rule (decision):** for each gap, a probability that the fill is within 0.05 g/cm³ (gap MAE ≤ 0.05). It is
  computed only from things known without the truth: how much the two flanks disagree, how variable the density
  is on the flanks (for TTM: its band width), the fill's range, flank lengths, and disagreement with the
  sonic-based Gardner estimate. A gap is **accepted** when
  p ≥ 0.80 and otherwise sent to **review** by a person. A *false accept* is an accepted gap that misses the
  tolerance. The **Brier score** is the mean squared error of these probabilities: lower is better, and a
  constant 50 % scores 0.25. The rule is fitted on C, recalibrated on D, and applied to the row selected on B
  (TSPulse tuned). *Decima-small* is a small language model given the same facts as text and asked the
  same yes/no question.

**How to read the main table.** *Support* is gaps filled out of 20. *Gaps ≤ 0.05* counts gaps within tolerance.
*Wells better than line* counts wells where the row beats the straight line. *B mean well MAE* is the error on the
selection wells (higher than test, because B gaps are harder). The pooled MAE in brackets is the benchmark
runner's own score, as a cross-check.

## 30-second result

We hid one real 50 m stretch of bulk density in each of the 20 FORCE test wells. The best fill came from Mistral's Timestral-5M, frozen, reading the other logs across the gap: 0.033 g/cm³ mean error, 45 % better than a straight line (0.061), better in 18 of 20 wells, 16 of 20 gaps within 0.05. From the density log alone the models tie: Timestral 0.047, IBM TTM-R3 0.048, TSPulse 0.049 (TTM − Timestral +0.0007, 95 % CI -0.0023 to +0.0045). So the gain is the other logs (-0.0135, 95 % CI -0.0221 to -0.0062), not the model. The brief's laptop fine-tune of the IBM heads (early-stopped after 75–125 updates, 5 s or less on a laptop GPU) changed TTM error by -0.0006 (95 % CI -0.0017 to +0.0004) and TSPulse by -0.0001. A stronger fine-tune afterwards (broader unfreezing, 8× more training gaps, longer schedule, chosen on B only): TTM-R3 head fine-tune (1.08 M weights trained, 1800 updates, 3.5 min) 0.047, -0.0010 vs frozen (95 % CI -0.0046 to +0.0028); TSPulse full fine-tune (1.07 M weights trained, 2900 updates, 2.7 min) 0.046, -0.0035 vs frozen (95 % CI -0.0070 to -0.0008). Gardner from sonic alone is worse than the line (0.100); corrected by the flank residual it reaches 0.056. Timestral's nominal 90 % band covers only 46 % of the hidden samples; the D-calibrated Granite bands cover 96 %. For each gap a small calibrated rule (Numeric logistic) says whether the TSPulse tuned fill is within 0.05 g/cm³: at p ≥ 0.80 it accepted 7 of 20 gaps with 0 false accepts and sent 13 to review; Brier 0.158 vs 0.261 for the base rate. A small verifier model (Decima-small) reading the same facts as text accepted 0 gaps (p_pass 0.47–0.53, Brier 0.239): its probabilities hardly vary and on the 236 development gaps its ranking is near chance, so it never reaches the 0.80 bar. Refitted on each model's own fills, the same rule ranks gaps with test AUC 0.75–0.91; adding two gamma-ray features (GR is measured inside the gap) lowers its test Brier for 10 of 10 models.

## Tables

| Row | Test mean per-well MAE | Pooled MAE (station check) | RMSE | Median per-well Δ vs line | Wells better than line | Support | Gaps ≤ 0.05 | Cover 90 % (width) | Cover 50 % | B mean well MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| TTM-R3 frozen | 0.0481 | 0.0481 (0.0481) | 0.0731 | -0.0039 | 15 | 20/20 | 13/20 | 0.96 (0.304) | 0.58 | 0.0799 |
| TTM-R3 tuned | 0.0475 | 0.0475 (0.0475) | 0.0726 | -0.0057 | 16 | 20/20 | 13/20 | 0.96 (0.296) | 0.58 | 0.0804 |
| TSPulse frozen | 0.0495 | 0.0495 (0.0495) | 0.0753 | -0.0030 | 13 | 20/20 | 13/20 | 0.96 (0.359) | 0.65 | 0.0789 |
| TSPulse tuned ★ | 0.0494 | 0.0494 (0.0494) | 0.0749 | -0.0030 | 15 | 20/20 | 13/20 | 0.96 (0.356) | 0.65 | 0.0789 |
| TTM-R3 head fine-tune | 0.0471 | 0.0471 (0.0471) | 0.0717 | -0.0030 | 15 | 20/20 | 13/20 | 0.96 (0.269) | 0.56 | 0.0774 |
| TSPulse full fine-tune | 0.0460 | 0.0460 (0.0460) | 0.0718 | -0.0031 | 18 | 20/20 | 13/20 | 0.96 (0.344) | 0.66 | 0.0748 |
| Linear interpolation | 0.0611 | 0.0611 (0.0611) | 0.0913 | 0.0000 | – | 20/20 | 8/20 | – | – | – |
| Gardner (fixed) | 0.0995 | 0.0995 (0.0995) | 0.1170 | 0.0207 | 3 | 20/20 | 1/20 | 0.99 (0.507) | 0.51 | – |
| Gardner + flank residual | 0.0563 | 0.0563 (0.0563) | 0.0861 | -0.0042 | 15 | 20/20 | 10/20 | – | – | – |
| Timestral-5M + other logs | 0.0333 | 0.0333 (0.0333) | 0.0541 | -0.0195 | 18 | 20/20 | 16/20 | 0.46 (0.055) | 0.24 | – |
| Timestral-5M target-only | 0.0469 | 0.0469 (0.0469) | 0.0721 | -0.0029 | 16 | 20/20 | 14/20 | 0.24 (0.039) | 0.14 | – |

★ selected before the test on B (lowest B mean well-level MAE). Test wells hold one gap each, so per-well MAE = gap MAE. Coverage is well-weighted over measured gap samples; Granite bands are D-calibrated, Timestral bands are nominal (Gaussian from its own P10/P90, as in red's adapter).

Timestral-5M rows: frozen weights `mistralai/Timestral-5M@bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f`, red's adapter unchanged (256-sample flanks, 32-step rollout, inverse-distance blend), CPU. *+ other logs* gets every log the test offers across the gap as covariates; *target-only* gets empty covariate dictionaries.

| Row | Fit s | Inference s (20 gaps, CPU) | Raw P10–P90 coverage on test (well-weighted, width) | Gaps with each covariate |
|---|---|---|---|---|
| Timestral-5M + other logs | 2.0 | 66.9 | 0.39 (0.043) | CALI 20, DTC 20, DTS 13, GR 20, NPHI 17, PEF 19, RDEP 20, RMED 20 |
| Timestral-5M target-only | 0.1 | 30.8 | 0.21 (0.030) | none (control) |

**Paired bootstrap** (mean of per-well MAE differences a − b; 2,000 resamples of wells; negative = a better)

| Set | a | b | Mean Δ | 95 % CI | P(Δ<0) | Wells a better / worse |
|---|---|---|---|---|---|---|
| test | TTM-R3 tuned | TTM-R3 frozen | -0.0006 | [-0.0017, +0.0004] | 0.87 | 13 / 7 |
| test | TSPulse tuned | TSPulse frozen | -0.0001 | [-0.0003, +0.0000] | 0.97 | 15 / 5 |
| test | TTM-R3 frozen | Linear interpolation | -0.0130 | [-0.0269, -0.0033] | 1.00 | 15 / 5 |
| test | TTM-R3 tuned | Linear interpolation | -0.0136 | [-0.0274, -0.0036] | 1.00 | 16 / 4 |
| test | TSPulse frozen | Linear interpolation | -0.0116 | [-0.0260, -0.0005] | 0.98 | 13 / 7 |
| test | TSPulse tuned | Linear interpolation | -0.0118 | [-0.0273, -0.0004] | 0.98 | 15 / 5 |
| test | Gardner + flank residual | Linear interpolation | -0.0048 | [-0.0173, +0.0084] | 0.77 | 15 / 5 |
| test | TSPulse tuned | Gardner + flank residual | -0.0069 | [-0.0210, +0.0055] | 0.84 | 12 / 8 |
| test | TTM-R3 tuned | TSPulse tuned | -0.0018 | [-0.0071, +0.0032] | 0.77 | 11 / 9 |
| test | Timestral-5M + other logs | Timestral-5M target-only | -0.0135 | [-0.0221, -0.0062] | 1.00 | 16 / 4 |
| test | Timestral-5M + other logs | Linear interpolation | -0.0278 | [-0.0432, -0.0148] | 1.00 | 18 / 2 |
| test | Timestral-5M target-only | Linear interpolation | -0.0143 | [-0.0292, -0.0029] | 0.99 | 16 / 4 |
| test | TTM-R3 tuned | Timestral-5M target-only | +0.0007 | [-0.0023, +0.0045] | 0.36 | 9 / 11 |
| test | TSPulse tuned | Timestral-5M target-only | +0.0025 | [-0.0018, +0.0071] | 0.12 | 7 / 13 |
| test | TTM-R3 tuned | Timestral-5M + other logs | +0.0142 | [+0.0063, +0.0231] | 0.00 | 4 / 16 |
| test | TSPulse tuned | Timestral-5M + other logs | +0.0160 | [+0.0077, +0.0257] | 0.00 | 3 / 17 |
| test | Gardner + flank residual | Timestral-5M + other logs | +0.0230 | [+0.0105, +0.0399] | 0.00 | 4 / 16 |
| test | TTM-R3 head fine-tune | TTM-R3 frozen | -0.0010 | [-0.0046, +0.0028] | 0.70 | 13 / 7 |
| test | TTM-R3 head fine-tune | TTM-R3 tuned | -0.0004 | [-0.0036, +0.0032] | 0.58 | 12 / 8 |
| test | TTM-R3 head fine-tune | Linear interpolation | -0.0140 | [-0.0293, -0.0034] | 1.00 | 15 / 5 |
| test | TTM-R3 head fine-tune | Timestral-5M + other logs | +0.0138 | [+0.0066, +0.0213] | 0.00 | 5 / 15 |
| test | TTM-R3 head fine-tune | Timestral-5M target-only | +0.0003 | [-0.0033, +0.0039] | 0.42 | 8 / 12 |
| test | TSPulse full fine-tune | TSPulse frozen | -0.0035 | [-0.0070, -0.0008] | 1.00 | 13 / 7 |
| test | TSPulse full fine-tune | TSPulse tuned | -0.0034 | [-0.0070, -0.0006] | 1.00 | 13 / 7 |
| test | TSPulse full fine-tune | Linear interpolation | -0.0152 | [-0.0298, -0.0044] | 1.00 | 18 / 2 |
| test | TSPulse full fine-tune | Timestral-5M + other logs | +0.0126 | [+0.0050, +0.0219] | 0.00 | 5 / 15 |
| test | TSPulse full fine-tune | Timestral-5M target-only | -0.0009 | [-0.0040, +0.0022] | 0.70 | 8 / 12 |
| B (selection wells) | TTM-R3 tuned | TTM-R3 frozen | +0.0006 | [-0.0001, +0.0014] | 0.07 | 6 / 6 |
| B (selection wells) | TSPulse tuned | TSPulse frozen | -0.0001 | [-0.0002, +0.0001] | 0.79 | 8 / 4 |
| B (selection wells) | TTM-R3 head fine-tune | TTM-R3 frozen | -0.0024 | [-0.0045, -0.0004] | 0.99 | 8 / 4 |
| B (selection wells) | TTM-R3 head fine-tune | TTM-R3 tuned | -0.0030 | [-0.0051, -0.0009] | 1.00 | 9 / 3 |
| B (selection wells) | TSPulse full fine-tune | TSPulse frozen | -0.0041 | [-0.0066, -0.0019] | 1.00 | 10 / 2 |
| B (selection wells) | TSPulse full fine-tune | TSPulse tuned | -0.0041 | [-0.0066, -0.0017] | 1.00 | 9 / 3 |

**Decisions** on the selected reconstruction (accept when p_pass ≥ 0.80; label = gap MAE ≤ 0.05)

| Method | Test Brier | Accepted | Review | False accepts | Risk @25 % cov. | Risk @50 % cov. | Reliability bins (n: mean p → pass rate) | D Brier | C Brier | D accepted (false) | Ranking AUC C / D / test |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Prevalence | 0.261 | 0 | 20 | 0 | 0.35 | 0.35 | 0: –→–; 20: 0.47→0.65; 0: –→– | 0.244 | 0.250 | 0/88 (0) | 0.50 / 0.50 / 0.50 |
| Width-only logistic | 0.196 | 2 | 18 | 0 | 0.00 | 0.10 | 5: 0.13→0.40; 7: 0.55→0.43; 8: 0.75→1.00 | 0.154 | 0.180 | 4/88 (1) | 0.81 / 0.85 / 0.81 |
| Numeric logistic ◆ | 0.158 | 7 | 13 | 0 | 0.00 | 0.10 | 5: 0.18→0.20; 4: 0.52→0.75; 11: 0.81→0.82 | 0.150 | 0.138 | 14/88 (2) | 0.89 / 0.86 / 0.86 |
| Numeric + Gardner | 0.156 | 7 | 13 | 0 | 0.00 | 0.10 | 5: 0.17→0.20; 5: 0.55→0.60; 10: 0.83→0.90 | 0.152 | 0.138 | 15/88 (2) | 0.89 / 0.85 / 0.87 |
| Numeric (recal. D) | 0.161 | 7 | 13 | 0 | 0.00 | 0.10 | 5: 0.17→0.20; 6: 0.54→0.67; 9: 0.82→0.89 | 0.150 | 0.139 | 11/88 (1) | 0.89 / 0.86 / 0.86 |
| Numeric + Gardner (recal. D) | 0.159 | 7 | 13 | 0 | 0.00 | 0.10 | 5: 0.17→0.20; 7: 0.55→0.57; 8: 0.82→1.00 | 0.152 | 0.141 | 10/88 (1) | 0.89 / 0.85 / 0.87 |
| Decima-small raw | 0.239 | 0 | 20 | 0 | 0.20 | 0.10 | 0: –→–; 20: 0.51→0.65; 0: –→– | 0.243 | 0.256 | 0/88 (0) | 0.48 / 0.62 / 0.82 |
| Decima-small (recal. D) | 0.280 | 0 | 20 | 0 | 0.20 | 0.10 | 0: –→–; 20: 0.39→0.65; 0: –→– | 0.232 | 0.272 | 0/88 (0) | 0.48 / 0.62 / 0.82 |

◆ headline rule: lowest D Brier among fitted methods (dev data only). Pass rates: C 0.49, D 0.40, test 0.65. *Ranking AUC*: the chance that a passing gap gets a higher p_pass than a failing one (0.5 = chance, 1 = perfect). The rules are fitted on C, so AUC on D and test is held out. The recalibration's two numbers (intercept, slope) are fitted on D, so D's accepted/false counts for the recalibrated rules are slightly optimistic. Test has only 20 gaps.

**A trust rule for every model** ([decide_any.py](decide_any.py); the same rule recipe, fitted on C and recalibrated on D for each model's own fills; numeric rule recalibrated on D, accept at p ≥ 0.80). Pass rate = share of that model's gaps within 0.05; a better model leaves fewer gaps to catch.

| Model | Test mean well MAE | Test pass rate | Test accepted / review / false | Test Brier (base rate) | D pass rate | D accepted (false) | Ranking AUC C / D / test | **+ GR features:** test accepted (false) · Brier · AUC D / test | D accepted (false) | Own band used |
|---|---|---|---|---|---|---|---|---|---|---|
| Timestral-5M + other logs | 0.0333 | 0.80 | 10 / 10 / 1 | 0.159 (0.169) | 0.59 | 31/88 (4) | 0.87 / 0.84 / 0.75 | 10 (0) · 0.118 · 0.86 / 0.88 | 25/88 (2) | P10-P90 |
| TSPulse full fine-tune | 0.0460 | 0.65 | 6 / 14 / 0 | 0.179 (0.247) | 0.43 | 10/88 (1) | 0.81 / 0.85 / 0.84 | 6 (0) · 0.157 · 0.88 / 0.88 | 18/88 (2) | P05-P95 |
| Timestral-5M target-only | 0.0469 | 0.70 | 8 / 12 / 0 | 0.177 (0.253) | 0.36 | 11/88 (2) | 0.90 / 0.89 / 0.86 | 5 (0) · 0.163 · 0.88 / 0.87 | 16/88 (3) | P10-P90 |
| TTM-R3 head fine-tune | 0.0471 | 0.65 | 8 / 12 / 0 | 0.120 (0.269) | 0.40 | 15/88 (3) | 0.89 / 0.86 / 0.91 | 8 (0) · 0.090 · 0.89 / 0.96 | 17/88 (3) | P05-P95 |
| TTM-R3 tuned | 0.0475 | 0.65 | 5 / 15 / 0 | 0.136 (0.255) | 0.40 | 11/88 (2) | 0.85 / 0.86 / 0.91 | 6 (0) · 0.114 · 0.88 / 0.95 | 16/88 (3) | P05-P95 |
| TTM-R3 frozen | 0.0481 | 0.65 | 5 / 15 / 0 | 0.147 (0.261) | 0.40 | 11/88 (2) | 0.86 / 0.85 / 0.89 | 6 (0) · 0.125 · 0.86 / 0.92 | 13/88 (2) | P05-P95 |
| TSPulse tuned | 0.0494 | 0.65 | 4 / 16 / 0 | 0.206 (0.261) | 0.40 | 4/88 (0) | 0.84 / 0.85 / 0.80 | 6 (0) · 0.185 · 0.86 / 0.85 | 12/88 (2) | P05-P95 |
| TSPulse frozen | 0.0495 | 0.65 | 4 / 16 / 0 | 0.209 (0.264) | 0.40 | 6/88 (0) | 0.85 / 0.85 / 0.80 | 6 (0) · 0.188 · 0.85 / 0.85 | 12/88 (2) | P05-P95 |
| Gardner + flank residual | 0.0563 | 0.50 | 6 / 14 / 1 | 0.204 (0.250) | 0.53 | 9/88 (3) | 0.92 / 0.75 / 0.81 | 4 (0) · 0.183 · 0.75 / 0.85 | 9/88 (1) | none |
| Linear interpolation | 0.0611 | 0.40 | 0 / 20 / 0 | 0.142 (0.243) | 0.36 | 2/88 (0) | 0.87 / 0.81 / 0.90 | 1 (0) · 0.121 · 0.85 / 0.91 | 7/88 (0) | none |

*+ GR features*: the same rule with two more inputs read from the gamma-ray log, which is measured inside the gap: how far GR inside the gap moves from GR on the flanks, and how much GR varies inside the gap (both log-scaled). A density-only fill cannot see a lithology change inside the gap; GR can.

**Per-gap routing (negative result).** For each gap, take the fill of the model whose own rule is most confident (no truth used). Mean per-well MAE: D routed 0.0605 vs best single model 0.0539 (Timestral-5M + other logs); test routed 0.0378 vs best single model 0.0333 (Timestral-5M + other logs). Routing is worse on both. Each rule answers *is this gap easy for me?*, not *am I better than the others here?*; the probabilities are not comparable across models.

**Training and resources** (head-only fine-tune on A wells, laptop GPU)

| Arm | Trainable / total | Examples (wells) | Microbatch × accum | s/update | Updates | Minutes | Stop | Peak VRAM MB | RSS MB | B step 0 → tuned (update) | Overfit/reload exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ttm | 1,080,478 / 3,157,904 | 3176 (54) | 8 × 1 | 0.061 | 75 | 0.08 | no improvement in 3 evaluations | 126 | 1988 | 0.07986 → 0.08042 (50) | True |
| tspulse | 200 / 1,068,958 | 1728 (54) | 8 × 1 | 0.020 | 125 | 0.05 | no improvement in 3 evaluations | 50 | 2029 | 0.07891 → 0.07885 (50) | True |

**Stronger fine-tune sweep** (extension after the brief's run; A wells, up to 256 gaps per well, batch 32, warm-up + cosine, eval on B every 100 updates, patience 10, ≤ 4,000 updates or 12 min; the best B evaluation wins; TTM trained on its blended two-sided output)

| Model | Unfrozen | Learning rate | Trained weights | Updates | Minutes | Stop | B: pretrained → best (update) | Selected |
|---|---|---|---|---|---|---|---|---|
| TTM-R3 | head | 0.001 | 1,080,478 | 1800 | 3.5 | no improvement in 10 evaluations | 0.07986 → 0.07741 (800) | yes |
| TTM-R3 | full | 0.0001 | 3,157,904 | 2000 | 8.1 | no improvement in 10 evaluations | 0.07986 → 0.07744 (1000) |  |
| TTM-R3 | full | 3e-05 | 3,157,904 | 2901 | 12.0 | time box 12.0 min | 0.07986 → 0.07755 (2300) |  |
| TSPulse | decoder | 0.0001 | 270,900 | 1500 | 0.7 | no improvement in 10 evaluations | 0.07891 → 0.07615 (500) |  |
| TSPulse | full | 0.0001 | 1,068,958 | 2900 | 2.7 | no improvement in 10 evaluations | 0.07891 → 0.07477 (1900) | yes |
| TSPulse | full | 3e-05 | 1,068,958 | 3800 | 3.5 | no improvement in 10 evaluations | 0.07891 → 0.07548 (2800) |  |

| Row | k50 | k90 | Raw TTM 80 % coverage on D (well-weighted) | Fit s | Warm inference s (20 gaps) |
|---|---|---|---|---|---|
| TTM-R3 frozen | 0.231 | 0.796 | 0.775 | 10.28 | 0.067 |
| TTM-R3 tuned | 0.216 | 0.736 | 0.800 | 2.03 | 0.062 |
| TSPulse frozen | 0.047 | 0.180 | – | 1.78 | 0.035 |
| TSPulse tuned | 0.047 | 0.178 | – | 1.80 | 0.058 |
| TTM-R3 head fine-tune | 0.265 | 0.854 | – | – | 0.500 |
| TSPulse full fine-tune | 0.042 | 0.172 | – | – | 0.032 |
