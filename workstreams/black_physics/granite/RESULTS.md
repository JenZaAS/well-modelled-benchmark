# Granite 50 m density gap: results (aggregate only)

Written by `python -m workstreams.black_physics.granite report` from the frozen artifacts; no well names, depths or curve values. Test `rhob_gap50`: one hidden 50 m RHOB gap in each of the 20 FORCE 2020 test wells (6,600 samples). Code hash `a00483f7fc540fff`, config `00bafc81acb0bec0`, split `e418285e7420ef26`, data `d80d06e1233764cb`; predictions sha256 `1914285b2893fcf3`, decision sidecar `65e6a89bf4d9d88c` (local files).

## 30-second result

We hid one real 50 m stretch of bulk density in each of the 20 FORCE test wells. The best fill came from Mistral's Timestral-5M, frozen, reading the other logs across the gap: 0.033 g/cm³ mean error, 45 % better than a straight line (0.061), better in 18 of 20 wells, 16 of 20 gaps within 0.05. From the density log alone the models tie: Timestral 0.047, IBM TTM-R3 0.048, TSPulse 0.049 (TTM − Timestral +0.0007, 95 % CI -0.0023 to +0.0045). So the gain is the other logs (-0.0135, 95 % CI -0.0221 to -0.0062), not the model. Laptop fine-tuning of the IBM heads (early-stopped after 75–125 updates, 5 s or less on a laptop GPU) changed TTM error by -0.0006 (95 % CI -0.0017 to +0.0004) and TSPulse by -0.0001. Gardner from sonic alone is worse than the line (0.100); corrected by the flank residual it reaches 0.056. Timestral's nominal 90 % band covers only 46 % of the hidden samples; the D-calibrated Granite bands cover 96 %. For each gap a small calibrated rule (Numeric logistic) says whether the TSPulse tuned fill is within 0.05 g/cm³: at p ≥ 0.80 it accepted 7 of 20 gaps with 0 false accepts and sent 13 to review; Brier 0.158 vs 0.261 for the base rate. A small verifier model (Decima-small) reading the same facts as text accepted 0 gaps (p_pass 0.47–0.53, Brier 0.239): it barely separates good from bad gaps.

## Tables

| Row | Test mean per-well MAE | Pooled MAE (station check) | RMSE | Median per-well Δ vs line | Wells better than line | Support | Gaps ≤ 0.05 | Cover 90 % (width) | Cover 50 % | B mean well MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| TTM-R3 frozen | 0.0481 | 0.0481 (0.0481) | 0.0731 | -0.0039 | 15 | 20/20 | 13/20 | 0.96 (0.304) | 0.58 | 0.0799 |
| TTM-R3 tuned | 0.0475 | 0.0475 (0.0475) | 0.0726 | -0.0057 | 16 | 20/20 | 13/20 | 0.96 (0.296) | 0.58 | 0.0804 |
| TSPulse frozen | 0.0495 | 0.0495 (0.0495) | 0.0753 | -0.0030 | 13 | 20/20 | 13/20 | 0.96 (0.359) | 0.65 | 0.0789 |
| TSPulse tuned ★ | 0.0494 | 0.0494 (0.0494) | 0.0749 | -0.0030 | 15 | 20/20 | 13/20 | 0.96 (0.356) | 0.65 | 0.0789 |
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
| B (selection wells) | TTM-R3 tuned | TTM-R3 frozen | +0.0006 | [-0.0001, +0.0014] | 0.07 | 6 / 6 |
| B (selection wells) | TSPulse tuned | TSPulse frozen | -0.0001 | [-0.0002, +0.0001] | 0.79 | 8 / 4 |

**Decisions** on the selected reconstruction (accept when p_pass ≥ 0.80; label = gap MAE ≤ 0.05)

| Method | Test Brier | Accepted | Review | False accepts | Risk @25 % cov. | Risk @50 % cov. | Reliability bins (n: mean p → pass rate) | D Brier | C Brier |
|---|---|---|---|---|---|---|---|---|---|
| Prevalence | 0.261 | 0 | 20 | 0 | 0.35 | 0.35 | 0: –→–; 20: 0.47→0.65; 0: –→– | 0.244 | 0.250 |
| Width-only logistic | 0.196 | 2 | 18 | 0 | 0.00 | 0.10 | 5: 0.13→0.40; 7: 0.55→0.43; 8: 0.75→1.00 | 0.154 | 0.180 |
| Numeric logistic ◆ | 0.158 | 7 | 13 | 0 | 0.00 | 0.10 | 5: 0.18→0.20; 4: 0.52→0.75; 11: 0.81→0.82 | 0.150 | 0.138 |
| Numeric + Gardner | 0.156 | 7 | 13 | 0 | 0.00 | 0.10 | 5: 0.17→0.20; 5: 0.55→0.60; 10: 0.83→0.90 | 0.152 | 0.138 |
| Numeric (recal. D) | 0.161 | 7 | 13 | 0 | 0.00 | 0.10 | 5: 0.17→0.20; 6: 0.54→0.67; 9: 0.82→0.89 | 0.150 | 0.139 |
| Numeric + Gardner (recal. D) | 0.159 | 7 | 13 | 0 | 0.00 | 0.10 | 5: 0.17→0.20; 7: 0.55→0.57; 8: 0.82→1.00 | 0.152 | 0.141 |
| Decima-small raw | 0.239 | 0 | 20 | 0 | 0.20 | 0.10 | 0: –→–; 20: 0.51→0.65; 0: –→– | 0.243 | 0.256 |
| Decima-small (recal. D) | 0.280 | 0 | 20 | 0 | 0.20 | 0.10 | 0: –→–; 20: 0.39→0.65; 0: –→– | 0.232 | 0.272 |

◆ headline rule: lowest D Brier among fitted methods (dev data only). Pass rates: C 0.49, D 0.40, test 0.65.

**Training and resources** (head-only fine-tune on A wells, laptop GPU)

| Arm | Trainable / total | Examples (wells) | Microbatch × accum | s/update | Updates | Minutes | Stop | Peak VRAM MB | RSS MB | B step 0 → tuned (update) | Overfit/reload exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ttm | 1,080,478 / 3,157,904 | 3176 (54) | 8 × 1 | 0.061 | 75 | 0.08 | no improvement in 3 evaluations | 126 | 1988 | 0.07986 → 0.08042 (50) | True |
| tspulse | 200 / 1,068,958 | 1728 (54) | 8 × 1 | 0.020 | 125 | 0.05 | no improvement in 3 evaluations | 50 | 2029 | 0.07891 → 0.07885 (50) | True |

| Row | k50 | k90 | Raw TTM 80 % coverage on D (well-weighted) | Fit s | Warm inference s (20 gaps) |
|---|---|---|---|---|---|
| TTM-R3 frozen | 0.231 | 0.796 | 0.775 | 10.28 | 0.067 |
| TTM-R3 tuned | 0.216 | 0.736 | 0.800 | 2.03 | 0.062 |
| TSPulse frozen | 0.047 | 0.180 | – | 1.78 | 0.035 |
| TSPulse tuned | 0.047 | 0.178 | – | 1.80 | 0.058 |
