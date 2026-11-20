# Ordinal density scoreboard

Official `rhob_gap50` test (20 wells, one 50 m gap each, every hidden depth) unless stated. Heads fitted on C (20 wells), temperature on D (12 wells). RPS: ranked probability score over the 15 fixed bins, lower is better; Δ vs pooled errors is a paired well-bootstrap 95 % interval (wells where the head is lower).

## TSPulse fine-tune (density only (RHOB flanks))

| Head | Test RPS | Δ RPS vs pooled errors | Cover 80 % | Width 80 % | Cover 90 % | Width 90 % | Brier P(<2.40) | MAE median | Dev (D) RPS |
|---|---|---|---|---|---|---|---|---|---|
| Empirical errors, pooled (raw) | 0.0266 | +0.0000 [+0.0000, +0.0000] (0/20) | 0.87 | 0.201 | 0.94 | 0.291 | 0.046 | 0.0467 | 0.0395 |
| Empirical errors, pooled (calibrated) | 0.0272 | +0.0006 [-0.0000, +0.0011] (4/20) | 0.90 | 0.242 | 0.97 | 0.360 | 0.048 | 0.0465 | 0.0394 |
| Empirical errors by flank variability (raw) | 0.0259 | -0.0007 [-0.0014, +0.0000] (13/20) | 0.89 | 0.197 | 0.95 | 0.281 | 0.045 | 0.0472 | 0.0391 |
| Empirical errors by flank variability (calibrated) | 0.0261 | -0.0004 [-0.0014, +0.0005] (12/20) | 0.92 | 0.227 | 0.97 | 0.320 | 0.045 | 0.0470 | 0.0391 |
| Numeric head (raw) | 0.0287 | +0.0021 [-0.0000, +0.0041] (4/20) | 0.88 | 0.233 | 0.94 | 0.302 | 0.053 | 0.0500 | 0.0408 |
| Numeric head (calibrated) | 0.0302 | +0.0036 [+0.0010, +0.0059] (3/20) | 0.94 | 0.291 | 0.98 | 0.374 | 0.057 | 0.0499 | 0.0411 |
| Numeric head + sonic check (raw) ◆ | 0.0276 | +0.0010 [-0.0008, +0.0031] (9/20) | 0.89 | 0.227 | 0.95 | 0.290 | 0.045 | 0.0487 | 0.0383 |
| Numeric head + sonic check (calibrated) | 0.0290 | +0.0024 [+0.0000, +0.0048] (5/20) | 0.94 | 0.277 | 0.97 | 0.351 | 0.049 | 0.0490 | 0.0387 |
| Native band of the model (90 %), reference | – | – | – | – | 0.96 | 0.344 | – | – | – |

◆ heatmap head, chosen by lowest D RPS. Base point MAE 0.0460 g/cm³; bin-resolution floor (MAE of each hidden value's own bin representative) 0.0263; tail depths low/high 0/3; 20 wells, 20 gaps, 6600 depths, 0 abstentions; 0 depths without flank variability (pooled fallback), 0 with one flank only. Head runtime on CPU: Empirical errors, pooled 1.0 µs per depth, Empirical errors by flank variability 2.2 µs per depth, Numeric head 0.7 µs per depth, Numeric head + sonic check 0.8 µs per depth.

Sonic check (numeric head + sonic − numeric head, raw; negative favours the sonic feature): test -0.0011 [-0.0025, +0.0005] (16/20), D -0.0024 [-0.0041, -0.0006] (10/12). Depths with a usable DTC (Gardner) value: test 100 %, D 96 %.

## Timestral-5M + other logs (RHOB flanks + other logs, also inside the gap)

| Head | Test RPS | Δ RPS vs pooled errors | Cover 80 % | Width 80 % | Cover 90 % | Width 90 % | Brier P(<2.40) | MAE median | Dev (D) RPS |
|---|---|---|---|---|---|---|---|---|---|
| Empirical errors, pooled (raw) | 0.0194 | +0.0000 [+0.0000, +0.0000] (0/20) | 0.93 | 0.176 | 0.97 | 0.252 | 0.034 | 0.0348 | 0.0298 |
| Empirical errors, pooled (calibrated) | 0.0196 | +0.0002 [+0.0001, +0.0004] (5/20) | 0.94 | 0.192 | 0.97 | 0.281 | 0.034 | 0.0346 | 0.0298 |
| Empirical errors by flank variability (raw) ◆ | 0.0189 | -0.0004 [-0.0008, -0.0000] (14/20) | 0.94 | 0.175 | 0.97 | 0.245 | 0.032 | 0.0351 | 0.0297 |
| Empirical errors by flank variability (calibrated) | 0.0190 | -0.0004 [-0.0008, +0.0001] (12/20) | 0.94 | 0.182 | 0.98 | 0.259 | 0.032 | 0.0350 | 0.0297 |
| Numeric head (raw) | 0.0214 | +0.0020 [+0.0004, +0.0035] (4/20) | 0.95 | 0.215 | 0.97 | 0.276 | 0.035 | 0.0357 | 0.0311 |
| Numeric head (calibrated) | 0.0227 | +0.0033 [+0.0017, +0.0047] (3/20) | 0.96 | 0.248 | 0.98 | 0.315 | 0.037 | 0.0357 | 0.0315 |
| Numeric head + sonic check (raw) | 0.0214 | +0.0020 [+0.0007, +0.0033] (3/20) | 0.95 | 0.216 | 0.97 | 0.273 | 0.036 | 0.0363 | 0.0342 |
| Numeric head + sonic check (calibrated) | 0.0230 | +0.0036 [+0.0022, +0.0050] (3/20) | 0.96 | 0.251 | 0.98 | 0.316 | 0.038 | 0.0368 | 0.0338 |
| Native band of the model (80 %), reference | – | – | 0.39 | 0.043 | – | – | – | – | – |

◆ heatmap head, chosen by lowest D RPS. Base point MAE 0.0333 g/cm³; bin-resolution floor (MAE of each hidden value's own bin representative) 0.0263; tail depths low/high 0/3; 20 wells, 20 gaps, 6600 depths, 0 abstentions; 0 depths without flank variability (pooled fallback), 0 with one flank only. Head runtime on CPU: Empirical errors, pooled 1.1 µs per depth, Empirical errors by flank variability 2.1 µs per depth, Numeric head 0.6 µs per depth, Numeric head + sonic check 0.6 µs per depth.

Sonic check (numeric head + sonic − numeric head, raw; negative favours the sonic feature): test +0.0000 [-0.0006, +0.0008] (12/20), D +0.0031 [-0.0008, +0.0104] (7/12). Depths with a usable DTC (Gardner) value: test 100 %, D 96 %.

## The two heatmap heads against each other

RPS of TSPulse fine-tune (numeric_sonic|raw) − Timestral-5M + other logs (empirical_flank|raw); negative favours TSPulse fine-tune: test +0.0086 [+0.0051, +0.0125] (2/20), D +0.0087 [+0.0055, +0.0116] (1/12) (wells where the first is lower).

## Matched comparison with Decima-small: TSPulse fine-tune

Eight sampled depths per gap (704 D and 160 test states); every head scored on exactly those depths. Decima: 143 ms per state (INT8 ONNX, one CPU thread), temperature 2.13.

| Head | D RPS | Test RPS | Test Δ RPS vs Decima-small raw | Test cover 80 % | Test width 80 % |
|---|---|---|---|---|---|
| Decima-small (pretrained) (raw) | 0.0987 | 0.0889 | +0.0000 [+0.0000, +0.0000] (0/20) | 0.52 | 0.357 |
| Empirical errors, pooled (raw) | 0.0387 | 0.0269 | -0.0620 [-0.0883, -0.0376] (19/20) | 0.84 | 0.201 |
| Empirical errors by flank variability (raw) | 0.0385 | 0.0262 | -0.0628 [-0.0890, -0.0388] (19/20) | 0.88 | 0.196 |
| Numeric head (raw) | 0.0399 | 0.0289 | -0.0601 [-0.0855, -0.0370] (18/20) | 0.85 | 0.233 |
| Numeric head + sonic check (raw) | 0.0376 | 0.0274 | -0.0616 [-0.0877, -0.0376] (18/20) | 0.88 | 0.227 |
| Decima-small (pretrained) (calibrated) | 0.0929 | 0.0843 | -0.0047 [-0.0102, +0.0007] (12/20) | 0.88 | 0.599 |
| Empirical errors, pooled (calibrated) | 0.0387 | 0.0275 | -0.0615 [-0.0875, -0.0374] (19/20) | 0.88 | 0.242 |
| Empirical errors by flank variability (calibrated) | 0.0385 | 0.0264 | -0.0625 [-0.0884, -0.0388] (19/20) | 0.92 | 0.227 |
| Numeric head (calibrated) | 0.0404 | 0.0303 | -0.0587 [-0.0838, -0.0359] (18/20) | 0.92 | 0.292 |
| Numeric head + sonic check (calibrated) | 0.0380 | 0.0288 | -0.0602 [-0.0860, -0.0365] (18/20) | 0.94 | 0.277 |
