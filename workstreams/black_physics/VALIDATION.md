# Black physics baselines: FORCE 2020 validation note

Aggregate-only results for the seven registered physics contenders on the station's FORCE 2020 tasks,
19 November 2026. The note contains no well-level values. The field results favour different contenders for
different targets, as reported below.

## Provenance

| Item | Value |
| --- | --- |
| Model code | branch `black-physics-baselines`, commit `9546143` (model code unchanged since) |
| Task spec | `benchmark/task_specs/force2020.yaml` + `force2020.windows.yaml` (sha256 prefix `b1c63396b28e6d0e`), cache `spec_hash` `355858f80204ff2a`, windows not rerolled |
| Split | official: 98 training wells for `fit`; 10 leaderboard + 10 hidden wells for testing |
| Tasks | 216 = (20 RHOB + 20 DTC + 14 DTS) × 4 scenarios (`missing_curve`, `gap_10`, `gap_50`, `gap_100`) |
| Data | `wmb.paths.get_dataset_path("force2020")` and the canonical loader; checksums OK (`python -m wmb.datasets --check force2020`) |
| Hardware | Intel64 Family 6 Model 165 (12 logical CPUs), Windows, Python 3.13.12 |

Reproduce from the repository root (PowerShell):

```powershell
python -m wmb.paths --check
python -m wmb.datasets --check force2020
python -m pytest -q tests workstreams/black_physics/tests
$physics = 'black_gardner,black_gardner_fixed,black_castagna_dts,black_gc_brine_gr,black_wyllie_nphi,black_density_nphi,black_vpvs_ratio'
foreach ($model in $physics.Split(',')) { python -m benchmark.check_model $model }
python -m benchmark.run --dataset synthetic --model $physics --target all --scenario all
python -m benchmark.make_tasks --dataset force2020
python -m benchmark.run --dataset force2020 --model $physics --target all --scenario all
python -m benchmark.scoreboard
python -m workstreams.black_physics.validate_force --compare template_linear --check-results
```

`validate_force` refits each model on the same 98 wells, predicts the same cached tasks and recomputes the
station's pooled scores with `benchmark.run.score_tasks`. It then adds the extra slices below using
`wmb.scoring.regression_scores`. With `--check-results`, it confirmed that all **28 unique model/target/scenario
combinations** succeeded. Every per-scenario `scores.json` was present and matched the recomputation exactly,
there was no `failure.json`, and the latest run with all seven IDs had no errors. The older copies of
`missing_curve` scores in the legacy layout are not counted.

## Fitted parameters and uncertainty scales (98 training wells)

| ID | Parameters | σ (target units) | σ from |
| --- | --- | ---: | --- |
| `black_gardner` | a = 0.16351, b = 0.33089, fitted on 966,713 pairs with 40 < DTC < 200 µs/ft | 0.1510 g/cm³ | out-of-fold, 98 wells, 967,062 rows |
| `black_gardner_fixed` | a = 0.31, b = 0.25 (published) | 0.1542 g/cm³ | training residuals, 98 wells, 967,062 rows |
| `black_density_nphi` | quartz 2.65, water 1.00 g/cm³, NPHI as φ | 0.3099 g/cm³ | training residuals, 98 wells, 750,294 rows |
| `black_wyllie_nphi` | quartz 55.5, water 189 µs/ft, NPHI as φ | 17.04 µs/ft | training residuals, 98 wells, 734,913 rows |
| `black_castagna_dts` | Vs = 0.8621 Vp − 1.1724 km/s, Vs > 0.3 km/s | 35.89 µs/ft | training residuals, 32 wells, 174,594 rows |
| `black_gc_brine_gr` | GR endpoints 41.03 / 101.92 gAPI (median P10 / P90 over 98 wells) | 24.49 µs/ft | out-of-fold, 32 wells, 174,596 rows |
| `black_vpvs_ratio` | r = DTS/DTC = 1.9456 (median over 32 wells) | 32.20 µs/ft | out-of-fold, 32 wells, 174,596 rows |

Out-of-fold scales use up to five whole-well folds (seed 2026) over training wells with the target; σ is
√(mean over wells of each well's mean squared residual). The mudrock scale is now finite. The earlier
implementation returned NaN on this training set because two valid pairs fall beyond the velocity floor.

## Station scores, fill and timing

**Fill** = finite predictions on masked, measured target depths ÷ all masked, measured target depths.
`n` is the number of scored depths. "Zero-fill" counts test wells in which the model predicts nothing.
MAE and RMSE are pooled over depths as on the station scoreboard. "Per-well MAE" gives each well equal
weight. Coverage and width refer to the station's Gaussian 90% interval. Times are wall-clock seconds from the runner.

| Target | Scenario | Model | n | Fill | Zero-fill wells | MAE | RMSE | Per-well MAE | cover_90 | Width 90 | fit_s | predict_s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RHOB | missing_curve | black_gardner | 231,236 | 0.994 | 0 / 20 | 0.121 | 0.151 | 0.120 | 0.943 | 0.497 | 0.59 | 0.018 |
| RHOB | missing_curve | black_gardner_fixed | 231,236 | 0.994 | 0 / 20 | 0.125 | 0.155 | 0.123 | 0.934 | 0.507 | 0.04 | 0.016 |
| RHOB | missing_curve | black_density_nphi | 199,621 | 0.858 | 0 / 20 | 0.243 | 0.278 | 0.251 | 0.974 | 1.019 | 0.02 | 0.011 |
| RHOB | gap_10 | black_gardner | 1,340 | 1.000 | 0 / 20 | 0.128 | 0.164 | 0.128 | 0.916 | 0.497 | 0.59 | 0.020 |
| RHOB | gap_10 | black_gardner_fixed | 1,340 | 1.000 | 0 / 20 | 0.134 | 0.165 | 0.134 | 0.919 | 0.507 | 0.04 | 0.015 |
| RHOB | gap_10 | black_density_nphi | 1,206 | 0.900 | 2 / 20 | 0.232 | 0.273 | 0.232 | 0.962 | 1.019 | 0.02 | 0.012 |
| RHOB | gap_50 | black_gardner | 6,600 | 1.000 | 0 / 20 | 0.098 | 0.116 | 0.098 | 0.990 | 0.497 | 0.57 | 0.021 |
| RHOB | gap_50 | black_gardner_fixed | 6,600 | 1.000 | 0 / 20 | 0.100 | 0.117 | 0.100 | 0.990 | 0.507 | 0.04 | 0.014 |
| RHOB | gap_50 | black_density_nphi | 5,610 | 0.850 | 3 / 20 | 0.228 | 0.261 | 0.228 | 0.971 | 1.019 | 0.02 | 0.014 |
| RHOB | gap_100 | black_gardner | 13,179 | 1.000 | 0 / 20 | 0.120 | 0.142 | 0.120 | 0.968 | 0.497 | 0.53 | 0.017 |
| RHOB | gap_100 | black_gardner_fixed | 13,179 | 1.000 | 0 / 20 | 0.127 | 0.149 | 0.127 | 0.966 | 0.507 | 0.04 | 0.017 |
| RHOB | gap_100 | black_density_nphi | 11,854 | 0.899 | 2 / 20 | 0.258 | 0.289 | 0.258 | 0.983 | 1.019 | 0.02 | 0.014 |
| DTC | missing_curve | black_wyllie_nphi | 199,590 | 0.785 | 0 / 20 | 13.38 | 18.47 | 12.92 | 0.863 | 56.06 | 0.01 | 0.013 |
| DTC | gap_10 | black_wyllie_nphi | 1,139 | 0.850 | 3 / 20 | 15.66 | 20.53 | 15.66 | 0.835 | 56.06 | 0.02 | 0.013 |
| DTC | gap_50 | black_wyllie_nphi | 5,280 | 0.800 | 4 / 20 | 12.85 | 17.02 | 12.85 | 0.856 | 56.06 | 0.02 | 0.013 |
| DTC | gap_100 | black_wyllie_nphi | 11,203 | 0.850 | 3 / 20 | 10.48 | 14.14 | 10.48 | 0.937 | 56.06 | 0.01 | 0.012 |
| DTS | missing_curve | black_castagna_dts | 116,081 | 1.000 | 0 / 14 | 28.88 | 51.56 | 28.00 | 0.858 | 118.06 | 0.02 | 0.009 |
| DTS | missing_curve | black_gc_brine_gr | 116,082 | 1.000 | 0 / 14 | 17.03 | 26.55 | 17.09 | 0.904 | 80.58 | 0.18 | 0.022 |
| DTS | missing_curve | black_vpvs_ratio | 116,082 | 1.000 | 0 / 14 | 30.67 | 48.93 | 31.24 | 0.800 | 105.92 | 0.08 | 0.011 |
| DTS | gap_10 | black_castagna_dts | 938 | 1.000 | 0 / 14 | 25.72 | 42.23 | 25.72 | 0.851 | 118.06 | 0.02 | 0.009 |
| DTS | gap_10 | black_gc_brine_gr | 938 | 1.000 | 0 / 14 | 14.81 | 24.12 | 14.81 | 0.894 | 80.58 | 0.12 | 0.016 |
| DTS | gap_10 | black_vpvs_ratio | 938 | 1.000 | 0 / 14 | 29.79 | 45.89 | 29.79 | 0.785 | 105.92 | 0.06 | 0.007 |
| DTS | gap_50 | black_castagna_dts | 4,620 | 1.000 | 0 / 14 | 23.26 | 33.55 | 23.26 | 0.885 | 118.06 | 0.02 | 0.010 |
| DTS | gap_50 | black_gc_brine_gr | 4,620 | 1.000 | 0 / 14 | 13.97 | 20.82 | 13.97 | 0.940 | 80.58 | 0.12 | 0.015 |
| DTS | gap_50 | black_vpvs_ratio | 4,620 | 1.000 | 0 / 14 | 43.86 | 64.48 | 43.86 | 0.658 | 105.92 | 0.06 | 0.008 |
| DTS | gap_100 | black_castagna_dts | 9,219 | 1.000 | 0 / 14 | 19.96 | 29.02 | 19.95 | 0.933 | 118.06 | 0.02 | 0.010 |
| DTS | gap_100 | black_gc_brine_gr | 9,219 | 1.000 | 0 / 14 | 16.29 | 23.13 | 16.29 | 0.914 | 80.58 | 0.13 | 0.016 |
| DTS | gap_100 | black_vpvs_ratio | 9,219 | 1.000 | 0 / 14 | 27.57 | 52.08 | 27.55 | 0.834 | 105.92 | 0.06 | 0.006 |

RHOB in g/cm³, DTC and DTS in µs/ft. Fill values shown as 1.000 are rounded. For example, the mudrock
missing-curve fill is 116,081 / 116,091.

**Zero-support cases.** Every zero-fill well above is a gap window in which **NPHI is missing over the whole
window**. In none of them is NPHI outside [0, 1]. Across all scenarios, NPHI is missing on 10–22% of masked,
measured RHOB/DTC depths. Negative apparent NPHI affects only about 0.05% of rows. For whole DTS curves,
9 masked measured depths have no DTC, and 1 lies beyond the mudrock floor (DTC ≥ 178.46 µs/ft). These
account for the 10 depths the mudrock leaves unfilled.

## Head-to-head on common support

Each block restricts all listed models to the depths where **every one of them** predicts, so the pooled
MAE no longer rewards abstention. "Rows (wells)" is the common denominator. Interval columns compare only
models that give intervals, on depths where all of them do (here the same rows). `template_linear` is the
registered example user model from `workstreams/_template_model`. `ref_linear_interp` is the gap-only
straight-line reference described under "Reuse".

| Target | Scenario | Rows (wells) | Model | MAE | Per-well MAE | cover_90 | Width 90 |
|---|---|---:|---|---:|---:|---:|---:|
| RHOB | missing_curve | 198,776 (20) | black_gardner | 0.121 | 0.121 | 0.948 | 0.497 |
| | | | black_gardner_fixed | 0.123 | 0.124 | 0.942 | 0.507 |
| | | | black_density_nphi | 0.243 | 0.251 | 0.974 | 1.019 |
| | | | template_linear | **0.095** | **0.098** | 0.936 | 0.453 |
| RHOB | gap_10 | 1,206 (18) | black_gardner | 0.131 | 0.131 | 0.907 | 0.497 |
| | | | black_gardner_fixed | 0.138 | 0.138 | 0.910 | 0.507 |
| | | | black_density_nphi | 0.232 | 0.232 | 0.962 | 1.019 |
| | | | template_linear | 0.098 | 0.098 | 0.902 | 0.453 |
| | | | ref_linear_interp | **0.052** | **0.052** | – | – |
| RHOB | gap_50 | 5,610 (17) | black_gardner | 0.102 | 0.102 | 0.989 | 0.497 |
| | | | black_gardner_fixed | 0.102 | 0.102 | 0.991 | 0.507 |
| | | | black_density_nphi | 0.228 | 0.228 | 0.971 | 1.019 |
| | | | template_linear | 0.078 | 0.078 | 0.964 | 0.452 |
| | | | ref_linear_interp | **0.062** | **0.062** | – | – |
| RHOB | gap_100 | 11,854 (18) | black_gardner | 0.121 | 0.121 | 0.971 | 0.497 |
| | | | black_gardner_fixed | 0.127 | 0.127 | 0.971 | 0.507 |
| | | | black_density_nphi | 0.258 | 0.258 | 0.983 | 1.019 |
| | | | template_linear | 0.101 | 0.101 | 0.954 | 0.456 |
| | | | ref_linear_interp | **0.068** | **0.068** | – | – |
| DTC | missing_curve | 199,590 (20) | black_wyllie_nphi | 13.38 | 12.92 | 0.863 | 56.06 |
| | | | template_linear | **9.87** | **9.60** | 0.926 | 39.49 |
| DTC | gap_10 | 1,139 (17) | black_wyllie_nphi | 15.66 | 15.66 | 0.835 | 56.06 |
| | | | template_linear | 8.55 | 8.55 | 0.968 | 44.29 |
| | | | ref_linear_interp | **2.74** | **2.74** | – | – |
| DTC | gap_50 | 5,280 (16) | black_wyllie_nphi | 12.85 | 12.85 | 0.856 | 56.06 |
| | | | template_linear | 8.98 | 8.98 | 0.926 | 39.53 |
| | | | ref_linear_interp | **5.41** | **5.41** | – | – |
| DTC | gap_100 | 11,203 (17) | black_wyllie_nphi | 10.48 | 10.48 | 0.937 | 56.06 |
| | | | template_linear | 9.29 | 9.29 | 0.931 | 37.98 |
| | | | ref_linear_interp | **6.86** | **6.86** | – | – |
| DTS | missing_curve | 116,081 (14) | black_castagna_dts | 28.88 | 28.00 | 0.858 | 118.06 |
| | | | black_gc_brine_gr | 17.03 | 17.09 | 0.904 | 80.58 |
| | | | black_vpvs_ratio | 30.67 | 31.24 | 0.800 | 105.92 |
| | | | template_linear | **16.43** | **16.91** | 0.884 | 61.70 |
| DTS | gap_10 | 938 (14) | black_castagna_dts | 25.72 | 25.72 | 0.851 | 118.06 |
| | | | black_gc_brine_gr | 14.81 | 14.81 | 0.894 | 80.58 |
| | | | black_vpvs_ratio | 29.79 | 29.79 | 0.785 | 105.92 |
| | | | template_linear | 15.83 | 15.83 | 0.905 | 60.47 |
| | | | ref_linear_interp | **7.39** | **7.39** | – | – |
| DTS | gap_50 | 4,620 (14) | black_castagna_dts | 23.26 | 23.26 | 0.885 | 118.06 |
| | | | black_gc_brine_gr | **13.97** | **13.97** | 0.940 | 80.58 |
| | | | black_vpvs_ratio | 43.86 | 43.86 | 0.658 | 105.92 |
| | | | template_linear | 19.03 | 19.03 | 0.879 | 63.76 |
| | | | ref_linear_interp | 14.32 | 14.32 | – | – |
| DTS | gap_100 | 9,219 (14) | black_castagna_dts | 19.96 | 19.95 | 0.933 | 118.06 |
| | | | black_gc_brine_gr | **16.29** | **16.29** | 0.914 | 80.58 |
| | | | black_vpvs_ratio | 27.57 | 27.55 | 0.834 | 105.92 |
| | | | template_linear | 17.78 | 17.77 | 0.845 | 58.08 |
| | | | ref_linear_interp | 18.60 | 18.60 | – | – |

**What the field result says (20 test wells, 14 with DTS).**

- **DTS.** `black_gc_brine_gr` beats the mudrock line and the constant ratio in every scenario on identical
  depths. On whole missing curves it is 17.0 vs 28.9 µs/ft pooled MAE and 17.1 vs 28.0 per-well. Its 90%
  coverage stays within 0.90 ± 0.04 in every scenario (0.89–0.94), with intervals about two-thirds as wide
  as the mudrock's. It is within 0.6 µs/ft of the template regression on whole curves, beats that
  regression at every gap length, and beats interpolation on 50 m and 100 m gaps. The fitted ratio is the
  weakest DTS predictor, and its single σ under-covers (0.66–0.83).
- **RHOB.** The fitted Gardner is slightly better than the published coefficients (0.121 vs 0.123 on whole
  curves). Both lose to the multi-input template regression (0.095). Density from NPHI with a quartz/water
  matrix roughly doubles the error. Its σ is wide enough to over-cover (0.96–0.98), and it fills 85–90% of
  depths.
- **DTC.** Wyllie with NPHI as porosity fills 78–85% of depths, with 13.4 µs/ft MAE on whole curves versus 9.9 for
  the template on the same depths.
- **Gaps.** Straight-line interpolation between the measured flanks is the best gap filler for every target
  at 10 m. It also wins for RHOB and DTC at every length. For DTS, the GC mixture overtakes interpolation at 50 m
  and 100 m.

## Training-split DTS reference

The plan's training-split pilot was rerun with the released code, using the procedure written in the plan:
32 DTS wells sorted by name, `default_rng(2026)`, `array_split(..., 5)`, ratio and GR endpoints fitted on the other
four folds, and scoring on the common domain of all three predictions. It reproduces the plan's table exactly:
mudrock **17.39**, constant ratio **18.72** and GC brine/GR **15.85** µs/ft mean per-well MAE on 174,594 common
depths. GC is better than the mudrock line in **19 of 32** wells. This is a debugging reference only, not a benchmark score.

## Reuse: your model next to a black model on the same depths

Register your model in your workstream's `contender.py` (see `workstreams/_template_model/`), then name both
models in one run. They see the same tasks, hidden windows and scoring:

```powershell
python -m benchmark.check_model my_package.my_module:MyModel      # check an unregistered class
python -m benchmark.run --model black_gc_brine_gr,template_linear --dataset force2020 --target DTS --scenario gap_50
python -m workstreams.black_physics.validate_force --compare template_linear
```

Replace `template_linear` with your registered ID. The second command scored the 14 DTS windows of 50 m:
GC 13.97 µs/ft, template 19.03 µs/ft. `benchmark.run --model` needs a registered name. `check_model` also
accepts `module:Class`. The third command adds error on common support, fill fraction and per-well MAE for
your model next to all seven black models and the interpolation reference.

**Gap-only interpolation reference.** The station has no interpolation contender, so
`validate_force.linear_gap_reference` provides one. For each hidden run it uses the nearest measured target value
above and below **within the same continuous depth segment**, where a step greater than
`wmb.splits.max_gap_step` breaks the segment. It draws a straight line in depth between those two values and
abstains when either flank is unavailable. It is point-only and uses only `target_observed`.

**Your own LAS wells (`--scenario missing_curve`).** A dataset folder with a canonical `wells.json` manifest
under the configured data root runs through the same contract:

```powershell
python -m benchmark.run --model black_gardner_fixed --dataset my_wells --target RHOB --scenario missing_curve
```

This route was exercised with a throwaway manifest (outside the repository) over four Volve
`WLC_PETRO_COMPUTED_INPUT_1.LAS` files. Their curves were already in canonical units: m, µs/ft, g/cm³, v/v.
The station's seeded well split held out one well. Gardner fixed scored MAE 0.064 g/cm³, Gardner fitted 0.064,
density from NPHI 0.111 and the template 0.056. With one test well this checks the route only; it is not
a validation. Volve data: Equinor and the Volve licence partners. At this revision, a dataset without a task
spec supports only `missing_curve`; asking for a gap scenario stops with
`dataset my_wells has no task spec; only --scenario missing_curve is available`. Route custom gap-task
onboarding through the benchmark teammate. Units must be canonical before the run. The LAS loader converts
depth to metres but does not convert curve values.

**Genuine holes have no truth.** Measure error on artificially hidden measured intervals, as the gap tasks do.
On real missing sections, report only the fill fraction: the share of depths where the model returns a finite
prediction.

## Caveats

- Twenty test wells (14 with DTS) is a small sample. Treat sub-0.01 g/cm³ or sub-1 µs/ft differences as ties.
- All σ are one constant per model, and the station derives Gaussian intervals from it. Coverage near 0.90 on
  pooled depths does not imply calibration per well or per lithology.
- The GR adapter assumes brine-filled sand/shale everywhere. Carbonates, radioactive sands and hydrocarbon
  intervals violate it by construction, and these FORCE results do not test the full Greenberg–Castagna method.
- NPHI-based models use an instrument response as porosity. Their errors measure that proxy as much as the
  equations themselves.
