# Benchmark definition (draft for the team to agree)

**Task:** hide one log, predict it from the others, and say how sure we are.

## Scenarios
1. **Missing curve** (main): the target curve is removed from whole test wells; train on other wells. Split by whole wells (no leakage).
2. **Gap filling:** random 20–50 m gaps masked inside test wells (`wmb.splits.gap_mask`).
3. **Lithology classification:** FORCE 2020 labels, official train / leaderboard / hidden split, scored with the contest penalty matrix.
4. **Fill every real gap (speed test):** every approach fills every real gap in the whole database on the same laptop CPU, and we time it. Matters most on the largest dataset.

## Targets (first)
RHOB, DTC and DTS first (DTS is the most often missing and the most useful). Then PEF and NPHI.
Not porosity, Vshale or Sw computed by us; if used, take independent CPI curves.

## Datasets
FORCE 2020 (118 wells, official split), Volve, NLOG (Netherlands), plus other open sets with a `wells.json`
(see `data/README.md`). Diskos only if access is given; then only scores and aggregates are published.

## Contenders
Black (physics), blue (boosted trees), red (time-series foundation models). Every contender implements
`wmb.contender.BaseContender`; the runner treats them all the same.

## Output
`results/<contender>/<dataset>/<target>/scores.json` and one scoreboard (`python -m benchmark.scoreboard`).
