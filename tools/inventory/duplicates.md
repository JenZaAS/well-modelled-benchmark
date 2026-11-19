# FORCE 2020 vs Volve well overlap

Well names were normalised by uppercasing and stripping spaces, slashes, underscores and hyphens.
Examples: `15/9-F-11 A`, `15_9-F-11 A` and `15/9-F-11A` all become `159F11A`.

Volve wellbores were taken from the 12 folders under the team's shared Volve subset,
splitting out distinct wellbores that share a folder (`15/9-19 B` vs `15/9-19 BT2`,
and `15/9-F-11` vs `15/9-F-11 T2`). Completeness figures for Volve are computed separately.

## Exact matches

**None.** No Volve wellbore in this subset has the same normalised name as a FORCE 2020 well
(train, leaderboard or hidden).

This is expected: the FORCE 2020 wells in block 15/9 are exploration/appraisal holes
(`15/9-13`, `15/9-14`, `15/9-15`, `15/9-17`, `15/9-23`), while the Volve subset is the
Volve field development and 19-series wells (`15/9-19 A/B/BT2/SR` and `15/9-F-*`).

## Near-neighbours: FORCE wells in the 15/9 block (Volve area)

These five FORCE wells sit in the same NCS block as Volve. They are not the same wellbores.
All have lithology labels on 100% of samples in the logged interval. None has a name-collision
with a Volve well in this subset (there is no FORCE `15/9-19`).

| FORCE well | Split | Interval (m) | Samples | GR | RDEP | RMED | CALI | RHOB | NPHI | PEF | DTC | DTS | Overall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15/9-23 | hidden | 1518.28–3212.62 | 11063 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 99 | 94 | 98.9 |
| 15/9-14 | leaderboard | 480.63–3564.56 | 20281 | 100 | 100 | 100 | 100 | 100 | 100 | 82 | 100 | 0 | 86.9 |
| 15/9-15 | train | 485.26–3200.13 | 17717 | 100 | 100 | 100 | 100 | 99 | 75 | 100 | 100 | 0 | 85.9 |
| 15/9-17 | train | 472.43–3114.49 | 17350 | 100 | 100 | 100 | 100 | 96 | 80 | 96 | 100 | 0 | 85.7 |
| 15/9-13 | train | 494.53–3272.02 | 18270 | 100 | 100 | 100 | 100 | 100 | 77 | 90 | 100 | 0 | 85.1 |

## Volve wellbores in the subset

13 wellbores after grouping by filename + LAS WELL header (12 folders; 15/9-19 BT2 is split out of the 15/9-19 B folder).
There is no separate 15/9-F-11 row: `15_9-F-11_appendNTR.las` has LAS WELL header `15/9-F-11 T2`.

- 15/9-19 A
- 15/9-19 B
- 15/9-19 BT2
- 15/9-19 SR
- 15/9-F-1
- 15/9-F-1 A
- 15/9-F-1 B
- 15/9-F-10
- 15/9-F-11 A
- 15/9-F-11 T2
- 15/9-F-14
- 15/9-F-15 C
- 15/9-F-4


## FORCE split counts

| Split | Wells |
|---|---:|
| train | 98 |
| leaderboard | 10 |
| hidden | 10 |
| total | 118 |

