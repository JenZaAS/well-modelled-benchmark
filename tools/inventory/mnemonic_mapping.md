# Volve mnemonic mapping (FH2026 well inventory)

Coverage is computed from the LAS files themselves, not from `curves_inventory.csv` (that file lists mnemonics and depth ranges, but not valid-sample counts).

For each well and property, the chosen curve is the one with the highest coverage over the well's logged interval (first to last depth where any of GR, RDEP, RHOB or DTC is valid). Ties prefer measured/raw or PETRO input files over combined (`appendNTR`), LFP and CPI files, and ignore `_no_1` duplicate curves unless they are the only option.

## Property ← mnemonic rules

Mnemonics are compared case-insensitively, with `-` treated as `_`. A trailing `_no_1` / `_NO_1` is stripped for matching and the curve is deprioritised.

| Property | Mnemonics accepted | Notes |
|---|---|---|
| GR | GR, GRD, GRR, LFP_GR | Not GRMAX, GRMIN, SGR |
| RDEP (deep / true res) | RDEP, RD, LLD, RT, LFP_RT, HLLD, AT90, AHT90 | Not RXO, RMIC |
| RMED (medium res) | RMED, RM, LLS, LFP_RMED, HLLS, AT30, AHT30, RS | RS is used only as a fallback (often shallow). Not RSHA, LFP_RSH, RMIC |
| CALI | CALI, HCAL, LCAL, CAL, C1, C2, LFP_CALI, CALI_* | Not BS (bit size) |
| RHOB | RHOB, DEN, RHOZ, DENS, LFP_RHOB, LFP_RHOB_LOG | Not RHOFL, RHOMA, DRHO |
| NPHI | NPHI, NEU, TNPH, NPOR, PHIN, LFP_NPHI, CNC, CNPOR | |
| PEF | PEF, PE, PEFZ, PEFS, PEFL, LFP_PEF, LFP_PE | |
| DTC (P slowness) | DTC, DT, AC, DTCO, DT4P, AC04, LFP_DT, LFP_DT_LOG | DT is not matched as DTS |
| DTS (S slowness) | DTS, DTSM, ACS, DTSH, DT4S, DT2, LFP_DTS, LFP_DTS_LOG | ACS is mapped as DTS but flagged; this is the open F-10 question |
| PHIF | PHIF, PHIE, LFP_PHIE, LFP_PHIEC, LFP_PHIEWC | Preferred order: PHIF, PHIE, then LFP_* |
| PHIT | PHIT, LFP_PHIT, PHITC, LFP_PHITC, PHITWC, LFP_PHITWC | |
| VSH | VSH, LFP_VSH, VSHGR, LFP_VSHGR, VSHDN, LFP_VSHDN | Not dry-clay variants (VSHDRY*) |
| SW | SW, SWT, SWE, LFP_SWT, LFP_SWE, LFP_SWTC, LFP_SWEC | Not SWIRR |

LFP / CPI / PETRO computed curves are included as candidates (an open question is whether LFP should later be excluded from the *target* set). If a property is only present in an `*_edited.las`, an `appendNTR` combined file, or an LFP/CPI/PETRO output file, that is noted in the `extras` column.

## Mnemonics actually used (best pick per well)

| Property | Mnemonics chosen |
|---|---|
| GR | GR, LFP_GR, gr, lfp_gr |
| RDEP | RD, RDEP, RT, lfp_rt, rd, rt |
| RMED | RMED, RS, rm, rs |
| CALI | CALI, cali, lfp_cali |
| RHOB | DEN, LFP_RHOB_LOG, RHOB, lfp_rhob_log, rhob |
| NPHI | LFP_NPHI, NEU, NPHI, lfp_nphi, nphi |
| PEF | PEF, pef |
| DTC | AC, DT, dt, lfp_dt_log |
| DTS | ACS, DTS, LFP_DTS, dts, lfp_dts_log |
| PHIF | LFP_PHIE, PHIF, lfp_phie, phif |
| PHIT | LFP_PHIT, lfp_phit |
| VSH | LFP_VSH, VSH, lfp_vsh, vsh |
| SW | SW, lfp_swt, sw |

## All matched mnemonics seen in the subset

- **GR**: GR, LFP_GR
- **RDEP**: LFP_RT, RD, RDEP, RT
- **RMED**: RM, RMED, RS
- **CALI**: CALI, LFP_CALI
- **RHOB**: DEN, LFP_RHOB, LFP_RHOB_LOG, RHOB
- **NPHI**: LFP_NPHI, NEU, NPHI
- **PEF**: PEF
- **DTC**: AC, DT, LFP_DT, LFP_DT_LOG
- **DTS**: ACS, DTS, LFP_DTS, LFP_DTS_LOG
- **PHIF**: LFP_PHIE, LFP_PHIEC, LFP_PHIEWC, PHIF
- **PHIT**: LFP_PHIT, LFP_PHITC, LFP_PHITWC
- **VSH**: LFP_VSH, LFP_VSHDN, LFP_VSHGR, VSH
- **SW**: LFP_SWE, LFP_SWEC, LFP_SWECNC, LFP_SWT, LFP_SWTC, LFP_SWTCNC, SW

## Unmapped mnemonics (first 80, alphabetical)

These were present in at least one LAS `~C` section and were not assigned to a property. Resistivity, sonic or density names in this list are worth a second look.

ABDCQF01, ABDCQF02, ABDCQF03, ABDCQF04, BS, BVW, BWV, CARB_FLAG, COAL_FLAG, DELTA, DENC, DRHO, DRHU, GRMAX, GRMIN, J, KLHC_INT, KLOGH, KLOGH_NEW, KLOGV, LFP_AI, LFP_AI_B, LFP_AI_G, LFP_AI_LOG, LFP_AI_O, LFP_AI_V, LFP_API, LFP_BADDATA, LFP_BS, LFP_BVWE, LFP_BVWT, LFP_COAL, LFP_DTCORFSFLAG, LFP_DTLOGFLAG, LFP_DTSLOGFLAG, LFP_DTS_B, LFP_DTS_G, LFP_DTS_GC, LFP_DTS_O, LFP_DTS_V, LFP_DT_B, LFP_DT_G, LFP_DT_O, LFP_DT_SYNT, LFP_DT_V, LFP_EI_HILT_B, LFP_EI_HILT_G, LFP_EI_HILT_O, LFP_EI_HILT_V, LFP_GAS, LFP_GOR, LFP_GRMAX, LFP_GRMIN, LFP_KFLG, LFP_KFLMF, LFP_KFLO, LFP_KFLW, LFP_KROCKV, LFP_KRROCKV, LFP_KSKEL, LFP_KVROCKV, LFP_MODEL, LFP_NPHISH, LFP_OIL, LFP_PR, LFP_PRESS, LFP_PR_B, LFP_PR_G, LFP_PR_LOG, LFP_PR_O, LFP_PR_SH_SYNT, LFP_PR_SS_B_SYNT, LFP_PR_SS_G_SYNT, LFP_PR_SS_O_SYNT, LFP_PR_V, LFP_RHOBFAR_HILT_B, LFP_RHOBFAR_HILT_G, LFP_RHOBFAR_HILT_O, LFP_RHOBFAR_HILT_V, LFP_RHOBLOGFLAG

… and 117 more.

Possibly relevant unmapped names: DENC, DRHO, GRMAX, GRMIN, LFP_DTCORFSFLAG, LFP_DTLOGFLAG, LFP_DTSLOGFLAG, LFP_DTS_B, LFP_DTS_G, LFP_DTS_GC, LFP_DTS_O, LFP_DTS_V, LFP_DT_B, LFP_DT_G, LFP_DT_O, LFP_DT_SYNT, LFP_DT_V, LFP_GRMAX, LFP_GRMIN, LFP_NPHISH, LFP_PRESS, LFP_RHOBFAR_HILT_B, LFP_RHOBFAR_HILT_G, LFP_RHOBFAR_HILT_O, LFP_RHOBFAR_HILT_V, LFP_RHOBLOGFLAG, LFP_RHOBSH, LFP_RHOBSHNC, LFP_RHOB_B, LFP_RHOB_G, LFP_RHOB_O, LFP_RHOB_SYNT, LFP_RHOB_V, LFP_RHOFLE, LFP_RHOFLEC, LFP_RHOFLEWC, LFP_RHOFLT, LFP_RHOFLTC, LFP_RHOFLTWC, LFP_RHOG, LFP_RHOHA, LFP_RHOHW, LFP_RHOMA, LFP_RHOMANC, LFP_RHOMF, LFP_RHOO, LFP_RHOW, LFP_VQRTZ, LFP_VQRTZC, LFP_VQRTZWC, LFP_VSHDRY, LFP_VSHDRYC, LFP_VSHDRYWC, NBGRCFM, PORD, RACEHM, RACELM, RHOFL, RHOMA, SWIRR

## ACS mapped as DTS

Wells where the chosen DTS curve is mnemonic `ACS`: 15/9-F-10.
This follows the requested mapping (DTS/DTSM/ACS) but it is still an open question whether F-10 ACS is shear sonic.

## Wellbore splitting

Files are grouped by wellbore, not only by folder. So `15/9-19 B` and `15/9-19 BT2` (same folder) are separate rows, and `15/9-F-11` and `15/9-F-11 T2` are separate rows if both appear in filenames or LAS headers.

## 15/9-F-11 vs 15/9-F-11 T2

There is no separate `15/9-F-11` row. Both `15_9-F-11 T2_appendNTR.las` and
`15_9-F-11_appendNTR.las` (and the PETRO input/output files in that folder)
have LAS WELL header `15/9-F-11 T2`, so they were grouped as 15/9-F-11 T2.
Sodir lists 15/9-F-11, 15/9-F-11 A and 15/9-F-11 B, but not T2.

## 15/9-19 B vs 15/9-19 BT2

`15/9-19 B` has only `15_9-19 B_appendNTR.las` in this subset (no separate raw
or CPI file). The other files in that folder belong to `15/9-19 BT2`. CURVES.md
notes that caliper is not present on 15/9-19 B; this inventory also finds no RMED or PEF for that wellbore.
