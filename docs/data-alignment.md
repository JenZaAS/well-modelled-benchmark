# Aligning a well-log dataset to the FORCE standard

Within one dataset segment, every well must use the **same curve names, the
same units, and the same depth reference**. FORCE 2020 is the reference
standard. This page is the procedure for bringing another public dataset up to
that standard so a test written for FORCE names can run on it.

The code lives in `src/wmb/align/` and is generic: it never writes data into
the git repository. You run it against a local folder and it writes an aligned
copy next to your other local data. **Never commit LAS, CSV, weights or
manifests that embed well curves.**

```bash
python -m wmb.align --input /path/to/raw/NLOG --output /path/to/aligned/NLOG --dataset nlog_dts
python -m wmb.align --input /path/to/raw/Kansas-KGS-DTS --output /path/to/aligned/Kansas-KGS --dataset kansas_kgs_dts
```

Point `data_paths.local.yaml` at the *aligned* folder (`nlog_dts:`,
`kansas_kgs_dts:`), not at the raw download. Then, if the folder is present,
optional tests such as `nlog_dts_missing` and `kansas_dts_missing` will run;
if it is absent, `python -m benchmark.run --test all` skips them.

## The standard (FORCE 2020)

Names and units below are what official FORCE 2020 LAS / contest CSV files
use. Depth is metres measured depth (MD). Nulls are `-999.25` (loaders turn
these, and a few other sentinels, into NaN).

| Standard name | Unit | Typical valid range (flag, do not drop) | Role |
|---|---|---|---|
| `DEPTH_MD` | m | 0–8000 | Index. In memory this is `Well.depth`. |
| `GR` | gAPI | 0–500 | Gamma ray |
| `RDEP` | ohm.m | 0.01–100000 | Deep / true resistivity |
| `RMED` | ohm.m | 0.01–100000 | Medium resistivity |
| `CALI` | in | 4–30 | Hole caliper (never use bit size) |
| `RHOB` | g/cm3 (g/cc) | 1.0–3.5 | Bulk density |
| `NPHI` | v/v (fraction, also written m3/m3) | −0.15–0.70 | Neutron porosity |
| `PEF` | b/e | 0–20 | Photoelectric factor |
| `DTC` | us/ft | 40–200 (soft sediment can be a little higher; treat a median above ~250 as possible us/m) | Compressional slowness |
| `DTS` | us/ft | 60–500 (soft shale can reach ~400; a 60–600 window is also used as a soft check) | Shear slowness |

Optional extras, kept when present, not used as benchmark targets:

- `BS` (bit size, in), `RSHA` (shallow resistivity, ohm.m).
- `TVD` / `TVDSS` when they already exist in the source (metres). See [Depth reference](#depth-reference-md-vs-tvd).

The alias lists, unit spellings, conversions and guardrails are in
`src/wmb/datasets/canonical_curves.yaml`. A per-dataset pack
(`datasets/<id>/mapping.yaml`) selects the alias set (`NLOG-DTS`,
`Kansas-KGS-DTS`, `Volve`, `FORCE-2020`) and records the licence and credit
line. Do not compute porosity, Vshale or water saturation as targets; if a
source file already has those curves you may display them, but they are not
part of this standard and the aligner does not invent them.

## Depth reference: MD vs TVD

Tests currently use **measured depth (MD)**. That is the index in FORCE 2020
(`DEPTH_MD`) and in every aligned file this tool writes.

On a deviated well, MD is longer than true vertical depth (TVD). Formations
that are a given thickness in TVD are *stretched* along the MD axis, so a
"50 m gap" is 50 m of hole, not 50 m of true vertical section. Many NLOG
wells are clearly deviated (published TD MD and TD TVD differ by more than
10 m). A few NLOG and Kansas files already contain a `TVD` / `TVDSS` curve
or an inclination / azimuth / deviation survey.

**Future work**, not done by this aligner:

- Flag wells as deviated from a survey (or from TD MD vs TD TVD in the
  well header) so a test can exclude or group them.
- An optional `TVD` / `TVDSS` column on the aligned well, still only when
  that curve is already in the source.
- Tests that choose MD or TVD as the depth axis.

If deviation, inclination or TVD data is already present in NLOG, Kansas or
FORCE files, the aligner **records it in the manifest** under
`wells[].depth_reference` (mnemonics, source file, unit, valid-sample count,
and header TD MD / TD TVD when the source manifest has them). It **does not
compute TVD** from a survey or from MD and inclination. When a source TVD
curve exists, it may be copied into the aligned LAS (converted to metres if
it was in feet) and listed in the manifest; that is pass-through, not a
computed trajectory.

## Step-by-step process

Work one dataset segment at a time (for example "NLOG DTS wells", not a mix
of NLOG and Kansas in one folder).

1. **Inventory names and units.** For every LAS (and LAS-like frame extract),
   list mnemonics, units and a short value summary (count of valid samples,
   median, min, max). Presence of a name is not enough: many headers have
   empty or all-null curves. Note depth units (m vs ft; reject a `.1IN`
   index). Record the licence and the required credit line now, from the
   source README, not from memory.
2. **Map names with a documented priority rule.** Use the alias list in
   `canonical_curves.yaml` for this dataset. Matching is case-insensitive;
   `-` counts as `_`; a trailing `:N` or `_NO_N` is stripped and that copy
   is deprioritised. When several candidates exist in one well:
   - first listed alias wins a coverage tie,
   - then the longest valid depth interval,
   - prefer raw / measured files over LFP, CPI, PETRO or other computed
     curves,
   - never use a curve that fails a unit or value guardrail, even if the
     name matches.
   Special priority rules worth repeating:
   - **RDEP** is the deepest-reading resistivity (90-inch / LLD / ILD / RT /
     RT90 / AT90 / AHT90 / AF90 / AO90) *before* a 60-inch curve. Never
     prefer AF60 over AF90 on a tie.
   - **RMED** prefers 30-inch over 20-inch. `RS` is a last, flagged fallback
     (often shallow).
   - **DTC** prefers DTCO / DTC / DTRP over a generic DT.
   - **DTS** prefers DTSM / DTS / DTRS. **Never** map `DTST` (Stoneley) as
     shear.
   - **CALI** prefers a true caliper (CALI, HCAL, CAL, CALIPER). `DCAL` is
     accepted only when the description or the values say it is hole size,
     not a differential caliper. Never use bit size (`BS`) as caliper.
   - **NPHI** prefers a stated or default matrix (NPHI, TNPH) over a
     single-matrix curve (NPHS, CNLS, NPHD); record the matrix in a flag.
3. **Convert units** to the standard in the table above. Normalise spellings
   first (`OHMM` and `ohm.m` are the same). Common conversions:
   - depth ft → m (× 0.3048)
   - sonic us/m → us/ft (× 0.3048)
   - neutron percent → v/v (÷ 100), but only after a value check: if the
     unit is PU or `%` and the median is already between about −0.2 and 0.7,
     it is already a fraction
   - density kg/m3 → g/cm3 (× 0.001), including when a curve labelled g/cm3
     has a median above 50
   - caliper mm / cm → in
   - conductivity mS/m (or MS/M, MM/M, MMHO) → ohm.m as `1000 / C`, and
     flag it. Do not treat conductivity as resistivity without that step.
   If the unit is missing, infer it from the median against the plausible
   ranges and flag the inference. If the labelled unit and the values
   disagree (DTC labelled us/ft with median > 250; NLOG Q13-12 is a known
   us/m well), trust the values, convert, and flag.
4. **Standardise nulls** to NaN. Treat at least `-999.25`, `-999`, `-9999`,
   `-999.95`, `999.25`, `9999`, `-32768` as missing. Aligned files may write
   `-999.25` as the LAS null; loaders convert that back to NaN.
5. **Check depth.** Convert to metres first. Make the index strictly
   increasing (reverse a file that is logged upwards; sort if needed and
   flag it). Use a regular MD grid at the well's native step (snapped to a
   common logging interval such as 0.1 m or 0.1524 m). Assign each source
   sample to the nearest grid node only when it falls within half a step.
   **Do not interpolate across gaps.** When several files cover one well,
   the highest-priority source is primary; lower-priority files may fill
   remaining NaNs only. Do not average overlapping values.
6. **Range flags.** After conversion, count samples outside the ranges in
   the table and flag a median that sits outside the typical window.
   **Flag, do not silently drop** those samples, and do not edit the source
   files. Reject a *candidate curve* only when it is the wrong physical
   quantity (Stoneley as DTS, RD with unit inches, a caliper median of
   thousands of "inches", an empty curve). Leave a note in the well's
   `rejected` list.
7. **Provenance.** For every standard curve, record the original mnemonic,
   source file (path relative to the raw folder), original unit, conversion
   flags, and any extra files used only to fill gaps. This goes in
   `wells.json` as `curve_map` and `curve_sources`.
8. **Completeness table.** One row per well, one column per standard curve,
   yes/no (at least one valid sample) and, usefully, a coverage percentage
   on the logged interval (first to last depth where any of GR, RDEP, RHOB
   or DTC is valid). Write `completeness.csv` next to `wells.json`.
9. **Review.** Spot-check wells that took a fallback alias, a unit
   inference, a conductivity conversion, or a `dcal_accepted_as_caliper`
   flag. Confirm DTS > DTC on the same interval after conversion. Confirm
   the README still has the source, licence and required credit line, copied
   from the raw-data README.

The aligned folder should look like:

```
<aligned-dataset>/
  README.md              # source, licence, required credit
  wells.json             # manifest_version 1, one LAS per well, metres MD
  completeness.csv
  alignment_summary.json
  <well>/aligned.las     # DEPTH_MD, GR, RDEP, … standard names and units
```

In the aligned `wells.json`, `curve_map` is the mnemonic *in the aligned file* (so `DTS: DTS`, and so on). The original raw name and file are in `source_curve_map` and `curve_sources`. Load with `wmb.io.iter_manifest_wells(<aligned-folder>)`.

## What agents should focus on

- Mapping and unit problems (the same property under many names; NPHI scale;
  mixed depth units; resistivity aliases that are actually conductivity,
  caliper or counts).
- Honest completeness: valid-sample counts, not header presence.
- A clear priority rule when several candidates exist, especially deep vs
  shallow resistivity and compressional vs shear vs Stoneley sonic.
- Provenance and flags that a person can review.
- Licence and credit, copied verbatim from the source.

## What agents must not do

- Do not interpolate across gaps, splice by averaging, or invent a curve
  that is not in the source files.
- Do not silently drop out-of-range samples or whole wells. Flag them.
- Do not compute porosity, Vshale or water saturation, and do not compute
  TVD from a survey.
- Do not treat `DTST` as DTS, bit size as caliper, or conductivity as
  resistivity without an explicit, flagged conversion.
- Do not commit data, weights, or `data_paths.local.yaml`. Do not put
  private folder paths, vendor-internal names or personal names in repo
  files. Do not replace a source licence with a different one.
- Do not point the benchmark at the *raw* multi-file folder once an aligned
  copy exists: `iter_manifest_wells` yields one well object per LAS file.

## Agent checklist

When you are asked to align a new public dataset (or to review an alignment):

- [ ] Confirm it is public open data and copy the licence plus the required
      credit line into the dataset README (and into `datasets/<id>/mapping.yaml`).
- [ ] Inventory mnemonics, units, depth reference and valid-sample counts.
- [ ] Add or extend a mapping file; reuse `canonical_curves.yaml` aliases
      and priority rules; document any new alias.
- [ ] Convert units to the FORCE standard; flag inferences and conversions.
- [ ] Nulls → NaN; depth in metres MD, strictly increasing; no interpolation.
- [ ] Range checks flagged, not dropped; rejected wrong-quantity candidates listed.
- [ ] Provenance per well and curve (original name, file, unit).
- [ ] Record existing TVD / inclination / deviation in the manifest; do not compute TVD.
- [ ] Completeness table (well × curve yes/no).
- [ ] Write aligned files *outside* git; never `git add` them.
- [ ] Unit tests for the new rules (synthetic LAS, no real well data in the repo).
- [ ] Optional benchmark test that runs only when the aligned folder is present locally.
- [ ] Spot-check flagged wells before telling anyone it is ready.

## Optional benchmark tests

`benchmark/test_defs/nlog_dts_missing.yaml` and
`kansas_dts_missing.yaml` hide DTS on the aligned wells (the model's training
split is a well-level split of that dataset). They are skipped when the
aligned data is not on the machine. They are a transfer-style check of how a
model behaves on a less uniform area than FORCE, after alignment, not a
replacement for the official FORCE tests.
