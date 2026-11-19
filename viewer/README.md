# FH2026 interactive well map

Shared tool. Run the scripts from this folder (`cd viewer`).

A data-agnostic, repo-ready viewer for well completeness and log plots.
The **code** lives here. **Data stays outside this folder** (and should not
be committed). To add wells, drop LAS/CSV files in a data folder, write a
`wells.json` manifest beside them, and rebuild.

```
python generate_force_manifest.py --force-dir /path/to/force2020 \
    --sodir-dir /path/to/sodir_factpages \
    --completeness /path/to/force_completeness.csv

python build_wellmap.py --root /path/to/force2020 --output wellmap.html
```

Open `wellmap.html` in a browser. Click a marker or a table row for the
log plot. Append `?well=15/9-13` to open a well directly. Volve (LAS) and
later NLOG wells plug in the same way: generate or write a manifest in the
data folder, then pass another `--root`.

## Manifest format (`wells.json`)

Canonical file name: `wells.json` in the **data root** (the folder that
holds the well files). Paths inside the manifest are relative to that
folder. The JSON Schema is [`manifest.schema.json`](manifest.schema.json)
(`manifest_version` is `1`).

A manifest has two parts:

1. **`dataset`** — `id`, `name`, `licence`, `source_url`, `credit`,
   optional `citation`, `depth_units` (`m` or `ft`), default `curve_map`,
   optional `lithology_column` / `lithology_scheme`, optional default
   `files`.
2. **`wells`** — one object per well: `name`, optional `split`,
   `lat`/`lon` (WGS84, EPSG:4326) **or** `x`/`y` plus `xy_crs`,
   `licence`/`credit` overrides, `curve_map` (standard property → source
   mnemonic or column), `files`, `depth_units`, optional
   `completeness`.

Standard property names the viewer plots:

`GR`, `RDEP`, `RMED`, `CALI`, `RHOB`, `NPHI`, `PEF`, `DTC`, `DTS`

plus optional lithology (`LITH` after mapping).

`files[].format` is `csv`, `csv-target` (a lithology table joined on well
+ depth), or `las`. For shared FORCE-style CSVs, many wells may point at
the same file; the builder reads each file once.

Minimal example:

```json
{
  "manifest_version": 1,
  "dataset": {
    "id": "nlog-demo",
    "name": "NLOG example wells",
    "licence": "CC BY 4.0",
    "source_url": "https://www.nlog.nl/",
    "credit": "NLOG / TNO",
    "depth_units": "m",
    "curve_map": { "GR": "GR", "RDEP": "RD", "DTC": "DT", "DTS": "DTS" }
  },
  "wells": [
    {
      "name": "F17-04",
      "lat": 54.12,
      "lon": 4.56,
      "crs": "EPSG:4326",
      "files": [{ "path": "las/F17-04.las", "format": "las" }]
    }
  ]
}
```

If `lat`/`lon` are omitted but `x`/`y` and `xy_crs` = `EPSG:23031`
(ED50 / UTM 31N) are present, the builder converts them to WGS84 with
EPSG:1613. Other CRS values are not auto-converted; put WGS84 lat/lon in
the manifest (an agent can do that when ingesting a new dataset).

### Helpers

- `generate_force_manifest.py` — builds a complete FORCE 2020 manifest
  from the competition CSVs, a completeness table, and Sodir FactPages
  wellbore tables. Writes `<force-dir>/wells.json`.
- `generate_las_manifest.py` — **stub**: scans a folder of `.las` files
  and writes a skeleton manifest (names + paths + guessed `curve_map`).
  You still need to fill `lat`/`lon` (or `x`/`y` + `xy_crs`) and confirm
  mnemonics. Used later for Volve / NLOG.

```
python generate_las_manifest.py --las-dir /path/to/volve/las \
    --dataset-id volve --dataset-name "Volve" \
    --licence "..." --credit "Equinor / Volve Data Village"
python build_wellmap.py --root /path/to/force2020 --root /path/to/volve/las
```

## Coordinates (FORCE 2020)

- Prefer Sodir FactPages `wlbNsDecDeg` / `wlbEwDecDeg`, matched on a
  normalised wellbore name (case, dashes, spaces, `7S` ↔ `7 S`). Sodir
  states these as **ED50** (EPSG:4230).
- Unmatched names fall back to median FORCE `X_LOC` / `Y_LOC`, assumed
  **ED50 / UTM zone 31N (EPSG:23031)**. This matched Sodir UTM to
  typically < 50 m for the 116 exact name matches, and every one of
  those wells is in zone 31.
- Both paths are transformed to WGS84 with **EPSG:1613** (“ED50 to WGS 84
  (24)”), the 7-parameter position-vector Helmert used for practical oil
  industry work in the Norwegian North Sea south of 62°N (~1 m).

## Viewer

Markers are coloured by overall completeness (red &lt; 60, amber 60–85,
green &gt; 85), shaped by split (circle train, square leaderboard,
triangle hidden), and given a heavier outline when the well has DTS.
Hover shows per-curve coverage. Click opens tracks vs depth (depth
increasing downward): GR; RDEP+RMED (log); RHOB+NPHI; DTC+DTS; CALI;
PEF; lithology strip. Gaps in a curve are left blank. Curves are
downsampled to at most 2,000 points per well.

Map tiles (OpenStreetMap) need a network connection. Leaflet and
a North Sea coastline are embedded, so markers still appear offline.

## Libraries and licences (MIT-compatible)

| Component | Licence |
|-----------|---------|
| This code | Intended for an MIT-licensed repo |
| Python / stdlib | PSF |
| pandas, numpy (build-time only) | BSD |
| Leaflet 1.9.4 (embedded) | BSD-2-Clause |
| Natural Earth coastline (embedded) | Public domain |

FORCE 2020 well logs are NLOD 2.0; lithofacies labels are CC-BY-4.0.
Credit: “Lithofacies data was provided by the FORCE Machine Learning
competition with well logs and seismic 2020.” Citation: Bormann P.,
Aursand P., Dilib F., Dischington P., Manral S. (2020). Source:
https://zenodo.org/records/4351156. Sodir FactPages are NLOD 2.0.

Do not commit `wellmap.html` (it embeds curve data) or the data roots.
