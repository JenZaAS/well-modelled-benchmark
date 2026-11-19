#!/usr/bin/env python3
"""Write nlog wells.json and print a README table from details + completeness."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import os as _os
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "viewer"))
from wellmaplib.manifest import validate_manifest, load_manifest

ROOT = Path(_os.environ.get("FH2026_DATA", Path.home() / "fh2026-cache" / "data")) / "nlog"
META = Path(_os.environ.get("FH2026_WORK", Path.home() / "fh2026-cache" / "work")) / "nlog-meta"
CORE = ["GR", "RDEP", "RMED", "CALI", "RHOB", "NPHI", "PEF", "DTC", "DTS"]

LICENCE = (
    "NLOG disclaimer (not a Creative Commons licence). Users may copy, download, "
    "disclose, distribute, process or edit the information, provided NLOG.NL is quoted "
    "as the source. The Dutch-language disclaimer prevails if the translation differs. "
    "See https://www.nlog.nl/en/disclaimer (Dutch: https://www.nlog.nl/disclaimer)."
)
CREDIT = "Data from NLOG.NL / TNO – Geological Survey of the Netherlands."
CITATION = (
    "NLOG (TNO / Geological Survey of the Netherlands). Digital well logs downloaded "
    "from https://www.nlog.nl on 19 November 2026. Shear-sonic shortlist guided by "
    "Hunfeld, L.B., Foeken, J.P.T. and van Kempen, B.M.M. (20 December 2021), "
    "'Geomechanical parameters derived from compressional and shear sonic logs for "
    "main geothermal targets in The Netherlands', "
    "https://www.nlog.nl/sites/default/files/2021-12/data_selection_and_methods.pdf."
)

FIRST15 = {
    "BLESDIJKE-01", "D12-A-02", "AMSWEER-12", "COEVORDEN-40", "GEESBRUG-01",
    "HUIBEVEN-01", "IJSSELMONDE-64-SIDETRACK2", "MAASGEUL-03-SIDETRACK1",
    "MIDDELIE-302", "P18-A-02", "PERNIS_WEST-01", "Q11-03",
    "ROTTERDAM_SCHULPWEG-02", "WAALWIJK_SOUTH-02", "WESTBEEMSTER-01",
}
BATCH2 = {
    "HEMPENS-01", "LUTTELGEEST-01", "SLOOTDORP-06-SIDETRACK1", "GROET_OOST-01",
    "MIDDELIE_ZEE-01", "WESTBEEMSTER-02", "P15-RIJN-A-09-SIDETRACK1", "Q14-03",
    "P15-F-02", "P15-14", "Q13-12", "Q13-A-02", "Q13-A-05-SIDETRACK1",
    "NAALDWIJK-GT-01", "CAPELLE-01", "DORDRECHT-01", "WERKENDAM-03",
    "WAALWIJK-NOORD-03", "NUMANSDORP-03",
}
BATCH3 = {
    "D12-07", "P15-11", "P15-13-SIDETRACK1", "P15-03", "P15-04", "P15-07-S1",
    "P15-RIJN-A-03", "P15-RIJN-A-04", "P15-RIJN-A-05", "P18-A-03-SIDETRACK2",
    "Q11-01-SIDETRACK2", "Q11-02", "Q13-A-03", "Q13-A-04", "Q14-01", "Q14-02",
    "AMSWEER-12-SIDETRACK1", "P15-RIJN-A-09-SIDETRACK2", "IJSSELMONDE-64",
}


def is_las_filename(name: str) -> bool:
    low = name.lower()
    if low.endswith(".las"):
        return True
    stem, _, ext = low.rpartition(".")
    return ext.startswith("a") and ext[1:].isdigit()


def year_from(date):
    if not date:
        return None
    s = str(date)
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def intish(v):
    x = num(v)
    if x is None:
        return None
    if float(x).is_integer():
        return int(x)
    return x


def batch_of(folder: str) -> str:
    if folder in FIRST15:
        return "first"
    if folder in BATCH2:
        return "tno_remainder"
    if folder in BATCH3:
        return "extra"
    return "other"


def main() -> None:
    mapping = json.loads((ROOT / "mnemonic_mapping_applied.json").read_text())
    by_folder = {}
    for w in mapping["wells"]:
        by_folder[w["folder"]] = w

    csv_rows = {}
    with (ROOT / "nlog_completeness.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csv_rows[row["well"]] = row

    wells_out = []
    size_rows = []
    for wdir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        folder = wdir.name
        rec = by_folder.get(folder)
        if not rec:
            raise SystemExit(f"No mapping record for {folder}")
        det = {}
        det_path = wdir / "nlog_details.json"
        if det_path.is_file():
            det = json.loads(det_path.read_text())
            if isinstance(det, list):
                det = det[0] if det else {}

        official = det.get("boreholeName") or rec.get("well") or folder
        short = det.get("shortName") or rec.get("shortName")
        dbk = det.get("boreholeDbk")
        lat = num(det.get("latitudeWgs84"))
        lon = num(det.get("longitudeWgs84"))
        x = intish(det.get("xutm31Wgs84Crd"))
        y = intish(det.get("yutm31Wgs84Crd"))
        lat_nat = det.get("latitude")
        lon_nat = det.get("longitude")
        crs_code = det.get("coordSystemCode") or "RD"
        note = (
            f"Native NLOG coordSystemCode={crs_code}; also published latitude/longitude "
            f"in that system ({lat_nat}, {lon_nat}). Map position uses NLOG's WGS84 pair, "
            f"not a local transform."
        )

        las_files = sorted(
            p for p in wdir.iterdir() if p.is_file() and is_las_filename(p.name)
        )
        las_bytes = sum(p.stat().st_size for p in las_files)

        # curve_map strings + multi-file sources
        cmap_str = {}
        sources = {}
        file_curves = defaultdict(list)
        raw_map = rec.get("curve_map") or {}
        for prop in CORE:
            ch = raw_map.get(prop)
            if not ch:
                continue
            mnem = ch.get("mnemonic")
            if not mnem:
                continue
            cmap_str[prop] = mnem
            primary_path = f"{folder}/{ch['file']}" if ch.get("file") else None
            also = []
            for a in ch.get("also") or []:
                also.append({
                    "mnemonic": a.get("mnemonic"),
                    "path": f"{folder}/{a['file']}" if a.get("file") else None,
                    "n_valid_grid": a.get("n_valid_grid"),
                })
                if a.get("file"):
                    if prop not in file_curves[a["file"]]:
                        file_curves[a["file"]].append(prop)
            sources[prop] = {
                "mnemonic": mnem,
                "path": primary_path,
                "unit": ch.get("unit"),
                "n_valid_grid": ch.get("n_valid_grid"),
                "also": also,
            }
            if ch.get("file") and prop not in file_curves[ch["file"]]:
                file_curves[ch["file"]].append(prop)

        files = []
        for p in las_files:
            item = {
                "path": f"{folder}/{p.name}",
                "format": "las",
                "well_name": official,
            }
            contrib = file_curves.get(p.name) or []
            if contrib:
                # keep CORE order
                item["curves"] = [c for c in CORE if c in contrib]
            files.append(item)

        csv_row = csv_rows.get(official) or csv_rows.get(folder)
        if not csv_row:
            raise SystemExit(f"No completeness row for {official} / {folder}")

        def fnum(k):
            v = csv_row.get(k)
            if v in (None, ""):
                return None
            return float(v)

        curves_pct = {c: int(csv_row[c]) for c in CORE}
        well = {
            "name": official,
            "dataset": "nlog",
            "split": "nlog",
            "short_name": short,
            "nlog_id": dbk,
            "nlog_url": f"https://www.nlog.nl/nlog-mapviewer/brh/{dbk}" if dbk else None,
            "lat": lat,
            "lon": lon,
            "crs": "EPSG:4326",
            "x": x,
            "y": y,
            "xy_crs": "EPSG:32631" if x is not None and y is not None else None,
            "coord_source": "NLOG mapviewer brh/details fields latitudeWgs84 and longitudeWgs84 (published WGS84)",
            "coord_source_crs": "EPSG:4326",
            "coord_note": note,
            "licence": LICENCE,
            "credit": CREDIT,
            "depth_units": "m",
            "operator": det.get("clientOrgName"),
            "legal_owner": det.get("legalOwnerName"),
            "spud_date": det.get("startDate"),
            "spud_year": year_from(det.get("startDate")),
            "td_md_m": intish(det.get("endAhDepthInMeters")),
            "td_tvd_m": num(det.get("tvdInMeters")),
            "field": det.get("fieldName"),
            "onshore_offshore": det.get("onOffshore"),
            "curve_map": cmap_str,
            "curve_sources": sources,
            "files": files,
            "completeness": {
                "overall": fnum("overall"),
                "depth_from": fnum("depth_from"),
                "depth_to": fnum("depth_to"),
                "n_samples": int(float(csv_row["n_samples"])),
                "curves": curves_pct,
            },
        }
        wells_out.append(well)
        size_rows.append({
            "folder": folder,
            "name": official,
            "short": short or "",
            "batch": batch_of(folder),
            "n_las": len(las_files),
            "bytes": las_bytes,
            "mb": round(las_bytes / (1024 * 1024), 1),
            "dts": curves_pct["DTS"],
            "dtc": curves_pct["DTC"],
            "overall": fnum("overall"),
            "depth_from": fnum("depth_from"),
            "depth_to": fnum("depth_to"),
            "n_samples": int(float(csv_row["n_samples"])),
            "operator": det.get("clientOrgName") or "",
            "spud": year_from(det.get("startDate")),
            "td": intish(det.get("endAhDepthInMeters")),
            "lat": lat,
            "lon": lon,
            "dbk": dbk,
            "n_split": sum(1 for s in sources.values() if s.get("also")),
        })

    wells_out.sort(key=lambda w: w["name"])
    manifest = {
        "manifest_version": 1,
        "dataset": {
            "id": "nlog",
            "name": "NLOG Netherlands well logs (FH2026 DTS set)",
            "licence": LICENCE,
            "source_url": "https://www.nlog.nl/",
            "credit": CREDIT,
            "citation": CITATION,
            "depth_units": "m",
            "crs": "EPSG:4326",
            "retrieval_date": "2026-11-19",
            "disclaimer_url": "https://www.nlog.nl/en/disclaimer",
            "note": (
                "Standard curves for a well are often split across several LAS files. "
                "completeness is computed on the union of all files. Per-well curve_map "
                "gives the primary source mnemonic; curve_sources lists the primary file "
                "and any additional files used to fill gaps; each files[].curves lists "
                "the standard properties that file contributes."
            ),
        },
        "wells": wells_out,
    }

    errors = validate_manifest(manifest)
    if errors:
        raise SystemExit("validate_manifest failed: " + "; ".join(errors))

    out = ROOT / "wells.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    loaded = load_manifest(out)
    print(f"wrote {out}  wells={len(loaded['wells'])}  validated OK")

    # size report
    total_b = sum(r["bytes"] for r in size_rows)
    total_n = sum(r["n_las"] for r in size_rows)
    print(f"TOTAL las files={total_n}  bytes={total_b}  MB={total_b/(1024*1024):.1f}")
    dts_pos = [r for r in size_rows if r["dts"] > 0]
    dts_zero = [r for r in size_rows if r["dts"] == 0]
    print(f"DTS>0: {len(dts_pos)}   DTS=0: {len(dts_zero)}")
    for b in ("first", "tno_remainder", "extra", "other"):
        rs = [r for r in size_rows if r["batch"] == b]
        if not rs:
            continue
        bsum = sum(r["bytes"] for r in rs)
        nd = sum(1 for r in rs if r["dts"] > 0)
        print(f"  {b}: {len(rs)} wells, {sum(r['n_las'] for r in rs)} files, {bsum/(1024*1024):.1f} MB, DTS>0={nd}")

    # markdown table sorted by overall desc
    print("\n======= MARKDOWN TABLE =======")
    hdr = (
        "| Well | Short | Batch | NLOG page | Operator | Spud | TD MD (m) | LAS | Size | "
        "DTS % | Overall % | WGS84 lat, lon |"
    )
    print(hdr)
    print("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in sorted(size_rows, key=lambda x: (-(x["overall"] or 0), x["name"])):
        url = f"[{r['dbk']}](https://www.nlog.nl/nlog-mapviewer/brh/{r['dbk']})" if r["dbk"] else ""
        latlon = f"{r['lat']:.5f}, {r['lon']:.5f}" if r["lat"] is not None else ""
        print(
            f"| {r['name']} | {r['short']} | {r['batch']} | {url} | {r['operator']} | "
            f"{r['spud'] or ''} | {r['td'] if r['td'] is not None else ''} | {r['n_las']} | "
            f"{r['mb']:.1f} MB | {r['dts']} | {r['overall']} | {latlon} |"
        )

    # dump size json for README authoring
    (META / "size_report.json").write_text(
        json.dumps(
            {
                "total_files": total_n,
                "total_bytes": total_b,
                "total_mb": round(total_b / (1024 * 1024), 1),
                "dts_pos": len(dts_pos),
                "dts_zero": len(dts_zero),
                "dts_zero_names": [r["name"] for r in sorted(dts_zero, key=lambda x: x["name"])],
                "wells": size_rows,
            },
            indent=2,
        )
    )
    print(f"wrote {META / 'size_report.json'}")


if __name__ == "__main__":
    main()
