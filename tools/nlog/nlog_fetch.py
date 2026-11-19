#!/usr/bin/env python3
"""Download published NLOG LAS files (no login) for FH2026 testing.

Polite: custom UA, 0.6s delay, retries. Skip DLIS/LIS > 100 MB.
Never invent wells or URLs; only use API-resolved borehole ids.
"""
from __future__ import annotations

import json
import sys
import time
import hashlib
from pathlib import Path
import os as _os
_DATA = Path(_os.environ.get("FH2026_DATA", Path.home() / "fh2026-cache" / "data"))
_WORK = Path(_os.environ.get("FH2026_WORK", Path.home() / "fh2026-cache" / "work"))
from typing import Any

import requests

UA = "FH2026 research (polite, no login; published NLOG LAS only; contact via event team)"
BASE_API = "https://www.nlog.nl/nlog-mapviewer-api/rest"
DL_URL = "https://www.nlog.nl/brh-web/rest/brh/logdocument/{id}"
# alternate that also 200'd:
DL_URL_ALT = "https://www.nlog.nl/nlog-mapviewer-api/rest/brh/logdocument/{id}"
OUT = _DATA / "nlog"
META = _WORK / "nlog-meta"
DELAY = 0.7
TIMEOUT = 120

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept": "*/*"})

# Well short names read from TNO 2021 Fig. 1A (methods PDF page 2).
# Uncertain readings are marked; all must be resolved via suggest API before use.
TNO_2021_MAP_NAMES = [
    "AMR-12",
    "TID-702",
    "VRS-08",
    "HPS-01",
    "BLD-01",
    "LTG-01",
    "GSB-01",
    "COV-41",
    "COV-40",
    "SLD-06-S1",
    "GRO-01",
    "MDZ-01",
    "MID-302",
    "WBMS-01",
    "WBMS-02",
    "WBMS-03",
    "P15-RIJN-A-09-S1",
    "Q11-03",
    "Q14-03",
    "P15-F-02",
    "P15-14",
    "Q13-A-01",
    "Q13-A-02",
    "Q13-A-05-S1",
    "P18-A-02",
    "MON-04",
    "NLW-GT-01",
    "MSG-03-S1",
    "CAP-01",
    "GAG-06-S1",
    "IJS-64-S2",  # map label may be IJS/LJS/LIS-64-S2; try alternates
    "BLK-01",
    "DRT-01",
    "WED-03",
    "PRW-01",
    "WWN-03",
    "NMD-01",
    "NMD-03",
    "WWS-02",
    "RTD-02",
    "HBV-01",
]

# Alternates for labels that were slightly ambiguous on the map
NAME_ALTERNATES = {
    "IJS-64-S2": ["LJS-64-S2", "LIS-64-S2", "LUS-64-S2", "IJS-64-S2", "IJSS-64-S2"],
    "P15-RIJN-A-09-S1": [
        "P15-RIJN-A-09-S1",
        "P15-RIJN-A-09-SIDETRACK1",
        "P15-RIJN-A-09",
        "RIJN-A-09-S1",
        "P15-RIJNA-09-S1",
    ],
    "MSG-03-S1": ["MSG-03-S1", "MSG-03-S", "MSG-03"],
    "SLD-06-S1": ["SLD-06-S1", "SLD-06-SIDETRACK1", "SLD-06"],
    "Q13-A-05-S1": ["Q13-A-05-S1", "Q13-A-05-SIDETRACK1", "Q13-A-05"],
    "GAG-06-S1": ["GAG-06-S1", "GAG-06-SIDETRACK1", "GAG-06"],
    "P18-A-02": ["P18-A-02", "P15-A-02"],
    "TID-702": ["TID-702", "TID-7-02", "TID702"],
    "MID-302": ["MID-302", "MID-3-02", "MID302"],
}


def sleep():
    time.sleep(DELAY)


def get(url: str, **kw) -> requests.Response:
    last = None
    for i in range(4):
        try:
            r = SESSION.get(url, timeout=kw.pop("timeout", TIMEOUT) if False else TIMEOUT, **kw)
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def post(url: str, **kw) -> requests.Response:
    last = None
    for i in range(4):
        try:
            r = SESSION.post(url, timeout=TIMEOUT, **kw)
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def suggest(query: str) -> list[dict]:
    sleep()
    r = get(f"{BASE_API}/search/suggest/brh/{requests.utils.quote(query, safe='')}")
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return data
    return data if isinstance(data, list) else []


def details(borehole_dbk: int | str) -> dict | None:
    sleep()
    r = post(
        f"{BASE_API}/brh/details",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        data=json.dumps([str(borehole_dbk)]),
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        return data[0]
    return None


def logdocuments(borehole_dbk: int | str) -> list[dict]:
    sleep()
    r = post(
        f"{BASE_API}/brh/logdocuments",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        data=str(int(borehole_dbk)),
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def pick_suggestion(query: str, hits: list[dict]) -> dict | None:
    """Prefer exact short-name or title match; else unique prefix."""
    q = query.strip().upper().replace(" ", "")
    if not hits:
        return None
    scored = []
    for h in hits:
        title = (h.get("title") or "").upper()
        # titles look like "BLESDIJKE-01 (BLD-01)"
        short = ""
        if "(" in title and title.endswith(")"):
            short = title[title.rfind("(") + 1 : -1].replace(" ", "")
        name = title.split("(")[0].strip().replace(" ", "")
        if short == q or name == q or title.replace(" ", "") == q:
            return h
        if short.startswith(q) or name.startswith(q) or q in (short, name):
            scored.append((0, h))
        elif q in title.replace(" ", ""):
            scored.append((1, h))
    if len(hits) == 1:
        return hits[0]
    if len(scored) == 1:
        return scored[0][1]
    exactish = [h for s, h in scored if s == 0]
    if len(exactish) == 1:
        return exactish[0]
    return None


def resolve_name(name: str) -> dict:
    """Return {query, status, chosen, hits, alternates_tried}."""
    tried = []
    queries = [name] + [a for a in NAME_ALTERNATES.get(name, []) if a != name]
    all_hits = []
    for q in queries:
        tried.append(q)
        try:
            hits = suggest(q)
        except Exception as e:
            return {"query": name, "status": "error", "error": str(e), "tried": tried}
        all_hits.extend(hits)
        chosen = pick_suggestion(q, hits)
        if chosen:
            return {
                "query": name,
                "status": "resolved",
                "chosen": chosen,
                "tried": tried,
                "n_hits": len(hits),
                "hits": hits[:8],
            }
    # last resort: if a later alternate's hits uniquely contain a close title
    # don't guess among many
    return {
        "query": name,
        "status": "unresolved",
        "tried": tried,
        "hits": all_hits[:15],
    }


def is_las(f: dict) -> bool:
    typ = (f.get("fileTypeCode") or "").upper()
    name = (f.get("fileName") or "").lower()
    group = f.get("documentGroup") or ""
    if typ == "LAS":
        return True
    if group in ("LAS_AND_ASCII_FILES", "COMPOSITE_LOG_FILE") and (
        name.endswith(".las") or name.endswith(".a1") or name.endswith(".a2")
        or name.endswith(".a3") or name.endswith(".a4") or name.endswith(".a5")
        or name.endswith(".a6") or name.endswith(".a7") or name.endswith(".a8")
        or name.endswith(".a9")
    ):
        return True
    return False


def has_group(f: dict, code: str) -> bool:
    g = f.get("groups") or []
    return code.upper() in [str(x).upper() for x in g]


def file_size(f: dict) -> int:
    try:
        return int(f.get("fileSize") or 0)
    except (TypeError, ValueError):
        return 0


def select_las_for_download(files: list[dict], mode: str = "all_las") -> list[dict]:
    """mode=all_las: every LAS (for the two confirmed wells).
    mode=priority: composite + GSON/sonic + density/neutron/resistivity LAS, skip huge files.
    Skip DLIS/LIS always here (we only want LAS). Skip files > 100 MB.
    """
    out = []
    for f in files:
        sz = file_size(f)
        typ = (f.get("fileTypeCode") or "").upper()
        if typ in ("DLIS", "LIS"):
            continue
        if sz > 100_000_000:
            continue
        if not is_las(f):
            continue
        if mode == "all_las":
            out.append(f)
        else:
            name = (f.get("fileName") or "").lower()
            group = f.get("documentGroup") or ""
            groups = f.get("groups") or []
            keep = False
            if group == "COMPOSITE_LOG_FILE" or "_comp.las" in name or name.endswith("_comp.las"):
                keep = True
            if has_group(f, "GSON"):
                keep = True
            if any(k in name for k in ("sonic", "dts", "dtsm", "dtco", "ac-array", "xmac", "aniso", "slow")):
                keep = True
            if has_group(f, "GDEN") or has_group(f, "GNEU") or has_group(f, "GRES"):
                keep = True
            if keep:
                out.append(f)
    # unique by id
    seen = set()
    uniq = []
    for f in out:
        i = f.get("documentBfileDbk")
        if i in seen:
            continue
        seen.add(i)
        uniq.append(f)
    return uniq


def download_file(f: dict, dest_dir: Path) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    fid = f["documentBfileDbk"]
    name = f.get("fileName") or f"file_{fid}"
    # sanitize
    name = Path(name).name
    dest = dest_dir / name
    rec = {
        "documentBfileDbk": fid,
        "fileName": name,
        "expectedSize": file_size(f),
        "path": str(dest),
    }
    if dest.exists() and dest.stat().st_size > 0:
        if rec["expectedSize"] and dest.stat().st_size == rec["expectedSize"]:
            rec["status"] = "exists"
            rec["bytes"] = dest.stat().st_size
            return rec
        # if size mismatches, re-download
    sleep()
    url = DL_URL.format(id=fid)
    r = get(url, stream=True)
    ctype = (r.headers.get("Content-Type") or "").lower()
    disp = r.headers.get("Content-Disposition") or ""
    if r.status_code in (401, 403):
        rec["status"] = "auth_required"
        rec["http"] = r.status_code
        rec["url"] = url
        return rec
    if r.status_code != 200:
        rec["status"] = "http_error"
        rec["http"] = r.status_code
        rec["url"] = url
        rec["body_start"] = r.text[:300]
        return rec
    if "text/html" in ctype and "login" in r.text[:2000].lower():
        rec["status"] = "login_page"
        rec["url"] = url
        return rec
    # write
    tmp = dest.with_suffix(dest.suffix + ".part")
    n = 0
    sha = hashlib.sha256()
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            fh.write(chunk)
            sha.update(chunk)
            n += len(chunk)
    tmp.rename(dest)
    rec["status"] = "downloaded"
    rec["bytes"] = n
    rec["sha256"] = sha.hexdigest()
    rec["url"] = url
    rec["content_type"] = ctype
    rec["content_disposition"] = disp
    # sanity: LAS should be mostly text
    if n < 50:
        rec["status"] = "too_small"
    return rec


def process_well(borehole_dbk: int | str, folder_name: str | None = None, las_mode: str = "all_las") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    det = details(borehole_dbk)
    if not det:
        return {"status": "no_details", "boreholeDbk": borehole_dbk}
    well_name = folder_name or det.get("boreholeName") or det.get("shortName") or str(borehole_dbk)
    # sanitize folder
    well_name = well_name.replace("/", "-").replace(" ", "_")
    wdir = OUT / well_name
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "nlog_details.json").write_text(json.dumps(det, indent=2))
    files = logdocuments(borehole_dbk)
    (wdir / "nlog_logdocuments.json").write_text(json.dumps(files, indent=2))
    selected = select_las_for_download(files, mode=las_mode)
    skipped_big = [
        {"fileName": f.get("fileName"), "fileTypeCode": f.get("fileTypeCode"), "fileSize": file_size(f)}
        for f in files
        if file_size(f) > 100_000_000
    ]
    downloads = []
    for f in selected:
        rec = download_file(f, wdir)
        downloads.append(rec)
        print(f"  {rec['status']:12} {rec.get('bytes', rec.get('expectedSize')):>12}  {rec['fileName']}", flush=True)
        if rec.get("status") in ("auth_required", "login_page"):
            return {
                "status": "stopped_login",
                "well": well_name,
                "details": det,
                "downloads": downloads,
                "selected": selected,
            }
    summary = {
        "status": "ok",
        "well": well_name,
        "boreholeDbk": det.get("boreholeDbk"),
        "shortName": det.get("shortName"),
        "n_inventory": len(files),
        "n_selected": len(selected),
        "n_downloaded_ok": sum(1 for d in downloads if d["status"] in ("downloaded", "exists")),
        "bytes": sum(d.get("bytes") or 0 for d in downloads if d.get("status") in ("downloaded", "exists")),
        "skipped_big": skipped_big,
        "downloads": downloads,
        "details_keys": {
            "latitudeWgs84": det.get("latitudeWgs84"),
            "longitudeWgs84": det.get("longitudeWgs84"),
            "latitude": det.get("latitude"),
            "longitude": det.get("longitude"),
            "coordSystemCode": det.get("coordSystemCode"),
            "clientOrgName": det.get("clientOrgName"),
            "startDate": det.get("startDate"),
            "endAhDepthInMeters": det.get("endAhDepthInMeters"),
            "tvdInMeters": det.get("tvdInMeters"),
        },
    }
    (wdir / "download_manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "resolve":
        results = []
        for name in TNO_2021_MAP_NAMES:
            print(f"resolve {name} ...", flush=True)
            r = resolve_name(name)
            results.append(r)
            st = r["status"]
            if st == "resolved":
                ch = r["chosen"]
                print(f"  OK  id={ch.get('objectId')}  {ch.get('title')}", flush=True)
            else:
                titles = [h.get("title") for h in r.get("hits") or []]
                print(f"  {st}  hits={titles[:6]}", flush=True)
        META.mkdir(parents=True, exist_ok=True)
        (META / "resolve.json").write_text(json.dumps(results, indent=2))
        print("wrote resolve.json")
    elif cmd == "download-two":
        # The two confirmed wells
        wells = [
            ("106520549", "BLESDIJKE-01", "all_las"),
            ("106532227", "D12-A-02", "all_las"),
        ]
        allsum = []
        for dbk, name, mode in wells:
            print(f"==== {name} ({dbk}) ====", flush=True)
            s = process_well(dbk, folder_name=name, las_mode=mode)
            allsum.append(s)
            if s.get("status") == "stopped_login":
                print("STOP: login required")
                break
        (META / "two_wells.json").write_text(json.dumps(allsum, indent=2))
        print("done two wells")
    elif cmd == "download-well":
        dbk, name, mode = sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "priority"
        print(f"==== {name} ({dbk}) mode={mode} ====", flush=True)
        s = process_well(dbk, folder_name=name, las_mode=mode)
        print(json.dumps({k: s[k] for k in s if k not in ("downloads", "details")}, indent=2))
        (META / f"well_{name}.json").write_text(json.dumps(s, indent=2))
        if s.get("status") == "stopped_login":
            sys.exit(2)
    else:
        print("usage: resolve | download-two | download-well ID NAME [mode]")
