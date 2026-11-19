"""Runtime manifests for the live viewer (no data is written next to the data).

* FORCE-style CSV datasets: a compact per-well index (split, median X/Y,
  cheap completeness) is built once and cached.
* Any dataset with LAS files: wells are grouped by folder; headers are read
  for coordinates.
* Coordinates prefer Sodir FactPages (matched on a normalised wellbore name),
  then FORCE X_LOC/Y_LOC (ED50 UTM 31N) or the LAS header.

Generated manifests and indexes live under
``<cache_dir>/viewer-manifests/<dataset>/`` and are reused while the input
files (size + mtime) are unchanged. Curves are never read here; the viewer
loads them on demand per well.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

REPO = Path(__file__).resolve().parents[2]
VIEWER_DIR = REPO / "viewer"
CACHE_VERSION = 2
SODIR_KEY = "sodir_factpages"
SODIR_TABLES = (
    "wellbore_exploration_all.csv",
    "wellbore_development_all.csv",
    "wellbore_other_all.csv",
)
_PANDAS_NOTE = "pandas is not installed (pip install pandas); coordinates and CSV wells are unavailable"
_LASIO_NOTE = "lasio is not installed (pip install lasio); wells are listed but log curves cannot be read"


# ---------------------------------------------------------------- helpers
def _lib(name: str):
    """Load a pandas-free wellmaplib module (names/coords) without importing the package."""
    full = f"_wmb_wellmaplib_{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, VIEWER_DIR / "wellmaplib" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _force_lib():
    """viewer/generate_force_manifest.py (needs pandas); reused for the Sodir matcher."""
    if str(VIEWER_DIR) not in sys.path:
        sys.path.insert(0, str(VIEWER_DIR))
    import generate_force_manifest as gfm

    return gfm


def manifest_cache_dir(key: str) -> Path:
    from wmb.paths import cache_dir

    return cache_dir() / "viewer-manifests" / key


def _stat_sig(path: Path, root: Path) -> list:
    st = path.stat()
    return [path.relative_to(root).as_posix(), st.st_size, st.st_mtime_ns]


def _read_cache(path: Path, fingerprint: Any) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("_fingerprint") != fingerprint or data.get("_version") != CACHE_VERSION:
        return None
    return data


def _write_cache(path: Path, fingerprint: Any, payload: dict, notes: list[str]) -> None:
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as fh:
            tmp = Path(fh.name)
            json.dump({**payload, "_fingerprint": fingerprint, "_version": CACHE_VERSION}, fh)
        os.replace(tmp, path)
        tmp = None
    except OSError as exc:
        notes.append(f"Could not write viewer cache: {exc.strerror}")
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError as exc:
                notes.append(f"Could not remove temporary viewer cache: {exc.strerror}")


# ------------------------------------------------------------------ Sodir
def sodir_folder() -> Path | None:
    from wmb.paths import get_dataset_path

    folder = get_dataset_path(SODIR_KEY)
    if not any((folder / t).is_file() for t in SODIR_TABLES):
        return None
    return folder


def sodir_signature() -> dict:
    folder = sodir_folder()
    if folder is None:
        return {}
    return {
        "root": str(folder.resolve()),
        "tables": [_stat_sig(folder / t, folder) for t in SODIR_TABLES if (folder / t).is_file()],
    }


def load_sodir(notes: list[str]) -> dict[str, dict]:
    """Alias -> Sodir wellbore record; empty (with a note) when unavailable."""
    folder = sodir_folder()
    if folder is None:
        notes.append("Sodir FactPages not available; wells are placed from other coordinates only")
        return {}
    if not _has("pandas"):
        notes.append(_PANDAS_NOTE)
        return {}
    try:
        return _force_lib().load_sodir_index(folder)
    except (SystemExit, Exception) as exc:
        notes.append(f"Sodir FactPages could not be read: {exc}")
        return {}


# ------------------------------------------------------------------ FORCE
def _force_files(root: Path) -> dict[str, Path]:
    gfm = _force_lib()
    return {s: root / info["csv"] for s, info in gfm.SPLIT_FILES.items() if (root / info["csv"]).is_file()}


def looks_like_force(root: Path) -> bool:
    """FORCE-style: semicolon CSV with a WELL column plus X_LOC or a standard curve."""
    train = root / "train.csv"
    if not train.is_file():
        return False
    try:
        with train.open(encoding="utf-8-sig", errors="replace") as fh:
            head = fh.readline().split(";")
    except OSError:
        return False
    cols = {c.strip() for c in head}
    curves = {"GR", "RDEP", "RMED", "CALI", "RHOB", "NPHI", "PEF", "DTC", "DTS", "X_LOC"}
    return "WELL" in cols and bool(cols & curves)


def index_force_csv(path: Path, split: str) -> list[dict]:
    import pandas as pd
    from wellmaplib.logs import STANDARD_CURVES, is_valid_numeric

    header = pd.read_csv(path, sep=";", nrows=0).columns.tolist()
    cols = [c for c in ["WELL", "DEPTH_MD", "X_LOC", "Y_LOC", *STANDARD_CURVES] if c in header]
    df = pd.read_csv(path, sep=";", usecols=cols, low_memory=False)
    out = []
    for well, g in df.groupby("WELL", sort=True):
        rec: dict[str, Any] = {"name": str(well), "split": split, "n_samples": int(len(g))}
        if "X_LOC" in g and "Y_LOC" in g:
            x = pd.to_numeric(g["X_LOC"], errors="coerce")
            y = pd.to_numeric(g["Y_LOC"], errors="coerce")
            ok = x.between(100000, 900000) & y.between(0, 10000000)
            if ok.any():
                rec["x"], rec["y"] = float(x[ok].median()), float(y[ok].median())
        if "DEPTH_MD" in g:
            d = pd.to_numeric(g["DEPTH_MD"], errors="coerce")
            d = d[is_valid_numeric(d) & d.abs().lt(float("inf"))]
            if not d.empty:
                rec["depth_from"], rec["depth_to"] = float(d.min()), float(d.max())
        pct = {
            c: int(round(100 * float(is_valid_numeric(g[c]).mean())))
            for c in STANDARD_CURVES if c in g
        }
        rec["curves"] = pct
        rec["overall"] = round(sum(pct.values()) / len(STANDARD_CURVES), 1)
        out.append(rec)
    return out


def force_index(root: Path, cdir: Path, notes: list[str]) -> list[dict]:
    files = _force_files(root)
    fp = {"root": str(root.resolve()), "csv": [_stat_sig(p, root) for p in files.values()]}
    cache = cdir / "well_index.json"
    hit = _read_cache(cache, fp)
    if hit is not None:
        return hit["wells"]
    wells: list[dict] = []
    for split, path in files.items():
        wells.extend(index_force_csv(path, split))
    _write_cache(cache, fp, {"wells": wells}, notes)
    return wells


def build_force_manifest(root: Path, label: str, notes: list[str]) -> dict | None:
    if not _has("pandas"):
        notes.append(_PANDAS_NOTE)
        return None
    gfm = _force_lib()
    cdir = manifest_cache_dir("force2020")
    files = _force_files(root)
    fp = {
        "root": str(root.resolve()),
        "csv": [_stat_sig(p, root) for p in files.values()],
        "sodir": sodir_signature(),
    }
    cached = _read_cache(cdir / "manifest.json", fp)
    if cached is not None:
        notes.extend(cached.get("notes", []))
        return cached["manifest"]

    index = force_index(root, cdir, notes)
    note_start = len(notes)
    sodir = load_sodir(notes)
    wells = []
    for item in index:
        name, split = item["name"], item["split"]
        xy = (item["x"], item["y"]) if "x" in item else None
        fields, _kind, _note = gfm.place_well(name, xy, sodir)
        wells.append(
            {
                "name": name,
                "dataset": "force2020",
                "split": split,
                "licence": gfm.LICENCE,
                "credit": gfm.CREDIT,
                "depth_units": "m",
                "curve_map": dict(gfm.FORCE_CURVE_MAP),
                "files": gfm.well_files(split),
                **fields,
                "completeness": {
                    "overall": item["overall"],
                    "depth_from": item.get("depth_from"),
                    "depth_to": item.get("depth_to"),
                    "n_samples": item["n_samples"],
                    "curves": item["curves"],
                },
            }
        )
    manifest = {
        "manifest_version": 1,
        "dataset": {
            "id": "force2020",
            "name": label,
            "licence": gfm.LICENCE,
            "source_url": gfm.SOURCE_URL,
            "credit": gfm.CREDIT,
            "citation": gfm.CITATION,
            "depth_units": "m",
            "crs": "EPSG:4326",
            "xy_crs": "EPSG:23031",
            "curve_map": dict(gfm.FORCE_CURVE_MAP),
            "lithology_column": gfm.LITHO_COL,
            "lithology_scheme": "force2020",
        },
        "wells": wells,
    }
    _write_cache(cdir / "manifest.json", fp, {"manifest": manifest, "notes": notes[note_start:]}, notes)
    return manifest


# -------------------------------------------------------------------- LAS
_W_LINE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*\.[^\s]*\s+(.*?)\s*:\s*(.*)$")
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_DECIMAL_ANGLE = re.compile(rf"({_NUM})\s*°?\s*([NSEW])?", re.I)
_DMS = re.compile(r"""([+-]?\d+)[\s°]+(\d+)[\s'′]+(\d+(?:\.\d+)?)[\s"″]*([NSEW])?""", re.I)

_LAT_KEYS = ("LATI", "LAT", "LATITUDE", "SLAT")
_LON_KEYS = ("LONG", "LON", "LONGITUDE", "SLON")
_X_KEYS = ("XCOORD", "X", "EASTING", "EAST", "XCOO")
_Y_KEYS = ("YCOORD", "Y", "NORTHING", "NORTH", "YCOO")


def read_las_header(path: Path, max_bytes: int = 65536) -> dict[str, tuple[str, str]]:
    """Mnemonic -> (value, description) from the ~Well section; no lasio needed."""
    out: dict[str, tuple[str, str]] = {}
    in_well = False
    with path.open("rb") as fh:
        text = fh.read(max_bytes).decode("utf-8-sig", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("~"):
            tag = s[1:2].upper()
            if tag == "W":
                in_well = True
                continue
            if in_well or tag == "A":
                break
            continue
        if in_well and s and not s.startswith("#"):
            m = _W_LINE.match(line)
            if m:
                out.setdefault(m.group(1).upper(), (m.group(2).strip(), m.group(3).strip()))
    return out


def _parse_angle(raw: str, hemispheres: str) -> float | None:
    if decimal := _DECIMAL_ANGLE.fullmatch(raw.strip()):
        degrees, hemi = decimal.groups()
        magnitude = abs(float(degrees))
    elif dms := _DMS.fullmatch(raw.strip()):
        degrees, minutes, seconds, hemi = dms.groups()
        if float(minutes) >= 60 or float(seconds) >= 60:
            return None
        magnitude = abs(float(degrees)) + float(minutes) / 60 + float(seconds) / 3600
    else:
        return None
    if hemi:
        hemi = hemi.upper()
        if hemi not in hemispheres or (degrees.startswith("-") and hemi in "NE"):
            return None
        return -magnitude if hemi in "SW" else magnitude
    return -magnitude if degrees.startswith("-") else magnitude


def _pick(header: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in header and header[k][0]:
            return header[k][0]
    return None


def las_header_coords(path: Path) -> dict | None:
    """WGS84 lat/lon (+ provenance) from LAT/LONG, or X/Y with a recognisable CRS."""
    try:
        header = read_las_header(path)
    except OSError:
        return None
    coords = _lib("coords")
    blob = " ".join(f"{k} {v[0]} {v[1]}" for k, v in header.items()).upper()
    lat_raw, lon_raw = _pick(header, _LAT_KEYS), _pick(header, _LON_KEYS)
    if lat_raw and lon_raw:
        lat, lon = _parse_angle(lat_raw, "NS"), _parse_angle(lon_raw, "EW")
        if lat is not None and lon is not None and abs(lat) <= 90 and abs(lon) <= 180 and (lat or lon):
            crs, how = "EPSG:4326", None
            if "ED50" in blob or "ED 50" in blob or re.search(r"EPSG\D{0,3}4230\b", blob):
                lat, lon = coords.ed50_ll_to_wgs84(lat, lon)
                crs, how = "EPSG:4230", "EPSG:1613"
            return _header_rec(path, lat, lon, crs, how)
    x_raw, y_raw = _pick(header, _X_KEYS), _pick(header, _Y_KEYS)
    if x_raw and y_raw:
        if re.fullmatch(_NUM, x_raw) and re.fullmatch(_NUM, y_raw):
            x, y = float(x_raw), float(y_raw)
            if not (100000 <= x <= 900000 and 0 <= y <= 10000000):
                return None
            epsg = re.search(r"EPSG\D{0,3}(\d{4,5})", blob)
            code = int(epsg.group(1)) if epsg else None
            zone_m = re.search(r"UTM\D{0,12}(\d{1,2})\s*([NS])?", blob)
            zone = int(zone_m.group(1)) if zone_m else None
            south = bool(zone_m and zone_m.group(2) == "S")
            ed50 = "ED50" in blob or "ED 50" in blob
            if code and 23028 <= code <= 23038:
                zone, ed50, south = code - 23000, True, False
            elif code and (
                32601 <= code <= 32660 or 32701 <= code <= 32760 or 25828 <= code <= 25838
            ):
                zone, ed50, south = code % 100, False, 32701 <= code <= 32760
            elif code:
                return None
            if zone and 1 <= zone <= 60 and (code or ed50 or "WGS" in blob or "ETRS" in blob):
                if ed50:
                    if south or not 28 <= zone <= 38:
                        return None
                    lat, lon = coords.utm_ed50_to_wgs84(x, y, zone=zone, northern=not south)
                    crs, how = f"EPSG:230{zone}", "EPSG:1613"
                else:
                    etrs = (code is not None and 25828 <= code <= 25838) or (not code and "ETRS" in blob)
                    if etrs and (south or not 28 <= zone <= 38):
                        return None
                    lat, lon = coords.utm_to_ll(
                        x, y, zone=zone, northern=not south, a=coords.WGS84_A,
                        f=1 / 298.257222101 if etrs else coords.WGS84_F,
                    )
                    prefix = 258 if etrs else (327 if south else 326)
                    crs, how = f"EPSG:{prefix}{zone:02d}", None
                if abs(lat) <= 90 and abs(lon) <= 180:
                    return _header_rec(path, lat, lon, crs, how, x=x, y=y)
    return None


def _header_rec(path, lat, lon, crs, transform, x=None, y=None) -> dict:
    rec = {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "crs": "EPSG:4326",
        "coord_source": "las_header",
        "coord_source_crs": crs,
        "coord_source_name": path.name,
    }
    if transform:
        rec["coord_transform"] = transform
    if x is not None:
        rec.update({"x": x, "y": y, "xy_crs": crs})
    return rec


def _las_groups(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    root_res = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in {".git", "__pycache__", ".venv"})
        for fn in sorted(filenames):
            if fn.lower().endswith(".las"):
                f = Path(dirpath) / fn
                parent = f.parent.resolve()
                groups.setdefault(parent.name if parent != root_res else f.stem, []).append(f)
    return groups


def place_from_las_headers(files: list[Path]) -> dict | None:
    for f in files:
        rec = las_header_coords(f)
        if rec:
            return rec
    return None


def build_las_manifest(key: str, root: Path, label: str, notes: list[str]) -> dict | None:
    groups = _las_groups(root)
    if not groups:
        return None
    cdir = manifest_cache_dir(key)
    fp = {
        "root": str(root.resolve()),
        "las": [_stat_sig(f, root) for fl in groups.values() for f in fl],
        "sodir": sodir_signature(),
    }
    if not _has("lasio"):
        notes.append(_LASIO_NOTE)
    cached = _read_cache(cdir / "manifest.json", fp)
    if cached is not None:
        notes.extend(cached.get("notes", []))
        return cached["manifest"]

    note_start = len(notes)
    sodir = load_sodir(notes)
    gfm = _force_lib() if (sodir and _has("pandas")) else None
    wells = []
    for name, flist in groups.items():
        rec: dict[str, Any] = {
            "name": name,
            "files": [{"path": f.relative_to(root).as_posix(), "format": "las"} for f in flist],
        }
        placed = False
        if gfm is not None:
            fields, kind, _note = gfm.place_well(name, None, sodir)
            if kind == "placed_sodir":
                rec.update(fields)
                placed = True
        if not placed:
            hdr = place_from_las_headers(flist)
            if hdr:
                rec.update(hdr)
            else:
                rec["coord_note"] = "No Sodir name match and no usable coordinates in the LAS header."
        wells.append(rec)
    manifest = {
        "manifest_version": 1,
        "dataset": {
            "id": key,
            "name": label,
            "licence": "",
            "source_url": "",
            "credit": "",
            "depth_units": "m",
            "curve_map": {},
        },
        "wells": wells,
    }
    _write_cache(cdir / "manifest.json", fp, {"manifest": manifest, "notes": notes[note_start:]}, notes)
    return manifest


# ------------------------------------------------------------ entry points
def auto_manifest(key: str, root: Path, label: str, notes: list[str]) -> dict | None:
    """Manifest for a dataset without wells.json; None when nothing can be listed."""
    if key == "force2020" and looks_like_force(root):
        return build_force_manifest(root, label, notes)
    return build_las_manifest(key, root, label, notes)


def enrich_unplaced(manifest: dict, notes: list[str]) -> None:
    """In memory only: place wells of a wells.json manifest that carry no coordinates."""
    wells = [
        w for w in manifest.get("wells") or []
        if (w.get("lat") is None or w.get("lon") is None) and (w.get("x") is None or w.get("y") is None)
    ]
    if not wells:
        return
    sodir = load_sodir(notes)
    gfm = _force_lib() if (sodir and _has("pandas")) else None
    root = Path(manifest.get("_data_root", "."))
    for w in wells:
        if gfm is not None:
            fields, kind, _note = gfm.place_well(w.get("name", ""), None, sodir)
            if kind == "placed_sodir":
                w.update(fields)
                continue
        las = [root / f["path"] for f in w.get("files") or [] if str(f.get("format", "")).lower() == "las"]
        hdr = place_from_las_headers([p for p in las if p.is_file()])
        if hdr:
            w.update(hdr)
