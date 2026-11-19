"""Local well-map viewer: data-free HTML + a tiny HTTP server.

    python -m wmb.viewer --launch

Reads data_paths.local.yaml, serves viewer/wellmap.html, and exposes a
read-only API for whichever datasets are present. Missing datasets are
listed as not available; they do not crash the viewer.

The baked/offline builder (viewer/build_wellmap.py) is unchanged and still
embeds curve data in a gitignored HTML file for sharing without a server.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

# Repo root: src/wmb/viewer.py -> parents[2]
REPO = Path(__file__).resolve().parents[2]
VIEWER_DIR = REPO / "viewer"
VENDOR_DIR = VIEWER_DIR / "vendor"
HTML_PATH = VIEWER_DIR / "wellmap.html"

DATASET_LABELS = {
    "force2020": "FORCE 2020",
    "sodir_factpages": "Sodir FactPages",
    "nlog_dts": "NLOG (DTS wells)",
    "kansas_kgs_dts": "Kansas KGS (DTS wells)",
    "volve": "Volve",
}

# Datasets that are supporting tables, not log wells on the map.
SUPPORT_KEYS = {"sodir_factpages"}


def _ensure_src_on_path() -> None:
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    env = os.environ.get("PYTHONPATH", "")
    if src not in env.split(os.pathsep):
        os.environ["PYTHONPATH"] = src + (os.pathsep + env if env else "")


def _bootstrap_yaml() -> None:
    try:
        import yaml  # noqa: F401
        return
    except ImportError:
        pass
    print("PyYAML is missing. One-line fix:  pip install pyyaml")
    ans = input("Create a .venv in this repo and install PyYAML there now? [y/N] ").strip().lower()
    if ans not in {"y", "yes"}:
        raise SystemExit(1)
    venv_dir = REPO / ".venv"
    import subprocess
    import venv

    print(f"Creating {venv_dir} …")
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    py = venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    subprocess.check_call([str(py), "-m", "pip", "install", "pyyaml"])
    _ensure_src_on_path()
    os.execv(str(py), [str(py), "-m", "wmb.viewer", *sys.argv[1:]])


def _prompt_data_root() -> Path:
    print("data_paths.local.yaml is missing. Please choose the shared team data folder")
    print("(it should contain subfolders such as FORCE2020, Volve, NLOG-DTS, …).")
    picked = ""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.wm_attributes("-topmost", 1)
        except Exception:
            pass
        picked = filedialog.askdirectory(title="Select the shared team data folder") or ""
        root.destroy()
    except Exception:
        picked = ""
    if not picked:
        picked = input("Path to the shared team data folder: ").strip().strip('"').strip("'")
    path = Path(picked).expanduser()
    if not picked or not path.is_dir():
        raise SystemExit(f"That folder was not found: {path}")
    return path


def ensure_config(interactive: bool) -> None:
    _bootstrap_yaml()
    from wmb.paths import DataPathsError, config_path, load_config, write_local_paths

    try:
        load_config()
        return
    except DataPathsError:
        if not interactive:
            raise
    root = _prompt_data_root()
    dest = config_path()
    write_local_paths(root, dest=dest)
    print(f"Wrote {dest}")


def discover_for_viewer() -> list[dict[str, Any]]:
    """Describe each known dataset for the API and the UI (no crash if missing)."""
    from wmb.paths import dataset_status

    rows = []
    for st in dataset_status():
        item: dict[str, Any] = {
            "key": st.key,
            "name": DATASET_LABELS.get(st.key, st.key),
            "path": str(st.path) if st.exists else None,
            "status": "available" if st.found else "not_available",
            "n_las": st.n_las,
            "n_csv": st.n_csv,
            "has_manifest": st.has_manifest,
            "role": "support" if st.key in SUPPORT_KEYS else "wells",
            "note": "",
        }
        if not st.exists:
            item["note"] = "not available"
        elif not st.found:
            item["note"] = "folder is present but no LAS or CSV files were found"
        elif st.key in SUPPORT_KEYS:
            item["note"] = "wellhead coordinates and tops (not plotted as a log dataset)"
        elif not st.has_manifest:
            item["note"] = "no wells.json yet; a temporary manifest will be used if possible"
        rows.append(item)
    return rows


def _load_json_manifest(root: Path) -> dict[str, Any] | None:
    for name in ("wells.json", "wells.manifest.json"):
        path = root / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_data_root"] = str(root)
            data["_manifest_path"] = str(path)
            data["_generated"] = False
            return data
    return None


def load_dataset_manifest(
    key: str, path: Path, label: str, notes: list[str] | None = None
) -> dict[str, Any] | None:
    """wells.json when present, else an auto-built (cached) manifest; never raises for missing deps."""
    _ensure_src_on_path()
    from wmb import viewer_manifest as vm

    notes = notes if notes is not None else []
    found = _load_json_manifest(path)
    if found is not None:
        vm.enrich_unplaced(found, notes)
        return found
    try:
        manifest = vm.auto_manifest(key, path, label, notes)
    except Exception as exc:
        notes.append(f"could not build a manifest: {type(exc).__name__}: {exc}")
        return None
    if manifest is None:
        return None
    manifest = dict(manifest)
    manifest["_data_root"] = str(path)
    manifest["_generated"] = True
    return manifest


def _maybe_place(well: dict) -> None:
    if well.get("lat") is not None and well.get("lon") is not None:
        return
    try:
        sys.path.insert(0, str(VIEWER_DIR))
        from build_wellmap import maybe_place_from_xy

        maybe_place_from_xy(well)
    except Exception:
        return


def wells_from_manifest(manifest: dict[str, Any], key: str) -> tuple[list[dict], list[dict]]:
    dataset = manifest.get("dataset") or {}
    ds_id = dataset.get("id") or key
    wells_out: list[dict] = []
    unplaced: list[dict] = []
    for well in manifest.get("wells") or []:
        _maybe_place(well)
        comp = well.get("completeness") or {}
        curves = comp.get("curves") or {}
        if not curves and isinstance(comp, dict):
            curves = {k: v for k, v in comp.items() if k not in {"overall", "curves"}}
        overall = comp.get("overall")
        if overall is not None:
            try:
                overall = float(overall)
                if overall <= 1 and not (key == "force2020" and manifest.get("_generated")):
                    overall *= 100
                overall = round(overall, 1)
            except (TypeError, ValueError):
                overall = None
        has_dts = False
        try:
            has_dts = float((curves or {}).get("DTS") or 0) > 0
        except (TypeError, ValueError):
            has_dts = False
        packed = {
            "id": f"{ds_id}::{well.get('name')}",
            "name": well.get("name"),
            "dataset": ds_id,
            "dataset_key": key,
            "split": well.get("split") or "",
            "lat": well.get("lat"),
            "lon": well.get("lon"),
            "coord_source": well.get("coord_source"),
            "depth_units": well.get("depth_units") or dataset.get("depth_units") or "m",
            "licence": well.get("licence") or dataset.get("licence") or "",
            "credit": well.get("credit") or dataset.get("credit") or "",
            "lithology_scheme": dataset.get("lithology_scheme") or "force2020",
            "overall": overall,
            "curves": curves,
            "has_dts": has_dts,
            "d": [],
            "c": {},
            "lith": None,
            "logs_pending": True,
        }
        wells_out.append(packed)
        if well.get("lat") is None or well.get("lon") is None:
            unplaced.append(
                {
                    "name": well.get("name"),
                    "dataset": ds_id,
                    "reason": well.get("coord_note") or "no lat/lon or convertible x/y",
                }
            )
    return wells_out, unplaced


def public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Drop absolute paths before sending a manifest to the browser."""
    out = {
        "manifest_version": manifest.get("manifest_version", 1),
        "dataset": dict(manifest.get("dataset") or {}),
        "wells": [],
        "generated": bool(manifest.get("_generated")),
    }
    for well in manifest.get("wells") or []:
        item = {k: v for k, v in well.items() if k not in {"abs_path"}}
        files = []
        for f in item.get("files") or []:
            files.append({k: v for k, v in f.items() if k not in {"abs_path"}})
        item["files"] = files
        out["wells"].append(item)
    return out


def safe_under(root: Path, rel: str) -> Path | None:
    """Resolve rel under root; None if missing or a path-traversal attempt."""
    if not rel or rel.startswith("/") or rel.startswith("\\"):
        return None
    root = root.resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def benchmark_run_time(row: dict) -> datetime:
    value = row.get("run_at")
    if isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def collect_benchmark_rows(root: Path | None = None) -> list[dict[str, Any]]:
    """Read-only twin of benchmark.scoreboard.collect(): both results layouts, unreadable files skipped."""
    from wmb import paths

    root = Path(root) if root is not None else paths.results_root()
    rows: dict[tuple, dict] = {}
    # rglob so an optional test/ folder (or other future layout) is still found; patterns used to be 3-4 levels.
    for p in sorted(root.rglob("scores.json")):
        try:
            found = safe_under(root, str(p.relative_to(root)))
            if found is None:
                continue
            r = json.loads(found.read_text(encoding="utf-8"), parse_constant=lambda _: None)
            if not isinstance(r, dict):
                continue
            r.setdefault("scenario", "missing_curve")
            test = r.get("test") if isinstance(r.get("test"), str) else ""
            key = (r["contender"], r["dataset"], r["target"], r["scenario"], test)
            if all(isinstance(part, str) and part for part in key[:4]) and (
                key not in rows or benchmark_run_time(r) >= benchmark_run_time(rows[key])
            ):
                if test:
                    rel = p.relative_to(root).parts
                    r["prediction_test"] = test if rel == (*key[:4], test, "scores.json") else ""
                rows[key] = r
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(rows.values(), key=lambda r: (
        r["dataset"], r.get("test") or "", r["target"], r["scenario"],
        r["MAE"] if isinstance(r.get("MAE"), (int, float)) else float("inf"),
    ))


def mark_stale_rows(rows: list[dict[str, Any]], current: dict[str, str | None] | None = None) -> list[dict[str, Any]]:
    """Add stale=True/False to rows that carry a model_hash; stale means the model code changed since the run.

    `current` maps contender -> present fingerprint (default: computed from the registry, None when it cannot be built).
    """
    cache: dict[str, str | None] = dict(current) if current is not None else {}
    api = None
    found = None

    def now(name: str) -> str | None:
        nonlocal api, found
        if name not in cache:
            cache[name] = None
            if current is None:
                try:
                    from wmb import predict_cache
                    if found is None:
                        api = _predict_api()
                        found, _ = api._registry()
                    if name in found:
                        cache[name] = api._inspect_model(lambda: predict_cache.fingerprint(found[name]()))
                except (Exception, SystemExit) as exc:
                    logging.getLogger(__name__).warning("Could not fingerprint contender %s: %s", name, exc)
        return cache[name]

    out = []
    for r in rows:
        stored = r.get("model_hash")
        cur = now(r["contender"]) if stored else None
        out.append({**r, "stale": bool(stored and cur and cur != stored)} if stored else r)
    return out


_SEGMENT_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def load_benchmark_prediction(contender: str, dataset: str, target: str, scenario: str, well: str,
                              root: Path | None = None, test: str = "") -> dict[str, Any]:
    """One well from results/<contender>/<dataset>/<target>/<scenario>[/<test>]/predictions.json or predictions.npz.

    Looks in the per-test folder first (when `test` is given), then the older layout without it. In a folder with
    both formats the newer file wins; both return the same JSON (see wmb.prediction_files).
    FileNotFoundError if neither exists.
    """
    from wmb import paths

    parts = [contender, dataset, target, scenario]
    if not well:
        raise ValueError("well is required")
    if any(not _SEGMENT_OK.fullmatch(p) or ".." in p for p in parts) or (
            test and (not _SEGMENT_OK.fullmatch(test) or ".." in test)):
        raise ValueError("contender, dataset, target, scenario and test must be plain names")
    root = Path(root) if root is not None else paths.results_root()

    def prediction_file(folder: list[str]) -> Path | None:
        found = [f for f in (safe_under(root, "/".join([*folder, n])) for n in ("predictions.json", "predictions.npz")) if f]
        return max(found, key=lambda f: f.stat().st_mtime) if found else None

    found = None
    if test:
        found = prediction_file([*parts, test])
        if found is None and safe_under(root, "/".join([*parts, test, "scores.json"])):
            raise FileNotFoundError("no predictions.json or predictions.npz for this test; re-run benchmark.run to write it")
    if found is None:
        found = prediction_file(parts)
    if found is None:
        raise FileNotFoundError("no predictions.json or predictions.npz for this run; re-run benchmark.run to write it")
    if found.suffix == ".npz":  # numpy only when needed: the viewer itself runs without it
        from wmb import prediction_files

        return prediction_files.read_npz_well(found, well)
    data = json.loads(found.read_text(encoding="utf-8"))
    if well == "*":
        return {"wells": [w.get("key") or w.get("well_id") for w in data.get("wells", [])]}
    for w in data.get("wells", []):
        if well in (w.get("key"), w.get("well_id")):
            return {k: data.get(k) for k in ("contender", "dataset", "target", "scenario")} | {"well": w}
    raise FileNotFoundError(f"well {well} not in predictions.json")


class ViewerState:
    def __init__(self) -> None:
        self.availability = discover_for_viewer()
        self._manifests: dict[str, dict[str, Any] | None] = {}
        self._csv_frames: dict[str, Any] = {}
        self.notes: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def manifest_for(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if key in self._manifests:
                return self._manifests[key]
        row = next((r for r in self.availability if r["key"] == key), None)
        if not row or row["status"] != "available" or row["role"] != "wells" or not row["path"]:
            with self._lock:
                self._manifests[key] = None
            return None
        notes: list[str] = []
        manifest = load_dataset_manifest(key, Path(row["path"]), row["name"], notes)
        with self._lock:
            self._manifests[key] = manifest
            self.notes[key] = list(dict.fromkeys(notes))
        return manifest

    def map_payload(self) -> dict[str, Any]:
        datasets = []
        wells: list[dict] = []
        unplaced: list[dict] = []
        lithology: dict[str, Any] = {}
        for row in self.availability:
            info = {
                "id": row["key"],
                "key": row["key"],
                "name": row["name"],
                "status": row["status"],
                "role": row["role"],
                "note": row["note"],
                "has_manifest": row["has_manifest"],
                "n_las": row["n_las"],
                "n_csv": row["n_csv"],
            }
            if row["status"] == "available" and row["role"] == "wells":
                manifest = self.manifest_for(row["key"])
                notes = "; ".join(self.notes.get(row["key"], []))
                if manifest is None:
                    info["note"] = notes or row["note"] or "no wells.json, and no LAS files to scan"
                    info["status"] = "available_no_wells"
                else:
                    if notes:
                        info["note"] = notes
                    elif manifest.get("_generated"):
                        info["note"] = "wells generated at runtime (cached)"
                    ds = manifest.get("dataset") or {}
                    info["id"] = ds.get("id") or row["key"]
                    info["licence"] = ds.get("licence") or ""
                    info["credit"] = ds.get("credit") or ""
                    info["generated_manifest"] = bool(manifest.get("_generated"))
                    w, u = wells_from_manifest(manifest, row["key"])
                    wells.extend(w)
                    unplaced.extend(u)
                    scheme = ds.get("lithology_scheme") or "force2020"
                    try:
                        sys.path.insert(0, str(VIEWER_DIR))
                        from wellmaplib.lithology import scheme_payload

                        lithology[scheme] = scheme_payload(scheme)
                    except Exception:
                        pass
            datasets.append(info)
        coast = {"type": "FeatureCollection", "features": []}
        coast_path = VENDOR_DIR / "ncs_coast.geojson"
        if coast_path.is_file():
            coast = json.loads(coast_path.read_text(encoding="utf-8"))
        return {
            "generated": "live",
            "datasets": datasets,
            "wells": wells,
            "unplaced": unplaced,
            "lithology": lithology,
            "coast": coast,
            "live": True,
        }

    def pack_logs(self, key: str, well_name: str, max_points: int = 20000) -> dict[str, Any]:
        manifest = self.manifest_for(key)
        if manifest is None:
            raise FileNotFoundError(f"No manifest for dataset {key}")
        well = next((w for w in manifest["wells"] if w.get("name") == well_name), None)
        if well is None:
            raise FileNotFoundError(f"Well {well_name!r} not in {key}")
        sys.path.insert(0, str(VIEWER_DIR))
        from build_wellmap import load_csv_groups, load_well_frame, pack_well

        dataset = manifest["dataset"]
        data_root = Path(manifest["_data_root"])
        split = well.get("split") if key == "force2020" and manifest.get("_generated") else None
        cache_key = f"{key}:{split}" if split else key
        with self._lock:
            frames = self._csv_frames.get(cache_key)
        if frames is None:
            sub = manifest
            if split:
                sub = {**manifest, "wells": [w for w in manifest["wells"] if w.get("split") == split]}
            frames = load_csv_groups(sub)
            with self._lock:
                self._csv_frames[cache_key] = frames
        frame = load_well_frame(well, dataset, data_root, frames)
        packed = pack_well(well, dataset, frame, max_points)
        return {"d": packed.get("d") or [], "c": packed.get("c") or {}, "lith": packed.get("lith"),
                "overall": packed.get("overall"), "curves": packed.get("curves") or {},
                "has_dts": packed.get("has_dts", False)}


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
MAX_BODY_BYTES = 16_384
BODY_TIMEOUT_S = 5


def is_local_client(client_address) -> bool:
    host = str(client_address[0]).split("%")[0]
    return host in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


def _local_http_authority(value: str) -> tuple[str, int] | None:
    try:
        parsed = urlparse(value)
        if (parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS
                and parsed.username is None and not any((parsed.path, parsed.params, parsed.query, parsed.fragment))):
            return parsed.hostname, parsed.port if parsed.port is not None else 80
    except ValueError:
        pass
    return None


def _predict_api():
    """benchmark.predict_api, imported lazily (the viewer itself must start without the benchmark package on the path)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    _ensure_src_on_path()
    from benchmark import predict_api

    return predict_api


def _send(handler: BaseHTTPRequestHandler, code: int, body: bytes, content_type: str) -> None:
    handler.send_response(code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _send_json(handler: BaseHTTPRequestHandler, code: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")
    _send(handler, code, body, "application/json; charset=utf-8")


def _guess_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".geojson": "application/geo+json; charset=utf-8",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
    }.get(suffix, "application/octet-stream")


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "wmb-viewer/0.1"
    state: ViewerState

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            self._route(path, parse_qs(parsed.query))
        except FileNotFoundError as exc:
            _send_json(self, 404, {"error": str(exc)})
        except ValueError as exc:
            _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            _send_json(self, 500, {"error": f"{type(exc).__name__}: {exc}"})

    def _local_only(self) -> bool:
        """Compute endpoints: loopback clients only, and no cross-site browser origins. Sends 403 and returns False otherwise."""
        authority = _local_http_authority("http://" + self.headers.get("Host", ""))
        origin = self.headers.get("Origin")
        if (not is_local_client(self.client_address) or authority is None
                or (origin is not None and _local_http_authority(origin) != authority)):
            _send_json(self, 403, {"error": "this endpoint is only available from localhost"})
            return False
        return True

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            if path != "/api/predict":
                _send_json(self, 404, {"error": "not found"})
                return
            if not self._local_only():
                return
            if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
                raise ValueError("Content-Type must be application/json")
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("request body must be a small JSON object")
            original_timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(BODY_TIMEOUT_S)
                raw = self.rfile.read(length)
            except TimeoutError:
                _send_json(self, 408, {"error": "timed out reading the request body"})
                return
            finally:
                self.connection.settimeout(original_timeout)
            try:
                body = json.loads(raw)
            except ValueError as exc:
                raise ValueError(f"invalid JSON: {exc}") from exc
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            for k in ("dataset", "well", "target"):
                if not isinstance(body.get(k), str) or not body[k]:
                    raise ValueError(f"{k} is required")
            force = body.get("force", False)
            if not isinstance(force, bool):
                raise ValueError("force must be a boolean")
            _send_json(self, 200, _predict_api().predict_well(body["dataset"], body["well"], body["target"], body.get("models", []),
                                                           force=force))
        except FileNotFoundError as exc:
            _send_json(self, 404, {"error": str(exc)})
        except ValueError as exc:
            _send_json(self, 400, {"error": str(exc)})
        except Exception as exc:
            _send_json(self, 500, {"error": f"{type(exc).__name__}: {exc}"})

    def _route(self, path: str, query: dict[str, list[str]]) -> None:
        if path in {"/", "/wellmap.html", "/index.html"}:
            if not HTML_PATH.is_file():
                raise FileNotFoundError(f"Missing {HTML_PATH}")
            _send(self, 200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/plot_utils.js":
            helper = VIEWER_DIR / "plot_utils.js"
            if not helper.is_file():
                raise FileNotFoundError(f"Missing {helper}")
            _send(self, 200, helper.read_bytes(), "application/javascript; charset=utf-8")
            return
        if path in ("/benchmark.js", "/benchmark_quality.js", "/prediction.js", "/well_filters.js", "/plot_style.js"):
            helper = VIEWER_DIR / path.lstrip("/")
            if not helper.is_file():
                raise FileNotFoundError(f"Missing {helper}")
            _send(self, 200, helper.read_bytes(), "application/javascript; charset=utf-8")
            return
        if path == "/api/predict/models":
            if not self._local_only():
                return
            q = {k: (query.get(k) or [""])[0] for k in ("dataset", "well", "target", "metric", "rank_scenario")}
            predict_api = _predict_api()
            out = predict_api.list_models(q["dataset"] or None, q["well"] or None, q["target"] or "RHOB")
            names = {m["name"] for m in out["models"] if m["eligible"]}
            out["best"] = predict_api.best_model(collect_benchmark_rows(), q["dataset"], out["target"], q["rank_scenario"] or "missing_curve",
                                                 q["metric"] or "MAE", names) if q["dataset"] else None
            _send_json(self, 200, out)
            return
        if path == "/api/benchmark/runs":
            _send_json(self, 200, {"rows": mark_stale_rows(collect_benchmark_rows())})
            return
        if path == "/api/benchmark/prediction":
            q = {k: (query.get(k) or [""])[0] for k in ("contender", "dataset", "target", "scenario", "well", "test")}
            _send_json(self, 200, load_benchmark_prediction(**q))
            return
        if path.startswith("/vendor/"):
            rel = path[len("/vendor/") :]
            found = safe_under(VENDOR_DIR, rel)
            if found is None:
                _send_json(self, 404, {"error": "not found"})
                return
            _send(self, 200, found.read_bytes(), _guess_type(found))
            return
        if path == "/api/datasets":
            datasets = []
            for row in self.state.availability:
                item = dict(row)
                if row["status"] == "available" and row["role"] == "wells":
                    manifest = self.state.manifest_for(row["key"])
                    if manifest is not None:
                        item["manifest"] = public_manifest(manifest)
                        item["n_wells"] = len(manifest.get("wells") or [])
                datasets.append(item)
            from wmb.paths import data_root

            _send_json(self, 200, {"data_root": str(data_root()), "datasets": datasets})
            return
        if path == "/api/map":
            _send_json(self, 200, self.state.map_payload())
            return
        if path == "/api/well":
            key = (query.get("dataset") or [""])[0]
            name = (query.get("well") or [""])[0]
            if not key or not name:
                raise ValueError("dataset and well query parameters are required")
            # Accept either canonical key or dataset id from the manifest
            resolved = key
            if key not in {r["key"] for r in self.state.availability}:
                for r in self.state.availability:
                    manifest = self.state.manifest_for(r["key"])
                    if manifest and (manifest.get("dataset") or {}).get("id") == key:
                        resolved = r["key"]
                        break
            _send_json(self, 200, self.state.pack_logs(resolved, name))
            return
        if path.startswith("/api/file/"):
            rest = path[len("/api/file/") :]
            parts = rest.split("/", 1)
            if len(parts) != 2:
                raise ValueError("path must be /api/file/<dataset>/<relative-file>")
            key, rel = parts[0], unquote(parts[1])
            row = next((r for r in self.state.availability if r["key"] == key), None)
            if not row or not row.get("path"):
                raise FileNotFoundError(f"Unknown or unavailable dataset: {key}")
            found = safe_under(Path(row["path"]), rel)
            if found is None:
                raise FileNotFoundError("file not found (or outside the dataset folder)")
            _send(self, 200, found.read_bytes(), _guess_type(found))
            return
        _send_json(self, 404, {"error": "not found"})


def pick_port(preferred: int = 8765) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 30)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def serve(host: str, port: int, open_browser: bool) -> int:
    state = ViewerState()
    handler = partial(ViewerHandler)
    ViewerHandler.state = state
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Well map viewer  {url}", flush=True)
    print("Close this window to stop the server.", flush=True)
    print(flush=True)
    available = [r for r in state.availability if r["status"] == "available"]
    missing = [r for r in state.availability if r["status"] != "available"]
    for r in available:
        print(f"  available     {r['name']}: {r['path']}")
    for r in missing:
        print(f"  not available {r['name']}")
    print(flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    ap = argparse.ArgumentParser(
        prog="python -m wmb.viewer",
        description="Serve the data-free well map using folders from data_paths.local.yaml.",
    )
    ap.add_argument("--launch", action="store_true", help="open the default browser and print a stop message")
    ap.add_argument("--port", type=int, default=0, help="port (default: first free port from 8765)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = ap.parse_args(argv)
    interactive = sys.stdin.isatty() or args.launch
    try:
        ensure_config(interactive=interactive)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    port = args.port or pick_port()
    open_browser = (not args.no_browser) and (args.launch or sys.stdin.isatty())
    return serve(args.host, port, open_browser=open_browser)


if __name__ == "__main__":
    sys.exit(main())
