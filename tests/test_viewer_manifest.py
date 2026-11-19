"""Auto-built viewer manifests: FORCE CSV + Sodir placement, LAS placement, cache location.

Tiny synthetic fixtures only (temp folders; no real well data).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

FORCE_CSV = (
    "WELL;DEPTH_MD;X_LOC;Y_LOC;GR;RHOB\n"
    "15/9-13;1000;440000;6480000;50;2.4\n"
    "15/9-13;1001;440000;6480000;55;-999.25\n"
    "99/9-1;2000;500000;6500000;60;2.5\n"
    "99/9-1;2001;500000;6500000;;2.6\n"
    "16/1-5;3000;0;0;70;2.3\n"
)
SODIR_CSV = (
    "wlbWellboreName,wlbNsDecDeg,wlbEwDecDeg,wlbEwUtm,wlbNsUtm,wlbUtmZone,wlbGeodeticDatum\n"
    "15/9-13,58.4,1.9,440100,6481000,31,ED50\n"
    "15/9-F-1,58.44,1.88,438000,6478000,31,ED50\n"
)


def _las(well: str, header_extra: str = "") -> str:
    return (
        "~Version\n VERS. 2.0 : \n WRAP. NO : \n"
        f"~Well\n STRT.M 1000 : \n STOP.M 1001 : \n STEP.M 1 : \n NULL. -999.25 : \n WELL. {well} : WELL\n"
        f"{header_extra}"
        "~Curve\n DEPT.M : Depth\n GR.GAPI : Gamma\n"
        "~A\n1000 50\n1001 55\n"
    )


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("FH2026_DATA", raising=False)
    monkeypatch.delenv("FH2026_CACHE", raising=False)
    paths = _load("wmb.paths", REPO / "src" / "wmb" / "paths.py")
    viewer = _load("wmb.viewer", REPO / "src" / "wmb" / "viewer.py")
    data, cache = tmp_path / "data", tmp_path / "cache"
    (data / "FORCE2020").mkdir(parents=True)
    (data / "FORCE2020" / "train.csv").write_text(FORCE_CSV, encoding="utf-8")
    (data / "Sodir-FactPages").mkdir()
    (data / "Sodir-FactPages" / "wellbore_exploration_all.csv").write_text(SODIR_CSV, encoding="utf-8")
    volve = data / "Volve"
    (volve / "15_9-F-1").mkdir(parents=True)
    (volve / "15_9-F-1" / "a.las").write_text(_las("15_9-F-1"), encoding="utf-8")
    (volve / "HDR-WELL").mkdir()
    for i in range(3):
        (volve / "HDR-WELL" / f"a{i}.las").write_text(_las("HDR-WELL"), encoding="utf-8")
    (volve / "HDR-WELL" / "b.las").write_text(
        _las("HDR-WELL", " LAT . 58.5 S : Latitude\n LONG. 2 15 0 W : Longitude\n"), encoding="utf-8"
    )
    (volve / "NOWHERE").mkdir()
    (volve / "NOWHERE" / "c.las").write_text(_las("NOWHERE"), encoding="utf-8")
    cfg = tmp_path / "data_paths.local.yaml"
    cfg.write_text(f'data_root: "{data.as_posix()}"\ncache_dir: "{cache.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("WMB_DATA_PATHS", str(cfg))
    paths._reset_for_tests()
    yield viewer, data, cache
    paths._reset_for_tests()


def _wells(manifest):
    return {w["name"]: w for w in manifest["wells"]}


def test_force_index_sodir_and_utm_fallback(env):
    viewer, data, cache = env
    notes: list[str] = []
    m = viewer.load_dataset_manifest("force2020", data / "FORCE2020", "FORCE 2020", notes)
    wells = _wells(m)
    assert set(wells) == {"15/9-13", "99/9-1", "16/1-5"}
    assert wells["15/9-13"]["coord_source"] == "sodir_factpages"
    assert 58 < wells["15/9-13"]["lat"] < 59 and 1 < wells["15/9-13"]["lon"] < 3
    assert wells["99/9-1"]["coord_source"] == "force_xy_utm"
    assert wells["99/9-1"]["lat"] is not None
    assert wells["16/1-5"]["lat"] is None  # unplaced but still listed
    assert wells["15/9-13"]["split"] == "train"
    comp = wells["15/9-13"]["completeness"]
    assert comp["curves"]["GR"] == 100 and comp["curves"]["RHOB"] == 50


def test_unplaced_wells_remain_in_payload(env):
    viewer, data, cache = env
    state = viewer.ViewerState()
    payload = state.map_payload()
    names = {w["name"] for w in payload["wells"] if w["dataset"] == "force2020"}
    assert "16/1-5" in names
    assert any(u["name"] == "16/1-5" for u in payload["unplaced"])
    assert all(w["logs_pending"] for w in payload["wells"])  # curves load on demand


def test_las_sodir_match_header_fallback_and_unplaced(env):
    viewer, data, cache = env
    m = viewer.load_dataset_manifest("volve", data / "Volve", "Volve", [])
    wells = _wells(m)
    assert wells["15_9-F-1"]["coord_source"] == "sodir_factpages"
    assert wells["HDR-WELL"]["coord_source"] == "las_header"
    assert wells["HDR-WELL"]["lat"] == pytest.approx(-58.5)
    assert wells["HDR-WELL"]["lon"] == pytest.approx(-2.25)
    assert wells["NOWHERE"].get("lat") is None
    assert wells["15_9-F-1"]["files"][0]["path"] == "15_9-F-1/a.las"


def test_las_header_xy_with_crs(env, tmp_path):
    from wmb import viewer_manifest as vm

    f = tmp_path / "xy.las"
    f.write_text(
        _las("XY", " XCOORD.M 440000 : Easting\n YCOORD.M 6480000 : Northing\n CRS. EPSG:23031 : ED50 UTM 31N\n"),
        encoding="utf-8",
    )
    rec = vm.las_header_coords(f)
    assert rec["coord_source"] == "las_header"
    assert 58 < rec["lat"] < 59


@pytest.mark.parametrize(
    ("crs", "expected_lat", "expected_crs"),
    [
        ("EPSG:32731", -31.82, "EPSG:32731"),
        ("EPSG:25831", 58.46, "EPSG:25831"),
        ("EPSG:3857 (not WGS84 UTM 31N)", None, None),
    ],
)
def test_las_header_projected_crs(tmp_path, crs, expected_lat, expected_crs):
    from wmb import viewer_manifest as vm

    f = tmp_path / "xy.las"
    f.write_text(
        _las("XY", f" X.M 4.4E5 : Easting\n Y.M 6480000 : Northing\n CRS. {crs} : CRS\n"),
        encoding="utf-8",
    )
    rec = vm.las_header_coords(f)
    if expected_crs is None:
        assert rec is None
    else:
        assert rec["lat"] == pytest.approx(expected_lat, abs=0.1)
        assert rec["coord_source_crs"] == expected_crs


def test_cache_under_cache_dir_and_reused(env, monkeypatch):
    viewer, data, cache = env
    from wmb import viewer_manifest as vm

    viewer.load_dataset_manifest("force2020", data / "FORCE2020", "FORCE 2020", [])
    cdir = cache / "viewer-manifests" / "force2020"
    assert (cdir / "manifest.json").is_file() and (cdir / "well_index.json").is_file()
    assert not list(data.rglob("wells.json"))
    assert not list(data.rglob("viewer-manifests"))

    def reject_csv(*args, **kwargs):
        pytest.fail("warm FORCE manifest should not re-read the CSV or rebuild the index")

    read_text = Path.read_text
    with monkeypatch.context() as patch:
        patch.setattr(vm, "index_force_csv", reject_csv)
        patch.setattr(Path, "read_text", lambda p, *a, **kw: (
            reject_csv() if p.suffix == ".csv" else read_text(p, *a, **kw)
        ))
        assert "15/9-13" in _wells(viewer.load_dataset_manifest("force2020", data / "FORCE2020", "FORCE 2020"))

    # Poison the cached index: a valid cache must be reused instead of re-reading the CSV.
    idx = json.loads((cdir / "well_index.json").read_text(encoding="utf-8"))
    (cdir / "manifest.json").unlink()
    idx["wells"][0]["name"] = "CACHED"
    (cdir / "well_index.json").write_text(json.dumps(idx), encoding="utf-8")
    m = viewer.load_dataset_manifest("force2020", data / "FORCE2020", "FORCE 2020", [])
    assert "CACHED" in _wells(m)

    # Changing an input invalidates it.
    (data / "FORCE2020" / "train.csv").write_text(FORCE_CSV + "17/1-1;1;;;1;1\n", encoding="utf-8")
    m = viewer.load_dataset_manifest("force2020", data / "FORCE2020", "FORCE 2020", [])
    assert "CACHED" not in _wells(m) and "17/1-1" in _wells(m)


def test_missing_datasets_stay_not_available(env):
    viewer, data, cache = env
    payload = viewer.ViewerState().map_payload()
    by_key = {d["key"]: d for d in payload["datasets"]}
    assert by_key["nlog_dts"]["status"] == "not_available"
    assert by_key["kansas_kgs_dts"]["status"] == "not_available"
    assert by_key["force2020"]["status"] == "available"
    assert by_key["volve"]["status"] == "available"


def test_clicked_well_loads_curves_on_demand(env):
    viewer, data, _ = env
    (data / "FORCE2020" / "hidden_test.csv").write_text(
        "WELL;DEPTH_MD;GR\n29/1-1;1000;72\n", encoding="utf-8"
    )
    state = viewer.ViewerState()
    assert state.pack_logs("force2020", "29/1-1")["c"]["GR"] == [72]
    assert set(state._csv_frames) == {"force2020:hidden"}


def test_unplaced_las_well_loads_curves_on_demand(env):
    pytest.importorskip("lasio")
    viewer, _, _ = env
    state = viewer.ViewerState()
    assert state.pack_logs("volve", "NOWHERE")["c"]["GR"] == [50, 55]


@pytest.mark.parametrize("mnemonic", ["LFP_GR", "VSHGR"])
def test_prefixed_las_mnemonics_map_to_core_curves(tmp_path, monkeypatch, mnemonic):
    pytest.importorskip("lasio")
    monkeypatch.syspath_prepend(str(REPO / "viewer"))
    from wellmaplib.logs import try_read_las, STANDARD_CURVES

    text = (
        "~Version\n VERS. 2.0 : \n WRAP. NO : \n"
        "~Well\n STRT.M 1000 : \n STOP.M 1001 : \n STEP.M 1 : \n NULL. -999.25 : \n WELL. X : WELL\n"
        f"~Curve\n DEPT.M : Depth\n {mnemonic}.GAPI : Gamma\n LFP_DT.US/F : Sonic\n LFP_DTS.US/F : Shear\n"
        "~A\n1000 50 80 160\n1001 55 81 161\n"
    )
    path = tmp_path / "lfp.las"
    path.write_text(text, encoding="utf-8")
    df = try_read_las(path, {c: c for c in STANDARD_CURVES})
    if mnemonic == "LFP_GR":
        assert list(df["GR"]) == [50, 55]
    else:
        assert "GR" not in df
    assert list(df["DTC"]) == [80, 81]
    assert list(df["DTS"]) == [160, 161]
