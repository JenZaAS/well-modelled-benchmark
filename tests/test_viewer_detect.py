"""Dataset-detection for the live viewer (temp folders only, no real data)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mods(tmp_path, monkeypatch):
    monkeypatch.delenv("FH2026_DATA", raising=False)
    monkeypatch.delenv("FH2026_WORK", raising=False)
    monkeypatch.delenv("FH2026_CACHE", raising=False)
    paths = _load("wmb.paths", REPO / "src" / "wmb" / "paths.py")
    viewer = _load("wmb.viewer", REPO / "src" / "wmb" / "viewer.py")
    paths._reset_for_tests()
    data = tmp_path / "data"
    (data / "FORCE2020").mkdir(parents=True)
    (data / "FORCE2020" / "train.csv").write_text("WELL;DEPTH_MD\nA;1\n", encoding="utf-8")
    (data / "Volve" / "15_9-F-1").mkdir(parents=True)
    (data / "Volve" / "15_9-F-1" / "log.las").write_text("~W\n", encoding="utf-8")
    (data / "NLOG-DTS").mkdir()  # empty
    cfg = tmp_path / "data_paths.local.yaml"
    cfg.write_text(f'data_root: "{data.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("WMB_DATA_PATHS", str(cfg))
    paths._reset_for_tests()
    yield paths, viewer, data
    paths._reset_for_tests()


def test_discover_marks_missing_and_present(mods):
    _paths, viewer, data = mods
    rows = {r["key"]: r for r in viewer.discover_for_viewer()}
    assert rows["force2020"]["status"] == "available"
    assert rows["force2020"]["n_csv"] == 1
    assert rows["force2020"]["has_manifest"] is False
    assert "temporary manifest" in (rows["force2020"]["note"] or "")
    assert rows["volve"]["status"] == "available"
    assert rows["volve"]["n_las"] == 1
    assert rows["nlog_dts"]["status"] == "not_available"
    assert rows["kansas_kgs_dts"]["status"] == "not_available"
    assert rows["kansas_kgs_dts"]["path"] is None
    assert rows["sodir_factpages"]["role"] == "support"


def test_las_scan_groups_by_well_folder(mods):
    _paths, viewer, data = mods
    manifest = viewer.load_dataset_manifest("volve", data / "Volve", "Volve")
    assert manifest is not None
    assert manifest["_generated"] is True
    assert [w["name"] for w in manifest["wells"]] == ["15_9-F-1"]
    assert manifest["wells"][0]["files"][0]["path"] == "15_9-F-1/log.las"


def test_force_without_manifest_has_no_las_scan(mods):
    _paths, viewer, data = mods
    # CSVs only: on-the-fly LAS scan returns None; the dataset stays listed.
    manifest = viewer.load_dataset_manifest("force2020", data / "FORCE2020", "FORCE 2020")
    assert manifest is None
    rows = {r["key"]: r for r in viewer.discover_for_viewer()}
    assert rows["force2020"]["status"] == "available"


def test_safe_under_blocks_traversal(mods):
    _paths, viewer, data = mods
    secret = data / "secret.txt"
    secret.write_text("nope", encoding="utf-8")
    root = data / "Volve"
    assert viewer.safe_under(root, "15_9-F-1/log.las") is not None
    assert viewer.safe_under(root, "../secret.txt") is None
    assert viewer.safe_under(root, str(secret)) is None
    assert viewer.safe_under(root, "/etc/passwd") is None
