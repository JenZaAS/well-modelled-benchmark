"""Tests for wmb.paths: temp yaml only, no real well data required."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import importlib.util
import pytest

REPO = Path(__file__).resolve().parents[1]

# Load paths.py directly so these tests do not require numpy/pandas
# (importing the wmb package executes src/wmb/__init__.py).
_spec = importlib.util.spec_from_file_location(
    "wmb.paths", REPO / "src" / "wmb" / "paths.py"
)
paths = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = paths
_spec.loader.exec_module(paths)

PATHS_FILE = REPO / "src" / "wmb" / "paths.py"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the loader at a temporary yaml and clear the cache."""
    monkeypatch.delenv("FH2026_DATA", raising=False)
    monkeypatch.delenv("FH2026_WORK", raising=False)
    monkeypatch.delenv("FH2026_CACHE", raising=False)
    monkeypatch.delenv("WMB_DATA_PATHS", raising=False)
    paths._reset_for_tests()
    yield tmp_path
    paths._reset_for_tests()


def _write_tree(root: Path, folders: dict[str, list[str]]) -> None:
    for folder, files in folders.items():
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        for name in files:
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x", encoding="utf-8")


def test_missing_file_points_at_setup_docs(isolated, monkeypatch):
    monkeypatch.setenv("WMB_DATA_PATHS", str(isolated / "does-not-exist.yaml"))
    with pytest.raises(paths.DataPathsError) as exc:
        paths.data_root()
    msg = str(exc.value)
    assert "docs/SETUP_DATA_PATHS.md" in msg
    assert "data_paths.local.yaml" in msg


def test_load_from_temp_yaml_and_default_folders(isolated, monkeypatch):
    data = isolated / "data"
    _write_tree(
        data,
        {
            "FORCE2020": ["train.csv"],
            "Volve": ["15_9-F-1/well.las"],
        },
    )
    cfg = isolated / "data_paths.local.yaml"
    cfg.write_text(f'data_root: "{data.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("WMB_DATA_PATHS", str(cfg))

    assert paths.data_root() == data
    assert paths.get_dataset_path("force2020") == data / "FORCE2020"
    assert paths.get_dataset_path("FORCE2020") == data / "FORCE2020"
    assert paths.get_dataset_path("volve") == data / "Volve"
    # Preferred name is used when the folder is absent.
    assert paths.get_dataset_path("nlog_dts") == data / "NLOG-DTS"
    assert not paths.get_dataset_path("nlog_dts").exists()


def test_per_dataset_override_and_alias_folders(isolated, monkeypatch):
    data = isolated / "data"
    other = isolated / "elsewhere" / "dutch"
    other.mkdir(parents=True)
    (other / "a.las").write_text("x", encoding="utf-8")
    _write_tree(data, {"nlog": ["w1.las"]})  # legacy folder name
    cfg = isolated / "cfg.yaml"
    cfg.write_text(
        f'data_root: "{data.as_posix()}"\n'
        f'nlog_dts: "{other.as_posix()}"\n'
        f'cache_dir: "{(isolated / "cache").as_posix()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("WMB_DATA_PATHS", str(cfg))

    assert paths.get_dataset_path("nlog") == other
    assert paths.get_dataset_path("nlog_dts") == other
    assert paths.cache_dir() == isolated / "cache"
    assert paths.work_root() == isolated / "cache" / "work"


def test_fh2026_data_env_fallback(isolated, monkeypatch):
    data = isolated / "legacy"
    data.mkdir()
    (data / "FORCE2020").mkdir()
    (data / "FORCE2020" / "train.csv").write_text("x", encoding="utf-8")
    monkeypatch.setenv("WMB_DATA_PATHS", str(isolated / "no-such.yaml"))
    monkeypatch.setenv("FH2026_DATA", str(data))
    assert paths.get_dataset_path("force2020") == data / "FORCE2020"


def test_dataset_status_found_and_missing(isolated, monkeypatch):
    data = isolated / "data"
    _write_tree(
        data,
        {
            "FORCE2020": ["train.csv", "wells.json"],
            "Kansas-KGS-DTS": ["x.las"],
        },
    )
    cfg = isolated / "cfg.yaml"
    cfg.write_text(f'data_root: "{data.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("WMB_DATA_PATHS", str(cfg))

    by_key = {row.key: row for row in paths.dataset_status()}
    assert by_key["force2020"].found
    assert by_key["force2020"].n_csv == 1
    assert by_key["force2020"].has_manifest
    assert by_key["kansas_kgs_dts"].found
    assert by_key["kansas_kgs_dts"].n_las == 1
    assert not by_key["volve"].found
    assert not by_key["nlog_dts"].exists


def test_write_local_paths(isolated):
    dest = isolated / "data_paths.local.yaml"
    root = isolated / "shared"
    root.mkdir()
    paths.write_local_paths(root, dest=dest)
    text = dest.read_text(encoding="utf-8")
    assert "Do not commit" in text
    assert root.as_posix() in text


def test_check_cli(isolated, monkeypatch):
    data = isolated / "data"
    _write_tree(data, {"FORCE2020": ["train.csv"]})
    cfg = isolated / "cfg.yaml"
    cfg.write_text(f'data_root: "{data.as_posix()}"\n', encoding="utf-8")
    env = {**os.environ, "WMB_DATA_PATHS": str(cfg), "PYTHONPATH": str(REPO / "src")}
    env.pop("FH2026_DATA", None)
    r = subprocess.run(
        [sys.executable, str(PATHS_FILE), "--check"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "force2020" in r.stdout and "found" in r.stdout
    assert "nlog_dts" in r.stdout and "missing" in r.stdout


def test_check_cli_missing_config(isolated, monkeypatch):
    env = {
        **os.environ,
        "WMB_DATA_PATHS": str(isolated / "none.yaml"),
        "PYTHONPATH": str(REPO / "src"),
    }
    env.pop("FH2026_DATA", None)
    r = subprocess.run(
        [sys.executable, str(PATHS_FILE), "--check"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1
    assert "docs/SETUP_DATA_PATHS.md" in r.stderr
