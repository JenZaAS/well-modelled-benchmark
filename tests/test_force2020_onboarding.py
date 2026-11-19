"""Tests for the FORCE 2020 loader, check command and onboarding. Synthetic data only."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("yaml")

from wmb import onboarding, paths
from wmb.datasets import load_canonical_spec, load_force2020
from wmb.datasets.force2020 import check_force2020, curve_availability

LITH = "FORCE_2020_LITHOFACIES_LITHOLOGY"
LITH_CONF = "FORCE_2020_LITHOFACIES_CONFIDENCE"
FEATURES = ["GR", "RDEP", "RMED", "CALI", "RHOB", "NPHI", "PEF", "DTC", "DTS"]


def _rows(well: str, with_dts: bool, labels: bool, n: int = 5) -> list[str]:
    out = []
    for i in range(n):
        vals = {"GR": 60 + i, "RDEP": 1.5, "RMED": 1.2, "CALI": 8.5, "RHOB": 2.4, "NPHI": 0.25,
                "PEF": 3.0, "DTC": 90.0, "DTS": 160.0 if with_dts else ""}
        if i == 1:
            vals["GR"] = -999.25
        cells = [well, str(1000.0 + i), "1.0", "2.0", "-1000.0", "SOME_GROUP", "SOME_FM"]
        cells += [str(vals[c]) for c in FEATURES]
        if labels:
            cells += ["30000", "1"]
        out.append(";".join(cells))
    return out


def _header(labels: bool) -> str:
    cols = ["WELL", "DEPTH_MD", "X_LOC", "Y_LOC", "Z_LOC", "GROUP", "FORMATION"] + FEATURES
    return ";".join(cols + ([LITH, LITH_CONF] if labels else []))


@pytest.fixture()
def force_dir(tmp_path: Path) -> Path:
    root = tmp_path / "FORCE2020"
    root.mkdir()
    (root / "train.csv").write_text(
        "\n".join([_header(True), *_rows("A/1", True, True), *_rows("B/2", False, True)]) + "\n")
    (root / "hidden_test.csv").write_text("\n".join([_header(True), *_rows("H/1", True, True)]) + "\n")
    (root / "leaderboard_test_features.csv").write_text(
        "\n".join([_header(False), *_rows("L/1", False, False)]) + "\n")
    (root / "leaderboard_test_target.csv").write_text(
        "WELL;DEPTH_MD;" + LITH + "\n" + "\n".join(f"L/1;{1000.0 + i};65000" for i in range(5)) + "\n")
    return root


def _mapping_for(root: Path) -> dict:
    files = []
    for name, role in [("train.csv", "required"), ("train.zip", "optional"),
                       ("leaderboard_test_features.csv", "required"),
                       ("leaderboard_test_target.csv", "required"),
                       ("hidden_test.csv", "required"), ("penalty_matrix.npy", "recommended")]:
        p = root / name
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "0" * 64
        files.append({"name": name, "role": role, "sha256": digest})
    return {"files": files}


def test_canonical_spec_loads():
    spec = load_canonical_spec()
    assert "DTS" in spec["curves"] and spec["curves"]["GR"]["plausible_range"]


def test_load_force2020(force_dir: Path):
    wells = load_force2020(root=force_dir)
    by_name = {w.name: w for w in wells}
    assert {w.split for w in wells} == {"train", "leaderboard", "hidden"}
    a = by_name["A/1"]
    assert set(FEATURES) <= set(a.curves) and "LITH" in a.curves and "LITH_CONF" in a.curves
    assert a.depth[0] == 1000.0
    assert np.isnan(a.curves["GR"][1]) and not np.isnan(a.curves["GR"][0])
    assert a.curves["LITH"][0] == 30000
    assert a.meta["X_LOC"] == 1.0 and a.meta["GROUP"][0] == "SOME_GROUP"
    assert by_name["L/1"].curves["LITH"][0] == 65000
    table = curve_availability(wells)
    assert table["train"]["wells"] == 2 and table["train"]["DTS"] == 1
    assert table["all"]["DTS"] == 2 and table["leaderboard"]["DTS"] == 0
    assert [w.split for w in load_force2020(root=force_dir, splits=["hidden"])] == ["hidden"]


def test_guardrail_flag_does_not_change_values(force_dir: Path):
    text = (force_dir / "hidden_test.csv").read_text().replace(";3.0;", ";383.0;")
    (force_dir / "hidden_test.csv").write_text(text)
    w = load_force2020(root=force_dir, splits=["hidden"])[0]
    assert "PEF" in w.meta["median_outside_plausible_range"]
    assert w.curves["PEF"][0] == 383.0


def test_check_passes(force_dir: Path):
    out = io.StringIO()
    assert check_force2020(force_dir, mapping=_mapping_for(force_dir), out=out) == 0
    assert "DTS" in out.getvalue() and "warning   penalty_matrix.npy" in out.getvalue()


def test_check_fails_missing_required(force_dir: Path):
    mapping = _mapping_for(force_dir)
    (force_dir / "hidden_test.csv").unlink()
    assert check_force2020(force_dir, mapping=mapping, out=io.StringIO()) == 1


def test_check_fails_on_mismatch_without_modifying(force_dir: Path):
    mapping = _mapping_for(force_dir)
    target = force_dir / "hidden_test.csv"
    target.write_text(target.read_text() + "extra\n")
    before = target.read_bytes()
    assert check_force2020(force_dir, mapping=mapping, out=io.StringIO()) == 1
    assert target.read_bytes() == before


def test_check_unzips_valid_train_zip(force_dir: Path):
    import zipfile

    with zipfile.ZipFile(force_dir / "train.zip", "w") as zf:
        zf.write(force_dir / "train.csv", "train.csv")
    mapping = _mapping_for(force_dir)
    (force_dir / "train.csv").unlink()
    assert check_force2020(force_dir, mapping=mapping, show_table=False, out=io.StringIO()) == 0
    assert (force_dir / "train.csv").is_file()


@pytest.fixture()
def onboard_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WMB_DATA_PATHS", str(tmp_path / "data_paths.local.yaml"))
    monkeypatch.setenv("WMB_ONBOARDING_STATE", str(tmp_path / ".wmb" / "onboarding_state.json"))
    paths._reset_for_tests()
    yield tmp_path
    paths._reset_for_tests()


def test_onboarding_flow(onboard_env: Path, monkeypatch, capsys):
    state_file = onboard_env / ".wmb" / "onboarding_state.json"
    assert onboarding.main(["--status"]) == 0
    assert "not started" in capsys.readouterr().out and not state_file.exists()

    calls = []
    monkeypatch.setattr(onboarding, "verify_dataset", lambda key: calls.append(key) or True)
    base = ["--non-interactive", "--data-root", str(onboard_env / "data"), "--choose", "force2020"]
    assert onboarding.main(base) == 0
    state = json.loads(state_file.read_text())
    assert state["onboarding_version"] == onboarding.ONBOARDING_VERSION
    assert state["datasets"]["force2020"]["choice"] == "chosen"
    assert state["datasets"]["force2020"]["verified"] is True
    assert (onboard_env / "data_paths.local.yaml").is_file()
    assert state["steps"]["data_paths"]["done"]
    capsys.readouterr()

    assert onboarding.main(["--non-interactive"]) == 0
    assert "skipping" in capsys.readouterr().out and calls == ["force2020", "force2020"]

    assert onboarding.main(["--non-interactive", "--redo", "force2020"]) == 0
    assert calls == ["force2020"] * 3
    assert json.loads(state_file.read_text())["datasets"]["force2020"]["choice"] == "chosen"

    assert onboarding.main(["--non-interactive", "--decline", "force2020"]) == 0
    assert json.loads(state_file.read_text())["datasets"]["force2020"]["choice"] == "declined"
    assert calls == ["force2020"] * 3

    monkeypatch.setattr(onboarding, "ONBOARDING_VERSION", 2)
    monkeypatch.setattr(onboarding, "CHANGELOG", {**onboarding.CHANGELOG, 2: "Added a new dataset."})
    capsys.readouterr()
    assert onboarding.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert "what's new" in out and "Added a new dataset." in out


def test_check_does_not_publish_bad_extracted_csv(force_dir: Path):
    import hashlib
    import zipfile

    (force_dir / "train.csv").unlink()
    with zipfile.ZipFile(force_dir / "train.zip", "w") as zf:
        zf.writestr("train.csv", "WELL;DEPTH_MD\nA/1;1\n")
    mapping = _mapping_for(force_dir)
    mapping["files"] = [
        f if f["name"] != "train.zip"
        else {**f, "sha256": hashlib.sha256((force_dir / "train.zip").read_bytes()).hexdigest()}
        for f in mapping["files"]
    ]
    assert check_force2020(force_dir, mapping=mapping, show_table=False, out=io.StringIO()) == 1
    assert not (force_dir / "train.csv").exists()
    assert not (force_dir / "train.csv.part").exists()


def test_onboarding_preserves_yaml_overrides(onboard_env: Path, monkeypatch):
    cfg = onboard_env / "data_paths.local.yaml"
    monkeypatch.setattr(onboarding, "verify_dataset", lambda key: True)
    data = onboard_env / "data"
    data.mkdir()
    custom = onboard_env / "custom"
    cache = onboard_env / "cache"
    custom.mkdir()
    cfg.write_text(
        "data_root: {0}\nforce2020: {1}\ncache_dir: {2}\n".format(
            data.as_posix(), custom.as_posix(), cache.as_posix()
        ),
        encoding="utf-8",
    )
    assert onboarding.main([
        "--non-interactive", "--redo", "data_paths",
        "--data-root", str(data), "--decline", "force2020",
    ]) == 0
    text = cfg.read_text(encoding="utf-8")
    assert "force2020:" in text and "cache_dir:" in text
    assert custom.as_posix() in text and cache.as_posix() in text


def test_status_hides_whats_new_until_state_exists(onboard_env: Path, capsys):
    assert onboarding.main(["--status"]) == 0
    assert "what's new" not in capsys.readouterr().out.lower()
