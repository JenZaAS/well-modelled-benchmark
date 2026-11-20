"""YAML test definitions: schema, old-CLI mapping, new_test/check_test, runner and scoreboard. Synthetic data only."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmark import check_test, new_test  # noqa: E402
from benchmark import test_def as td  # noqa: E402
from wmb import synthetic  # noqa: E402

SHIPPED = {
    "rhob_missing": ("RHOB", "missing_curve"), "rhob_gap50": ("RHOB", "gap_50"), "dtc_missing": ("DTC", "missing_curve"),
    "dtc_gap50": ("DTC", "gap_50"),
    "dts_missing": ("DTS", "missing_curve"), "dts_gap50": ("DTS", "gap_50"),
}

GOOD = {
    "version": 1, "name": "x", "description": "d", "dataset": "synthetic", "target": "RHOB",
    "scenario": {"type": "missing_curve"}, "wells": {"split": "test", "names": []}, "inputs_policy": "model",
    "metrics": ["MAE", "RMSE"],
}


def good(**over):
    d = copy.deepcopy(GOOD)
    for k, v in over.items():
        if k in ("scenario", "wells"):
            d[k].update(v)
        else:
            d[k] = v
    return d


def write(dirpath, name, data):
    p = Path(dirpath) / f"{name}.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def run_cli(tmp_path, *args):
    env = {**os.environ, "WMB_RESULTS": str(tmp_path / "res"), "WMB_TASK_CACHE": str(tmp_path / "tasks")}
    return subprocess.run([sys.executable, "-m", "benchmark.run", *args], cwd=REPO, env=env, capture_output=True, text=True)


def test_shipped_defs_are_valid():
    defs = {d.name: d for d in td.list_defs()}
    assert set(SHIPPED) <= set(defs)
    for name, (target, key) in SHIPPED.items():
        d = defs[name]
        assert d.path.stem == name and d.dataset == "force2020" and d.wells_split == "test"
        assert (d.target, d.scenario_key) == (target, key) and d.n_gaps == 1
        assert td.validate_def(td.read_yaml(d.path), path=d.path) == []


def test_template_is_not_runnable():
    template = td.DEFS_DIR / "_template.yaml"
    assert template.is_file()
    with pytest.raises(td.TestDefError, match="not a runnable"):
        td.load_def(template)
    assert all(not d.name.startswith("_") and d.name.lower() != "template" for d in td.list_defs())


@pytest.mark.parametrize("data, text", [
    (good(target="GR"), "unknown target"),
    (good(scenario={"type": "bogus"}), "scenario.type"),
    (good(scenario={"type": "gap"}), "gap_length_m"),
    (good(scenario={"type": "gap", "gap_length_m": float("inf")}), "gap_length_m"),
    (good(scenario={"gap_length_m": 50}), "missing_curve"),
    (good(scenario={"n_gaps": 0}), "n_gaps"),
    (good(wells={"split": "everything"}), "wells.split"),
    (good(wells={"split": "list", "names": []}), "non-empty"),
    (good(inputs_policy="all"), "inputs_policy"),
    (good(metrics=["MAE", "nonsense"]), "nonsense"),
    (good(metrics=[["MAE"]]), "unknown metrics"),
    (good(dataset="no_such_dataset"), "unknown dataset"),
    (good(version=2), "version"),
    (good(version=True), "version"),
])
def test_invalid_defs(data, text):
    assert any(text in e for e in td.validate_def(data, path="x.yaml")), td.validate_def(data, path="x.yaml")


def test_good_def_and_name_must_match_stem():
    assert td.validate_def(good(), path="x.yaml") == []
    assert any("file stem" in e for e in td.validate_def(good(), path="other.yaml"))
    assert td.validate_def(good(scenario={"type": "gap", "gap_length_m": 50, "n_gaps": 2})) == []


def test_unknown_wells_names():
    assert any("not in the dataset" in e for e in td.validate_def(
        good(wells={"split": "list", "names": ["SYN-01", "NOPE"]}), dataset_wells=["SYN-01"]))


def test_old_cli_mapping_names():
    assert td.generated_def("synthetic", "RHOB", "missing_curve").name == "synthetic_rhob_missing"
    assert td.generated_def("synthetic", "DTC", "gap_50").name == "synthetic_dtc_gap_50"
    assert td.generated_def("force2020", "RHOB", "missing_curve").name == "rhob_missing"
    assert td.generated_def("force2020", "DTS", "gap_50").name == "dts_gap50"
    assert td.generated_def("force2020", "DTS", "gap_10").name == "force2020_dts_gap_10"
    assert td.match_shipped("force2020", "RHOB", "gap_100") is None


def test_new_test_and_check_test(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(td, "_load_wells", lambda tdef: (_ for _ in ()).throw(FileNotFoundError("no data")))
    assert new_test.main([], defs_dir=tmp_path) == 2
    assert new_test.main(["my_test"], defs_dir=tmp_path) == 0
    text = (tmp_path / "my_test.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(text)["name"] == "my_test"
    assert td.validate_def(yaml.safe_load(text), path=tmp_path / "my_test.yaml") == []
    assert new_test.main(["my_test"], defs_dir=tmp_path) == 2  # no silent overwrite
    (tmp_path / "my_test.yaml").write_text("changed", encoding="utf-8")
    assert new_test.main(["my_test", "--force"], defs_dir=tmp_path) == 0
    assert new_test.main(["bad name"], defs_dir=tmp_path) == 2
    assert new_test.main(["all"], defs_dir=tmp_path) == 2
    assert check_test.main(["my_test"], defs_dir=tmp_path) == 0
    write(tmp_path, "broken", good(name="broken", target="GR"))
    assert check_test.main(["all"], defs_dir=tmp_path) == 1
    assert new_test.main(["--check", "my_test"], defs_dir=tmp_path) == 0
    assert "unknown target" in capsys.readouterr().err
    assert sorted(p.name for p in td.DEFS_DIR.glob("*.yaml")) == sorted(
        [f"{n}.yaml" for n in [*SHIPPED, "synthetic_rhob_missing", "nlog_dts_missing", "kansas_dts_missing"]] + ["_template.yaml"])


def test_new_test_exit_status_is_schema_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(td, "_load_wells", lambda tdef: ([], []))
    assert new_test.main(["yes"], defs_dir=tmp_path) == 0
    assert td.load_def("yes", tmp_path).name == "yes"  # not a YAML boolean
    assert "warning: no test well" in capsys.readouterr().out
    assert check_test.main(["yes"], defs_dir=tmp_path) == 1


def test_check_def_data_checks(tmp_path):
    wells = synthetic.make_wells(8, 500, 0)
    train, test = wells[:5], wells[5:]
    ok = td.load_def(write(tmp_path, "ok", good(name="ok")))
    assert td.check_def(ok, wells=(train, test)) == ([], [])  # 3 usable wells: no warning
    one = td.load_def(write(tmp_path, "one", good(name="one", wells={"split": "list", "names": [test[0].name]})))
    errors, warnings = td.check_def(one, wells=(train, test))
    assert errors == [] and any("only 1" in w for w in warnings)
    lb = td.load_def(write(tmp_path, "lb", good(name="lb", wells={"split": "leaderboard"})))
    assert any("leaderboard" in e for e in td.check_def(lb, wells=(train, test))[0])
    ghost = td.load_def(write(tmp_path, "ghost", good(name="ghost", wells={"split": "list", "names": ["NOPE"]})))
    assert any("not in the dataset" in e for e in td.check_def(ghost, wells=(train, test))[0])
    train_name = td.load_def(write(tmp_path, "tn", good(name="tn", wells={"split": "list", "names": [train[0].name]})))
    assert any("training wells" in e for e in td.check_def(train_name, wells=(train, test))[0])
    for w in test:  # no test well has RHOB
        w.curves.pop("RHOB", None)
    assert any("no test well" in e for e in td.check_def(ok, wells=(train, test))[0])
    huge = td.load_def(write(tmp_path, "huge", good(name="huge", target="DTC", scenario={"type": "gap", "gap_length_m": 1e6})))
    assert any("no test well" in e for e in td.check_def(huge, wells=(train, test))[0])


@pytest.mark.parametrize("dataset", ["force2020", "volve"])
def test_check_def_skips_data_checks_when_dataset_missing(tmp_path, monkeypatch, dataset):
    monkeypatch.setattr(td.paths, "get_dataset_path", lambda name: tmp_path / name)
    monkeypatch.setattr(td, "_load_wells", lambda tdef: (_ for _ in ()).throw(FileNotFoundError("no data")))
    d = td.load_def(write(tmp_path, "f", good(name="f", dataset=dataset)))
    errors, warnings = td.check_def(d)
    assert errors == [] and warnings == ["skipped well-presence check: no data"]


def test_multi_gap_placement_is_disjoint_and_deterministic():
    well = synthetic.make_wells(2, 1500, 0)[0]
    y = well.curves["RHOB"]
    a = td.place_gaps(well.depth, y, 50, 3, 7, well.name, "RHOB")
    assert a == td.place_gaps(well.depth, y, 50, 3, 7, well.name, "RHOB") and len(a) == 3
    assert all(a[i][1] < a[i + 1][0] for i in range(2))
    from wmb import splits
    assert td.place_gaps(well.depth, y, 50, 1, 7, well.name, "RHOB") == [
        splits.place_gap(well.depth, y, 50, splits.gap_seed(7, well.name, "RHOB", 50))]


def test_multi_gap_placement_retries_if_first_gap_blocks_the_rest():
    depth = np.arange(11, dtype=float)
    target = np.ones_like(depth)
    seed = next(s for s in range(32) if td.place_gaps(depth, target, 4, 1, s, "w", "RHOB")[0][0] not in (0, 6))
    assert td.place_gaps(depth, target, 4, 2, seed, "w", "RHOB") == [(0, 5), (6, 11)]


def test_runner_with_tmp_def_writes_test_field(tmp_path):
    p = write(tmp_path, "tmp_gap", good(name="tmp_gap", target="DTC", scenario={"type": "gap", "gap_length_m": 50, "n_gaps": 2, "seed": 5}))
    r = run_cli(tmp_path, "--model", "template_linear", "--test", str(p))
    assert r.returncode == 0, r.stderr
    out = tmp_path / "res/template_linear/synthetic/DTC/gap_50/tmp_gap"
    scores = json.loads((out / "scores.json").read_text())
    pred = json.loads((out / "predictions.json").read_text())
    assert scores["test"] == "tmp_gap" and pred["test"] == "tmp_gap" and "[tmp_gap]" in r.stdout
    assert all(len(w["gaps"]) == 2 for w in pred["wells"])


@pytest.mark.parametrize("length", [23, 50.5])
def test_runner_honors_gap_lengths_outside_the_dataset_spec(tmp_path, length):
    key = f"gap_{length}"
    p = write(tmp_path, "custom", good(name="custom", scenario={"type": "gap", "gap_length_m": length}))
    r = run_cli(tmp_path, "--model", "template_linear", "--test", str(p), "--scenario", key)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "res" / "template_linear" / "synthetic" / "RHOB" / key / "custom"
    pred = json.loads((out / "predictions.json").read_text())
    assert pred["wells"]
    for well in pred["wells"]:
        [(top, bottom)] = well["gaps"]
        assert length <= bottom - top < length + 0.151
    assert not (tmp_path / "tasks" / "synthetic").exists()


def test_runner_shipped_synthetic_def_and_old_cli_names(tmp_path):
    r = run_cli(tmp_path, "--model", "template_linear", "--test", "synthetic_rhob_missing", "--dataset", "synthetic")
    assert r.returncode == 0, r.stderr
    base = tmp_path / "res/template_linear/synthetic/RHOB"
    new = json.loads((base / "missing_curve/synthetic_rhob_missing/scores.json").read_text())
    assert new["test"] == "synthetic_rhob_missing"
    assert not (base / "missing_curve/scores.json").exists() and not (base / "scores.json").exists()
    r = run_cli(tmp_path, "--model", "template_linear", "--dataset", "synthetic", "--target", "DTC", "--scenario", "gap_50")
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "res/template_linear/synthetic/DTC/gap_50/scores.json").read_text())
    assert s["test"] == "synthetic_dtc_gap_50"


def test_test_with_conflicting_flag_fails(tmp_path):
    assert run_cli(tmp_path, "--test", "synthetic_rhob_missing", "--target", "DTC").returncode == 2
    assert run_cli(tmp_path, "--test", "synthetic_rhob_missing", "--scenario", "gap_50").returncode == 2
    assert run_cli(tmp_path, "--test", "synthetic_rhob_missing", "--dataset", "force2020").returncode == 2
    assert run_cli(tmp_path, "--test", "nope").returncode == 2


def test_list_tests():
    r = subprocess.run([sys.executable, "-m", "benchmark.run", "--list-tests"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and "rhob_gap50" in r.stdout and "_template" not in r.stdout


def test_test_all_isolates_failing_defs(tmp_path, monkeypatch, capsys):
    from benchmark import run as runner

    real = runner.load_dataset

    def fake(name, seed):
        if name == "force2020":
            raise FileNotFoundError("no data")
        return real(name, seed)

    monkeypatch.setattr(runner, "load_dataset", fake)
    monkeypatch.setattr(td, "DEFS_DIR", tmp_path)
    monkeypatch.setenv("WMB_TASK_CACHE", str(tmp_path / "tasks"))
    out = tmp_path / "res"
    monkeypatch.setenv("WMB_RESULTS", str(out))
    write(tmp_path, "ok", good(name="ok"))
    write(tmp_path, "bad", good(name="bad", dataset="force2020"))
    (tmp_path / "broken.yaml").write_text("target: [", encoding="utf-8")
    (tmp_path / "bad_date.yaml").write_text("version: 2020-99-99", encoding="utf-8")
    assert runner.main(["--list-tests"]) == 0
    listed = capsys.readouterr()
    assert "ok" in listed.out and "skipping test broken" in listed.err and "skipping test bad_date" in listed.err
    with pytest.raises(SystemExit) as exc:
        runner.main(["--test", "broken"])
    assert exc.value.code == 2
    assert runner.main(["--model", "template_linear", "--test", "all"]) == 0
    assert "skipping test broken" in capsys.readouterr().err
    [receipt] = (out / "runs").glob("*.json")
    rows = json.loads(receipt.read_text())
    by = {r["test"]: r for r in rows}
    assert "error" in by["bad"] and "error" not in by["ok"] and by["ok"]["test"] == "ok"
    assert json.loads((out / "template_linear/force2020/RHOB/missing_curve/bad/failure.json").read_text())["test"] == "bad"


def test_scoreboard_includes_test_column(tmp_path, monkeypatch):
    from benchmark import scoreboard

    base = tmp_path / "m/synthetic/RHOB/missing_curve"
    base.mkdir(parents=True)
    row = {"contender": "m", "dataset": "synthetic", "target": "RHOB", "scenario": "missing_curve", "MAE": 1.0}
    (base / "scores.json").write_text(json.dumps({**row, "test": "mine"}))
    old = tmp_path / "m/synthetic/DTC"
    old.mkdir()
    (old / "scores.json").write_text(json.dumps({**row, "target": "DTC"}))  # pre-"test" file
    per_test = base / "mine"  # per-test layout, same result as the classic file: not double-counted
    per_test.mkdir()
    (per_test / "scores.json").write_text(json.dumps({**row, "test": "mine"}))
    new = tmp_path / "m/synthetic/RHOB/gap_50/other"
    new.mkdir(parents=True)
    (new / "scores.json").write_text(json.dumps({**row, "scenario": "gap_50", "test": "other"}))
    monkeypatch.setenv("WMB_RESULTS", str(tmp_path))
    rows = scoreboard.collect()
    assert sorted(r["test"] for r in rows) == ["mine", "other", "synthetic_dtc_missing"]
    assert "test" in scoreboard.COLS and "| test |" in scoreboard.to_markdown(rows)
