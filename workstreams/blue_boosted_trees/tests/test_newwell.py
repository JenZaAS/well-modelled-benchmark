"""Tests for saving models and predicting a new well. Run: python -m pytest workstreams/blue_boosted_trees/tests"""
import io

import lasio
import numpy as np
import pandas as pd
import pytest

from workstreams.blue_boosted_trees.ml_study import newwell, store
from workstreams.blue_boosted_trees.ml_study.data import UNKNOWN
from workstreams.blue_boosted_trees.ml_study.models import MLContender
from wmb import synthetic
from wmb.io import Well

KNOWN = {"group": ["NORDLAND GP.", "VIKING GP."], "formation": ["Utsira Fm.", "Draupne Fm.", "BASEMENT"]}


def _las_bytes(depth, curves, well="TEST 1/1", unit="M") -> bytes:
    las = lasio.LASFile()
    las.well["WELL"].value = well
    las.append_curve("DEPT", depth, unit=unit)
    for mnemonic, values in curves.items():
        las.append_curve(mnemonic, values)
    out = io.StringIO()
    las.write(out)
    return out.getvalue().encode("utf-8")


def _tops(rows):
    return pd.DataFrame(rows, columns=["name", "md"])


def test_load_las_maps_mnemonics_and_names_the_well():
    depth = np.arange(1000.0, 1100.0, 0.5)
    data = _las_bytes(depth, {"GR": np.linspace(20, 120, depth.size), "RHOZ": np.full(depth.size, 2.4),
                              "LLD": np.full(depth.size, 10.0), "XYZ": np.ones(depth.size)})
    well, mapping = newwell.load_las(data)
    assert well.name == "TEST 1/1" and mapping == {"GR": "GR", "RHOB": "RHOZ", "RDEP": "LLD"}
    assert set(well.curves) == {"GR", "RHOB", "RDEP"} and well.depth[0] == 1000.0
    assert newwell.sample_spacing(well) == pytest.approx(0.5)


def test_load_las_converts_feet_and_sorts_depth():
    depth_ft = np.arange(3000.0, 3100.0, 1.0)[::-1]                      # decreasing depth, in feet
    well, _ = newwell.load_las(_las_bytes(depth_ft, {"GR": depth_ft.copy()}, unit="FT"))
    assert np.all(np.diff(well.depth) > 0) and well.depth[0] == pytest.approx(3000 * 0.3048)
    assert well.curves["GR"][0] == 3000.0                                 # curve values follow their depths


def test_well_warnings_flag_wrong_units_and_odd_spacing():
    ranges = {"RHOB": [1.5, 2.9], "GR": [5.0, 200.0]}
    depth = np.arange(0.0, 100.0, 0.152)
    ok = Well("w", depth, {"RHOB": np.full(depth.size, 2.4), "GR": np.full(depth.size, 80.0)})
    assert newwell.well_warnings(ok, ranges) == []
    bad = Well("w", depth, {"RHOB": np.full(depth.size, 2400.0)})         # kg/m3 instead of g/cm3
    assert any("RHOB" in w and "unit" in w for w in newwell.well_warnings(bad, ranges))
    assert any("None of the 9" in w for w in newwell.well_warnings(Well("w", depth, {}), ranges))
    sparse = Well("w", np.arange(0.0, 500.0, 5.0), {"GR": np.full(100, 80.0)})
    assert any("spacing" in w for w in newwell.well_warnings(sparse, ranges))


def test_tops_become_labels_down_to_the_next_top():
    depth = np.arange(0.0, 100.0, 1.0)
    tops = _tops([("viking gp. Top", 60.0), ("NORDLAND GP. Top", 10.0), ("Utsira Fm. Top", 20.0),
                  ("Draupne Fm. Top", 70.0), ("Mystery Fm. Top", 90.0)])
    labels, unmatched = newwell.tops_to_labels(depth, tops, KNOWN)
    assert labels["group"][5] == UNKNOWN and labels["group"][10] == "NORDLAND GP."     # above the first top: unknown
    assert labels["group"][59] == "NORDLAND GP." and labels["group"][60] == "VIKING GP."  # case matched to the training label
    assert labels["formation"][19] == UNKNOWN and labels["formation"][20] == "Utsira Fm."
    assert labels["formation"][70] == "Draupne Fm." and labels["formation"][95] == "Mystery Fm."
    assert unmatched == ["Mystery Fm."]                                                 # not a training label


def test_read_tops_picks_the_well_and_the_npd_columns():
    table = pd.DataFrame({"Well identifier": ["15/9-13"] * 2 + ["7/1-2 S"] * 2,
                          "Surface": ["VIKING GP. Top", "Utsira Fm. Top", "Sola Fm. Top", "BASEMENT Top"],
                          "MD": [2741.0, 869.0, 2828.0, 3000.0], "Interpreter": ["NPD"] * 4})
    buf = io.BytesIO()
    table.to_excel(buf, index=False)
    tops = newwell.read_tops(buf.getvalue(), "tops.xlsx", "15/9-13")
    assert list(tops["name"]) == ["VIKING GP. Top", "Utsira Fm. Top"] and list(tops["md"]) == [2741.0, 869.0]
    with pytest.raises(ValueError, match="several wells"):
        newwell.read_tops(buf.getvalue(), "tops.xlsx", "other well")
    csv = "Surface;MD\nVIKING GP. Top;2741\n".encode()
    assert len(newwell.read_tops(csv, "tops.csv")) == 1                               # one well, no well column needed
    with pytest.raises(ValueError, match="name column"):
        newwell.read_tops(b"a,b\n1,2\n", "x.csv")


def test_well_matching_uses_the_id_at_the_start_of_the_las_name():
    ids = ["15/9-1", "15/9-13", "15/9-23", "7/1-2 S"]
    assert newwell._match_well(ids, "15/9-23 Skardkollen") == "15/9-23"
    assert newwell._match_well(ids, "15/9-13 Skardkollen") == "15/9-13"             # not 15/9-1
    assert newwell._match_well(ids, "7/1-2 S Field") == "7/1-2 S"
    assert newwell._match_well(ids, "15_9-23") == "15/9-23"                          # exact after ignoring / _ -
    assert newwell._match_well(ids, "30/6-5") is None and newwell._match_well(ids, None) is None


def test_saved_model_predicts_the_same_after_reloading(tmp_path):
    wells = synthetic.make_wells(n_wells=4, n_samples=300, seed=2)
    for w in wells:
        w.meta["group"], w.meta["formation"] = np.full(300, "NORDLAND GP."), np.full(300, "Utsira Fm.")
    model = MLContender("gbt", use_strat=True, max_train_rows=500, trend=True)
    model.fit(wells[:3], "RHOB")
    path = store.save_model(tmp_path, model, "RHOB")
    assert path.name == "gbt_trend_strat.joblib" and store.list_models(tmp_path) == {"RHOB": ["gbt_trend_strat"]}
    expected = model.predict(wells[3].without("RHOB"), "RHOB").mean
    reloaded = newwell.predict_log(tmp_path, wells[3], "RHOB", "gbt_trend_strat")     # also ignores the well's own RHOB
    np.testing.assert_allclose(reloaded, expected)
    with pytest.raises(FileNotFoundError):
        store.load_model(tmp_path, "RHOB", "rf")


def test_manifest_and_version_warnings(tmp_path):
    store.write_manifest(tmp_path, {"trained_on": "test"})
    assert store.read_manifest(tmp_path)["trained_on"] == "test" and store.version_warnings(store.read_manifest(tmp_path)) == []
    old = {"versions": {"scikit-learn": "0.0.1"}}
    assert "scikit-learn" in store.version_warnings(old)[0]
    assert store.list_models(tmp_path / "missing") == {}


def test_sparse_depths_are_masked():
    depth = np.arange(0.0, 6.0)
    well = Well("w", depth, {"GR": np.ones(6), "RHOB": np.array([1, 1, 1, np.nan, np.nan, np.nan]),
                             "NPHI": np.array([1, 1, np.nan, np.nan, np.nan, np.nan]), "DTC": np.full(6, np.nan)})
    counts = newwell.input_counts(well, "GR")                    # the target itself is not an input
    assert counts.tolist() == [2, 2, 1, 0, 0, 0]
    assert newwell.input_counts(well, "RHOB").tolist() == [2, 2, 1, 1, 1, 1]
    masked = newwell.mask_sparse(np.arange(6.0), counts, min_inputs=2)
    assert masked[:2].tolist() == [0.0, 1.0] and np.isnan(masked[2:]).all()
    assert newwell.input_counts(Well("w", depth, {}), "GR").tolist() == [0] * 6


def test_usable_variants_and_filled_curve():
    saved = {"GR": ["gbt", "gbt_strat", "linear_trend_strat"]}
    assert newwell.usable_variants(saved, "GR", with_strat=False) == ["gbt"]
    assert newwell.usable_variants(saved, "GR", with_strat=True) == saved["GR"]
    assert newwell.usable_variants(saved, "DTS", with_strat=True) == []
    filled = newwell.filled_curve(np.array([1.0, np.nan, 3.0]), np.array([9.0, 9.0, 9.0]))
    assert filled.tolist() == [1.0, 9.0, 3.0]
    assert newwell.filled_curve(None, np.array([9.0])).tolist() == [9.0]
