"""Station wrappers: registration, contract, abstention, uncertainty and leakage checks. Synthetic only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark import make_tasks as mt  # noqa: E402
from benchmark.registry import discover  # noqa: E402
from wmb import synthetic  # noqa: E402
from wmb.contract import Task, validate_prediction  # noqa: E402
from wmb.io import Well  # noqa: E402
from wmb.model_spec import load_model_spec  # noqa: E402
from workstreams.black_physics import contender as reg  # noqa: E402
from workstreams.black_physics.models._shared import calibration as cal  # noqa: E402
from workstreams.black_physics.models.castagna_mudrock.equation import mudrock_dts  # noqa: E402

WS = REPO / "workstreams" / "black_physics"
IDS = {"black_gardner": "RHOB", "black_gardner_fixed": "RHOB", "black_castagna_dts": "DTS",
       "black_gc_brine_gr": "DTS", "black_wyllie_nphi": "DTC", "black_density_nphi": "RHOB",
       "black_vpvs_ratio": "DTS"}
FACTORIES = {cls().name: cls for cls in reg.CONTENDERS}
FITTED = {"black_gardner", "black_gc_brine_gr", "black_vpvs_ratio"}


@pytest.fixture(scope="module")
def wells():
    w = synthetic.make_wells(n_wells=5, n_samples=500, seed=3)
    return w[:4], w[4]


def fitted(name, train):
    m = FACTORIES[name]()
    m.fit(train, IDS[name])
    return m


def tasks_for(well, target):
    spec = {"dataset": "synthetic", "seed": 2020, "targets": [target], "gap_lengths_m": [10, 50]}
    return mt.build_tasks(spec, [well])[0]


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


# --- registration ----------------------------------------------------------------------------------

def test_registry_lists_the_seven_ids_with_metadata():
    found = discover()
    assert set(IDS) <= set(found) and set(FACTORIES) == set(IDS)
    for name, target in IDS.items():
        m = found[name]()                                         # no-argument default constructor
        assert m.workstream == "black_physics" and m.targets == [target]
        assert m.declared_required_inputs() and set(m.declared_required_inputs()) <= {"DTC", "GR", "NPHI"}
        assert m.licence_name == "MIT" and m.licence_url and m.licence_restrictions
        card = WS / m.card
        assert card.is_file() and name in card.read_text(encoding="utf-8")
        assert (card.parent / "LICENSE.md").is_file()
        spec = load_model_spec(m)                                 # <name>.model.yaml agrees with the class
        assert spec.source == "yaml" and not spec.errors and not spec.warnings, spec.errors + spec.warnings
        assert spec.path.parent == card.parent and spec.licence == m.licence_name
        assert list(spec.targets) == m.targets
        assert spec.targets[target].required == m.declared_required_inputs() and spec.targets[target].optional == []
    assert (WS / "LICENSE.md").is_file()


def test_existing_imports_still_work(wells):
    from workstreams.black_physics.contender import Gardner, MudrockDTS
    train, test = wells
    g = Gardner()
    g.fit(train, "RHOB")
    assert np.isfinite(g.a) and g.b > 0 and np.isfinite(g.sigma)
    assert g.predict(test.without("RHOB"), "RHOB").mean.shape == test.depth.shape
    np.testing.assert_array_equal(MudrockDTS._dts(np.array([100.0, 300.0])), mudrock_dts([100.0, 300.0]))


def test_unsupported_target_is_rejected(wells):
    with pytest.raises(ValueError):
        FACTORIES["black_gardner"]().fit(wells[0], "DTS")


# --- contract on both scenarios --------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(IDS))
def test_both_scenarios_validate_and_batch_matches(name, wells):
    train, test = wells
    m = fitted(name, train)
    tasks = tasks_for(test, IDS[name])
    assert {t.scenario for t in tasks} == {"missing_curve", "gap"}
    single = [m.predict_task(t) for t in tasks]
    for t, p in zip(tasks, single):
        ok = validate_prediction(t, p)
        assert len(ok.p50) == len(t.depth) and np.isfinite(ok.p50[t.predict_mask]).any()
        assert p.std is not None                                  # synthetic training gives a scale
        np.testing.assert_array_equal(np.isfinite(p.std), np.isfinite(p.mean))
        assert (p.std[np.isfinite(p.std)] > 0).all()
    for p, b in zip(single, m.predict_batch(tasks)):
        assert same(p.mean, b.mean) and same(p.std, b.std)


@pytest.mark.parametrize("name", sorted(IDS))
def test_refit_on_one_instance_is_stateless(name, wells):
    train, test = wells
    m = FACTORIES[name]()
    m.fit(train, IDS[name])
    first = m.predict(test.without(IDS[name]), IDS[name])
    m.fit(train[:2], IDS[name])
    m.fit(train, IDS[name])
    again = m.predict(test.without(IDS[name]), IDS[name])
    assert same(first.mean, again.mean) and same(first.std, again.std)


@pytest.mark.parametrize("name", sorted(IDS))
def test_hidden_truth_flanks_names_and_row_labels_do_not_change_predictions(name, wells):
    train, test = wells
    target = IDS[name]
    m = fitted(name, train)
    for t in tasks_for(test, target):
        ref = m.predict_task(t)
        inputs = t.inputs.copy()
        inputs.index = pd.RangeIndex(1000, 1000 + len(inputs))[::-1]
        observed = None if t.target_observed is None else t.target_observed * 1.7 + 3.0
        twin = Task("RENAMED-WELL", t.depth.copy(), inputs, target, t.predict_mask.copy(), t.scenario,
                    t.gap_length_m, observed, "other_dataset", "hidden", {"note": "changed"})
        got = m.predict_task(twin)
        assert same(ref.mean, got.mean) and same(ref.std, got.std)
    # changing the withheld target values themselves leaves the inputs, and so the predictions, unchanged
    altered = Well(test.name, test.depth, {**test.curves, target: test.curves[target] * 0.5 + 1}, test.dataset)
    for a, b in zip(tasks_for(test, target), tasks_for(altered, target)):
        assert same(m.predict_task(a).mean, m.predict_task(b).mean)


@pytest.mark.parametrize("name", sorted(IDS))
def test_only_declared_inputs_are_read(name, wells):
    train, test = wells
    m = fitted(name, train)
    declared = m.declared_required_inputs()
    only = Well("X", test.depth, {c: test.curves[c] for c in declared})
    noisy = Well("Y", test.depth, {c: (test.curves[c] if c in declared else np.full(len(test.depth), -1e9))
                                   for c in ("GR", "DTC", "NPHI", "RHOB", "DTS", "RDEP")})
    target = IDS[name]
    assert same(m.predict(only, target).mean, m.predict(noisy, target).mean)


# --- abstention ------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(IDS))
def test_missing_and_invalid_inputs_abstain_locally(name, wells):
    train, test = wells
    target = IDS[name]
    m = fitted(name, train)
    n = 40
    depth = np.arange(2000, 2000 + n)                             # integer depth array
    assert depth.dtype.kind == "i"
    empty = m.predict(Well("E", depth, {}), target)              # missing columns
    assert empty.mean.shape == (n,) and np.isnan(empty.mean).all()
    nan_curves = {c: np.full(n, np.nan) for c in m.declared_required_inputs()}
    assert np.isnan(m.predict(Well("N", depth, nan_curves), target).mean).all()

    good = {"DTC": np.full(n, 90.0), "GR": np.full(n, 70.0), "NPHI": np.full(n, 0.2)}
    bad_rows = [3, 4, 5, 6, 7]
    for c, bad in (("DTC", [0.0, -90.0, np.inf, -np.inf, np.nan]), ("NPHI", [-0.05, 1.5, np.inf, -np.inf, np.nan]),
                   ("GR", [np.nan, np.inf, -np.inf, np.nan, np.nan])):
        if c not in m.declared_required_inputs():
            continue
        curves = {k: v.copy() for k, v in good.items() if k in m.declared_required_inputs()}
        curves[c][bad_rows] = bad
        p = m.predict(Well("B", depth, curves), target)
        assert np.isnan(p.mean[bad_rows]).all(), c
        keep = np.setdiff1d(np.arange(n), bad_rows)
        assert np.isfinite(p.mean[keep]).all(), c                  # neighbours keep their predictions
        assert np.isfinite(p.std[keep]).all() and np.isnan(p.std[bad_rows]).all()
        assert curves[c][bad_rows[1]] == bad[1]                    # the source log is not modified
    validate_prediction(Task("B", depth.astype(float), pd.DataFrame({"DEPTH": depth}), target, np.ones(n, bool),
                             "missing_curve"), p)


def test_negative_nphi_is_not_used_as_porosity():
    for cls in (FACTORIES["black_wyllie_nphi"], FACTORIES["black_density_nphi"]):
        m = cls()                                                 # fixed equation, no training wells
        p = m.predict(Well("W", np.arange(3.0), {"NPHI": np.array([-0.02, 0.0, 1.0])}), m.targets[0])
        assert np.isnan(p.mean[0]) and np.isfinite(p.mean[1:]).all() and p.std is None


# --- uncertainty -----------------------------------------------------------------------------------

def well(name, dtc, dts=None, **extra):
    curves = {"DTC": np.asarray(dtc, float), **{k: np.asarray(v, float) for k, v in extra.items()}}
    if dts is not None:
        curves["DTS"] = np.asarray(dts, float)
    return Well(name, np.arange(len(dtc), dtype=float), curves)


def test_mudrock_scale_survives_an_invalid_transform():
    dtc = np.array([80.0, 100.0, 120.0, 190.0])                   # 190 us/ft: mudrock Vs <= 0.3, no prediction
    train = [well(f"W{i}", dtc, mudrock_dts(np.r_[dtc[:3], 150.0]) + 5 * (i + 1), GR=np.full(4, 60.0))
             for i in range(3)]
    m = FACTORIES["black_castagna_dts"]()
    m.fit(train, "DTS")
    assert np.isnan(mudrock_dts(190.0))
    assert m.calibration.scale == pytest.approx(np.sqrt(np.mean([25.0, 100.0, 225.0])))
    assert m.calibration.n_wells == 3 and m.calibration.n_rows == 9
    p = m.predict(well("T", dtc), "DTS")
    assert np.isfinite(p.std[:3]).all() and np.isnan(p.std[3]) and np.isnan(p.mean[3])


def test_scale_gives_wells_equal_weight():
    c = cal.summarise([np.array([1.0]), np.full(100, 3.0)], "test")
    assert c.scale == pytest.approx(np.sqrt(5.0))                 # not the row-pooled sqrt(901/101)
    assert cal.summarise([np.array([2.0]), np.array([])], "test").scale is None
    assert cal.summarise([np.zeros(3), np.zeros(4)], "test").scale is None


def test_residual_mask_needs_positive_finite_truth_and_finite_prediction():
    r = cal.residuals(np.array([1.0, np.nan, -1.0, 0.0, 2.0, 3.0]), np.array([1.5, 1.0, 1.0, 1.0, np.nan, np.inf]))
    np.testing.assert_array_equal(r, [0.5])


def test_whole_well_folds_are_deterministic_and_disjoint():
    names = [f"W{i:02d}" for i in range(12)]
    a = cal.whole_well_folds(names)
    b = cal.whole_well_folds(list(reversed(names)))
    assert len(a) == 5 and sorted(np.concatenate(a)) == list(range(12))
    assert [sorted(names[i] for i in f) for f in a] == [sorted(names[::-1][i] for i in f) for f in b]
    assert len(cal.whole_well_folds(names[:3])) == 3 and cal.whole_well_folds([]) == []


def test_ratio_uncertainty_is_out_of_fold():
    train = [well("A", np.full(5, 100.0), np.full(5, 160.0)), well("B", np.full(5, 100.0), np.full(5, 200.0))]
    m = FACTORIES["black_vpvs_ratio"]()
    m.fit(train, "DTS")
    assert m.params.ratio == pytest.approx(1.8) and m.params.n_wells == 2
    assert m.calibration.scale == pytest.approx(40.0)              # in-sample would give 20
    assert "out-of-fold" in m.calibration.method


def test_one_contributing_well_gives_point_predictions():
    one = [well("A", np.linspace(80, 120, 20), np.linspace(150, 220, 20), GR=np.linspace(30, 120, 20))]
    for name in ("black_castagna_dts", "black_gc_brine_gr", "black_vpvs_ratio"):
        m = FACTORIES[name]()
        m.fit(one, "DTS")
        p = m.predict(one[0].without("DTS"), "DTS")
        assert p.std is None and np.isfinite(p.mean).any(), name


def test_fixed_equations_predict_with_zero_training_wells():
    for name in ("black_gardner_fixed", "black_castagna_dts", "black_wyllie_nphi", "black_density_nphi"):
        m = FACTORIES[name]()
        m.fit([], IDS[name])
        p = m.predict(well("T", np.full(4, 90.0), NPHI=np.full(4, 0.2)), IDS[name])
        assert np.isfinite(p.mean).all() and p.std is None, name


# --- fitted parameters -----------------------------------------------------------------------------

def test_gardner_fallback_matches_fixed_when_underdetermined():
    tiny = [well("A", np.linspace(80, 120, 5), RHOB=np.linspace(2.2, 2.4, 5)),
            well("B", np.linspace(80, 120, 5), RHOB=np.linspace(2.2, 2.4, 5))]
    g, f = FACTORIES["black_gardner"](), FACTORIES["black_gardner_fixed"]()
    g.fit(tiny, "RHOB")
    f.fit(tiny, "RHOB")
    assert not g.params.fitted and (g.a, g.b) == (0.31, 0.25)
    x = well("T", np.array([70.0, 100.0, 150.0]))
    np.testing.assert_allclose(g.predict(x, "RHOB").mean, f.predict(x, "RHOB").mean)


def test_vpvs_without_paired_training_wells_is_unsupported():
    m = FACTORIES["black_vpvs_ratio"]()
    m.fit([well("A", np.full(5, 100.0)), well("B", np.full(5, 90.0), np.full(5, np.nan))], "DTS")
    p = m.predict(well("T", np.full(5, 100.0)), "DTS")
    assert m.params.ratio is None and np.isnan(p.mean).all() and p.std is None


def test_gc_endpoints_fitted_explicit_and_degenerate():
    gr_only = well("G", np.full(101, 100.0), GR=np.arange(101.0))     # no DTS: endpoints still use its GR
    with_dts = [well(f"S{i}", np.full(101, 100.0), np.full(101, 190.0 + i), GR=np.arange(101.0) + 20)
                for i in range(2)]
    m = FACTORIES["black_gc_brine_gr"]()
    m.fit([gr_only, *with_dts], "DTS")
    assert (m.params.gr_low, m.params.gr_high, m.params.n_wells) == (pytest.approx(30.0), pytest.approx(110.0), 3)
    assert "median over 3 training wells" in m.params.source

    explicit = FACTORIES["black_gc_brine_gr"](gr_low=25.0, gr_high=140.0, endpoint_source="core-calibrated, field X")
    explicit.fit(with_dts, "DTS")
    assert explicit.params.gr_low == 25.0 and explicit.params.source == "core-calibrated, field X"
    assert "training-well residuals" in explicit.calibration.method
    with pytest.raises(ValueError):
        FACTORIES["black_gc_brine_gr"](gr_low=25.0)

    flat = FACTORIES["black_gc_brine_gr"]()
    flat.fit([well(f"F{i}", np.full(20, 100.0), np.full(20, 190.0), GR=np.full(20, 60.0)) for i in range(3)], "DTS")
    assert not flat.params.usable
    p = flat.predict(well("T", np.full(5, 100.0), GR=np.full(5, 60.0)), "DTS")
    assert np.isnan(p.mean).all() and p.std is None


def test_describe_records_parameters_without_well_data(wells):
    for name in IDS:
        d = fitted(name, wells[0]).describe()
        assert d["id"] == name and d["calibration"]["n_wells"] >= 2 and d["params"] is not None
        assert d["required_inputs"] and d["card"].endswith("README.md")
