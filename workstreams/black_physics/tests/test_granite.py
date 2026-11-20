"""Granite gap experiment: geometry, splits, leakage, fallbacks, intervals and decisions. NumPy only, synthetic.

The real-model checks (checkpoint changes, TSPulse output units, label perturbation through the actual
networks) are in test_granite_torch.py and skip without the Granite environment.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from wmb.contract import Task, validate_prediction  # noqa: E402
from wmb.io import Well  # noqa: E402
from workstreams.black_physics.granite import calibrate as CAL  # noqa: E402
from workstreams.black_physics.granite import decision as DEC  # noqa: E402
from workstreams.black_physics.granite import features as F  # noqa: E402
from workstreams.black_physics.granite import gaps as G  # noqa: E402
from workstreams.black_physics.granite.reconstruct import TSPulseReconstructor, TTMReconstructor  # noqa: E402
from workstreams.black_physics.validate_force import linear_gap_reference  # noqa: E402

STEP = 0.152


def well(n=3000, seed=0, name="W-1", nan_runs=(), jump_at=None):
    rng = np.random.default_rng(seed)
    depth = 1000 + STEP * np.arange(n)
    if jump_at is not None:
        depth[jump_at:] += 40.0                               # an acquisition break
    y = 2.3 + 0.1 * np.sin(depth / 7) + rng.normal(0, 0.01, n)
    for a, b in nan_runs:
        y[a:b] = np.nan
    dtc = 304800 / ((y / 0.31) ** 4)                          # exact Gardner inverse
    return Well(name, depth, {"RHOB": y, "DTC": dtc}, "synthetic", "train")


def task_for(w, a, b, target="RHOB"):
    gap = G._make_gap(w.name, "B", 0, w.depth, a, b)
    return G.gap_task(w, gap)


# ------------------------------------------------------------------ fake backends
def const_forecast(contexts):
    """Each context forecasts its own last value for 336 steps, at all nine quantiles (+-0.01 spread)."""
    out = np.zeros((len(contexts), 9, 336))
    for i, c in enumerate(contexts):
        out[i] = c[-1] + np.linspace(-0.04, 0.04, 9)[:, None]
    return out


def step_forecast(contexts):
    """Median = context's last value + 0.001 * step; lets a test see which end each forecast starts from."""
    out = np.zeros((len(contexts), 9, 336))
    for i, c in enumerate(contexts):
        out[i] = c[-1] + 0.001 * np.arange(1, 337)[None, :] + np.linspace(-0.04, 0.04, 9)[:, None]
    return out


def mean_reconstruct(windows):
    return np.tile(np.nanmean(windows, axis=1, keepdims=True), (1, windows.shape[1]))


# ------------------------------------------------------------------ splits
def test_split_is_disjoint_deterministic_and_order_free():
    names = [f"15/{i}-{i % 7}" for i in range(98)]
    a = G.split_wells(names, {"A": 54, "B": 12, "C": 20, "D": 12}, 2026)
    b = G.split_wells(list(reversed(names)), {"A": 54, "B": 12, "C": 20, "D": 12}, 2026)
    assert a == b
    assert [len(a[g]) for g in G.GROUPS] == [54, 12, 20, 12]
    assert len(set().union(*map(set, a.values()))) == 98
    c = G.split_wells(names, {"A": 54, "B": 12, "C": 20, "D": 12}, 7)
    assert c != a                                            # the seed matters
    with pytest.raises(ValueError):
        G.split_wells(names[:-1], {"A": 54, "B": 12, "C": 20, "D": 12}, 2026)
    with pytest.raises(ValueError):
        G.assert_disjoint({"A": ["x"], "B": ["x"]})


# ------------------------------------------------------------------ masks
def test_evaluation_gaps_are_reproducible_non_overlapping_and_measured():
    w = well(8000, nan_runs=[(3000, 3050)])
    kw = dict(group="C", n_max=8, overlap=False, seed=2026, length_m=50.0)
    g1 = G.index_gaps(w.name, w.depth, w.curves["RHOB"], **kw)
    g2 = G.index_gaps(w.name, w.depth, w.curves["RHOB"], **kw)
    assert g1 == g2 and len(g1) == 8
    for x, y in zip(g1, g1[1:]):
        assert y.start > x.end                                # at least one sample between gaps
    for g in g1:
        assert np.isfinite(w.curves["RHOB"][g.start:g.end]).all()
        assert w.depth[g.end - 1] - w.depth[g.start] >= 50.0 - 1e-6
        assert w.depth[g.end - 2] - w.depth[g.start] < 50.0
    many = G.index_gaps(w.name, w.depth, w.curves["RHOB"], group="A", n_max=32, overlap=True, seed=2026, length_m=50.0)
    assert len({g.start for g in many}) == len(many) == 32
    short = well(400)                                         # 60 m of log: only one 50 m gap fits
    assert len(G.index_gaps("S", short.depth, short.curves["RHOB"], **kw)) == 1


def test_gaps_never_cross_an_acquisition_break():
    w = well(4000, jump_at=2000)
    for g in G.index_gaps(w.name, w.depth, w.curves["RHOB"], group="A", n_max=32, overlap=True, seed=1, length_m=50.0):
        assert (g.seg_start, g.seg_end) in ((0, 2000), (2000, 4000))
        assert g.end <= g.seg_end


# ------------------------------------------------------------------ TTM geometry
def test_ttm_reverses_the_deeper_flank_and_blends_by_distance():
    w = well(3000)
    a, b = 1000, 1330
    t = task_for(w, a, b)
    obs = t.target_observed
    p = TTMReconstructor(const_forecast).run([t])[0][0]
    assert p.status == "two_sided"
    # constant directional forecasts -> the blend is the straight line between the boundary values
    np.testing.assert_allclose(p.p50, G.linear_fill(w.depth, obs, a, b, 0, 3000), atol=1e-12)
    q = TTMReconstructor(step_forecast).run([t])[0][0]
    # shallow forecast starts at the shallow boundary; the deep one, reversed, ends next to the deep boundary
    assert q.dir_above[0] == pytest.approx(obs[a - 1] + 0.001)
    assert q.dir_below[-1] == pytest.approx(obs[b] + 0.001)
    assert q.dir_below[0] == pytest.approx(obs[b] + 0.001 * (b - a))
    u = (w.depth[a:b] - w.depth[a - 1]) / (w.depth[b] - w.depth[a - 1])
    np.testing.assert_allclose(q.p50, (1 - u) * q.dir_above + u * q.dir_below)
    assert q.ctx_above_m == pytest.approx(512 * STEP, rel=1e-6) and len(q.ctx_above) == 512


def test_ttm_short_flank_one_sided_fallback_and_abstain():
    w = well(3000)
    t = task_for(w, 100, 430)                                 # 100 samples above < 256: one-sided from below
    p = TTMReconstructor(const_forecast).run([t])[0][0]
    assert p.status == "one_sided_below" and p.dir_above is None and np.isfinite(p.p50).all()
    w2 = well(700)
    t2 = task_for(w2, 120, 450)                               # both flanks < 256 -> straight-line fallback
    p2 = TTMReconstructor(const_forecast).run([t2])[0][0]
    assert p2.status == "fallback_interp"
    np.testing.assert_allclose(p2.p50, linear_gap_reference(t2).p50[120:450])
    t3 = task_for(w, 1000, 1400)                              # 400 samples > 336-step horizon
    assert TTMReconstructor(const_forecast).run([t3])[0][0].status == "fallback_interp"
    w4 = well(1000, nan_runs=[(0, 300)])
    t4 = task_for(w4, 300, 630)                               # one side has no measured value at all
    t4.target_observed[630:] = np.nan
    assert TTMReconstructor(const_forecast).run([t4])[0][0].status == "abstain"


def test_missing_curve_tasks_abstain():
    w = well(800)
    frame = G.inputs_frame(w, "RHOB")
    t = Task(w.name, w.depth, frame, "RHOB", np.ones(800, bool), "missing_curve")
    for rec in (TTMReconstructor(const_forecast), TSPulseReconstructor(mean_reconstruct)):
        p = rec.run([t])[0][0]
        assert p.status == "abstain" and np.isnan(p.p50).all()


# ------------------------------------------------------------------ TSPulse geometry
def test_tspulse_window_keeps_the_gap_masked_and_centred():
    w = well(3000)
    rec = TSPulseReconstructor(mean_reconstruct)
    a, b = 1000, 1330
    vals, w0 = rec.window_values(w.curves["RHOB"].copy(), a, b, 0, 3000)
    assert w0 == a - (512 - 330) // 2
    assert np.isnan(vals[a - w0:b - w0]).all() and np.isfinite(vals[:a - w0]).all() and np.isfinite(vals[b - w0:]).all()
    assert (a - w0, w0 + 512 - b) == (91, 91)
    # near the top of the segment the window shifts down instead of padding
    v2, w2 = rec.window_values(w.curves["RHOB"].copy(), 20, 350, 0, 3000)
    assert w2 == 0 and np.isnan(v2[20:350]).all() and np.isfinite(v2[:20]).all()
    # a segment shorter than the window: padding after the segment end is unobserved
    v3, w3 = rec.window_values(w.curves["RHOB"].copy(), 100, 300, 0, 400)
    assert w3 == 0 and np.isnan(v3[400:]).all()
    seen = {}
    rec2 = TSPulseReconstructor(lambda W: np.full_like(seen.setdefault("w", W.copy()), 2.0))
    p = rec2.run([task_for(w, a, b)])[0][0]
    assert p.status == "two_sided" and np.isnan(seen["w"][0, a - w0:b - w0]).all()
    assert np.allclose(p.p50, 2.0) and len(p.ctx_above) == 91


def test_tspulse_full_window_hole_falls_back():
    w = well(3000)
    t = task_for(w, 1000, 1600)                               # 600 samples >= 512
    p = TSPulseReconstructor(mean_reconstruct).run([t])[0][0]
    assert p.status == "fallback_interp" and np.isfinite(p.p50).all()


# ------------------------------------------------------------------ leakage
def test_withheld_labels_never_reach_inputs_outputs_or_features():
    w = well(3000)
    a, b = 1000, 1330
    seen = []

    def spy(contexts):
        seen.append([c.copy() for c in contexts])
        return step_forecast(contexts)

    t1 = task_for(w, a, b)
    p1 = TTMReconstructor(spy).run([t1])[0][0]
    w2 = well(3000)
    w2.curves["RHOB"][a:b] += 0.5 * np.random.default_rng(3).normal(size=b - a)   # perturb only the hidden truth
    t2 = task_for(w2, a, b)
    p2 = TTMReconstructor(spy).run([t2])[0][0]
    assert all(np.array_equal(x, y) for x, y in zip(seen[0], seen[1]))
    np.testing.assert_array_equal(p1.p50, p2.p50)
    g = np.full(b - a, 2.3)
    f1, f2 = F.gap_features("ttm", p1, g), F.gap_features("ttm", p2, g)
    assert f1 == f2 and F.state_text("ttm", f1) == F.state_text("ttm", f2)
    # even a task whose target_observed still holds the truth on the mask is blanked before use
    t3 = task_for(w, a, b)
    t3.target_observed[a:b] = w.curves["RHOB"][a:b]
    np.testing.assert_array_equal(TTMReconstructor(step_forecast).run([t3])[0][0].p50, p1.p50)


def test_state_text_has_no_identity_or_outcome():
    f = {"width": 0.031, "flank_median_diff": float("nan"), "pred_range": 0.1, "ctx_above_m": 78, "ctx_below_m": 0,
         "gardner_mad": float("nan"), "dtc_fraction": 0.0}
    s = F.state_text("ttm", f)
    assert "W-1" not in s and "error" not in s.lower() and "label" not in s.lower()
    assert "Only one flank" in s and "No sonic-derived density" in s and "0.031 g/cm³" in s


# ------------------------------------------------------------------ intervals
def test_calibrated_bands_are_nested_and_cover_on_the_calibration_set():
    w = well(6000)
    rec = TTMReconstructor(const_forecast)
    gaps = G.index_gaps(w.name, w.depth, w.curves["RHOB"], group="D", n_max=8, overlap=False, seed=1, length_m=50.0)
    tasks = [G.gap_task(w, g) for g in gaps]
    pieces = [ps[0] for ps in rec.run(tasks)]
    recs = [(w.name, p, w.curves["RHOB"][p.a:p.b]) for p in pieces]
    cal = CAL.fit("ttm", recs)
    assert 0 < cal.k50 <= cal.k90
    hit = []
    for p, (_, _, y) in zip(pieces, recs):
        bd = cal.bands(p)
        assert np.all(bd["p05"] <= bd["p25"]) and np.all(bd["p25"] <= p.p50) and np.all(p.p50 <= bd["p75"])
        assert np.all(bd["p75"] <= bd["p95"])
        hit.append((y >= bd["p05"]) & (y <= bd["p95"]))
    assert 0.85 <= np.concatenate(hit).mean() <= 0.95
    assert CAL.weighted_quantile([1, 2, 3, 100], [1, 1, 1, 0], 0.5) == pytest.approx(2.0)


def test_station_wrapper_output_shapes_validate():
    from workstreams.black_physics.models.granite.wrapper import TTMR3Frozen

    w = well(3000)
    m = TTMR3Frozen()
    m.reconstructor = TTMReconstructor(const_forecast)
    m.calibrator = CAL.IntervalCalibrator("ttm", 0.5, 1.5, 0.005, 1, 1, "test")
    t = task_for(w, 1000, 1330)
    pred = validate_prediction(t, m.predict_task(t))
    assert pred.p50.shape == t.depth.shape and np.isnan(pred.p50[~t.predict_mask]).all()
    assert np.isfinite(pred.p05[t.predict_mask]).all()
    assert set(pred.quantiles) == {0.05, 0.25, 0.5, 0.75, 0.95}


# ------------------------------------------------------------------ physics rows
def test_gardner_flank_residual_and_missing_dtc():
    from workstreams.black_physics.models.granite.wrapper import gardner_flank_residual

    w = well(3000)
    t = task_for(w, 1000, 1330)
    out = gardner_flank_residual(t)
    np.testing.assert_allclose(out[1000:1330], w.curves["RHOB"][1000:1330], atol=1e-9)   # exact Gardner world
    assert np.isnan(out[:1000]).all()
    t.inputs["DTC"] = np.nan
    assert np.isnan(gardner_flank_residual(t)).all()
    p = TTMReconstructor(const_forecast).run([task_for(w, 1000, 1330)])[0][0]
    f = F.gap_features("ttm", p, np.full(330, np.nan))
    assert np.isnan(f["gardner_mad"]) and f["dtc_fraction"] == 0.0


# ------------------------------------------------------------------ decisions
def _rows(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        width = rng.uniform(0.01, 0.2)
        rows.append({"key": f"k{i}", "well": f"w{i % 10}", "complete": True, "width": width,
                     "flank_median_diff": rng.uniform(0, 0.2) if i % 5 else float("nan"), "pred_range": rng.uniform(0, 0.3),
                     "ctx_above_m": 78.0, "ctx_below_m": 78.0, "gardner_mad": float("nan") if i % 3 == 0 else 0.05,
                     "dtc_fraction": 0.0 if i % 3 == 0 else 1.0, "label": int(width + rng.normal(0, 0.02) < 0.1)})
    return rows


def test_logistic_rules_learn_direction_and_handle_missing_values():
    rows = _rows()
    r = DEC.fit_rule("numeric_gardner", rows, F.WITH_GARDNER, 1.0)
    assert "flank_median_diff" in r.prep.indicators and "gardner_mad" in r.prep.indicators
    cols = r.prep.columns()
    assert r.beta[cols.index("width")] < 0                                   # wider -> less likely to pass
    p = r.proba(rows)
    y = np.array([x["label"] for x in rows])
    assert np.mean((p - y) ** 2) < np.mean((y.mean() - y) ** 2)
    one = [dict(x, label=1) for x in rows]
    r1 = DEC.fit_rule("numeric", one, F.BASE, 1.0)
    assert r1.constant is not None and 0.5 < r1.constant < 1 and "one observed class" in r1.note
    rec = DEC.fit_recalibration(np.zeros(10), np.ones(10), np.ones(10), 1.0)
    assert rec.constant is not None


def test_decision_metrics_and_tie_aware_risk_coverage():
    p = [0.9, 0.9, 0.85, 0.5, 0.95]
    y = [1, 0, 1, 1, None]
    complete = [True, True, True, True, False]
    m = DEC.decision_metrics(p, y, complete, 0.8)
    assert (m["accepted"], m["review"], m["false_accepts"], m["n_labelled"]) == (3, 2, 1, 4)
    assert m["accepted_risk"] == pytest.approx(1 / 3) and sum(b["n"] for b in m["reliability"]) == 4
    curve = DEC.risk_coverage(p, y, complete)
    assert [c["accepted"] for c in curve] == [0, 2, 3, 4]                    # the two 0.9s enter together
    assert DEC.risk_at_coverage(curve, 0.2) == pytest.approx(0.5)
    none = DEC.decision_metrics([0.1, 0.2], [1, 0], [True, True], 0.8)
    assert none["accepted"] == 0 and none["accepted_risk"] is None
