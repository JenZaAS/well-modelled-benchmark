"""Ordinal density heads: bins, readouts, scores, leakage, missing inputs and the Decima interface. Synthetic only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytest.importorskip("scipy")

from wmb.contract import Task  # noqa: E402
from workstreams.black_physics.ordinal_density import bins as B  # noqa: E402
from workstreams.black_physics.ordinal_density import decima as DZ  # noqa: E402
from workstreams.black_physics.ordinal_density import features as FT  # noqa: E402
from workstreams.black_physics.ordinal_density import heads as H  # noqa: E402
from workstreams.black_physics.ordinal_density import scoring as S  # noqa: E402

TAILS = B.Tails(np.array([1.50, 1.55]), np.array([2.95, 3.05]), False, False)


def gap_task(depth, rhob, a, b, hide=True):
    """A Task with RHOB hidden on [a, b). hide=False leaves the truth in target_observed (a careless caller)."""
    depth, rhob = np.asarray(depth, float), np.asarray(rhob, float)
    m = np.zeros(len(depth), bool)
    m[a:b] = True
    obs = rhob.copy()
    if hide:
        obs[m] = np.nan
    inputs = pd.DataFrame({"DEPTH": depth, "GR": np.linspace(40, 90, len(depth))})
    return Task("W-1", depth, inputs, "RHOB", m, "gap", 50.0, obs, "synthetic", "test", {"gap_id": f"W-1|{a}:{b}"})


def toy(n=400, step=0.152, seed=0):
    rng = np.random.default_rng(seed)
    depth = 1000 + step * np.arange(n)
    rhob = 2.35 + 0.08 * np.sin(np.arange(n) / 17) + 0.01 * rng.standard_normal(n)
    return depth, rhob


# ------------------------------------------------------------------------------------------------- bins
def test_exact_edges_and_both_tails():
    assert B.N_BINS == 15 and len(B.EDGES) == 14
    assert np.allclose(B.EDGES[[0, -1]], [1.60, 2.90])
    y = np.array([1.0, 1.5999, 1.60, 1.65, 1.70, 2.3999, 2.40, 2.8999, 2.90, 3.5, np.nan])
    assert B.to_bin(y).tolist() == [0, 0, 1, 1, 2, 8, 9, 13, 14, 14, -1]
    for k, e in enumerate(B.EDGES):                       # every edge value goes to the higher bin
        assert B.to_bin([e])[0] == k + 1


def test_option_order_matches_bins():
    lab = B.option_labels()
    assert len(lab) == 15 and lab[0].startswith("below 1.60") and lab[-1].startswith("2.90")
    assert lab[1].startswith("1.60 to below 1.70") and lab[9].startswith("2.40 to below 2.50")
    assert DZ.CHOICES == lab and DZ.KIND == "score"


def test_tails_use_training_values_or_a_flagged_extension():
    t = B.Tails.fit([1.40, 1.50, 2.0, 2.5])
    assert not t.low_extended and t.high_extended
    assert np.isclose(t.means()[0], 1.45) and np.isclose(t.means()[-1], 2.95)
    p = np.zeros((1, 15))
    p[0, -1] = 1.0
    assert B.HIGH_EXTENSION[0] <= B.quantile(p, 0.5, t)[0] <= B.HIGH_EXTENSION[1]


def test_quantiles_nest_and_invert_the_cdf():
    rng = np.random.default_rng(1)
    p = B.normalise(rng.dirichlet(np.full(15, 0.3), 200))
    r = B.readouts(p, TAILS)
    assert np.all(r["q05"] <= r["q10"] + 1e-12) and np.all(r["q10"] <= r["median"] + 1e-12)
    assert np.all(r["median"] <= r["q90"] + 1e-12) and np.all(r["q90"] <= r["q95"] + 1e-12)
    one = np.zeros((1, 15))
    one[0, 5] = 0.5                                      # [2.00, 2.10)
    one[0, 6] = 0.5                                      # [2.10, 2.20)
    assert np.isclose(B.quantile(one, 0.5, TAILS)[0], 2.10)
    assert np.isclose(B.quantile(one, 0.25, TAILS)[0], 2.05)
    assert np.isclose(B.readouts(one, TAILS)["mean"][0], 2.10)


# ------------------------------------------------------------------------------------------------ scores
def test_rps_zero_for_correct_one_hot_and_grows_with_distance():
    b = np.full(15, 7)
    p = np.eye(15)
    s = B.rps(p, b)
    assert s[7] == 0.0
    d = np.abs(np.arange(15) - 7)
    for i in range(15):
        for j in range(15):
            if d[i] < d[j]:
                assert s[i] < s[j]
    assert np.isclose(s[0], 7 / 14)                      # 7 of the 14 edge terms are wrong by 1


def test_below_2_40_by_hand():
    p = np.zeros((2, 15))
    p[0, 8], p[0, 9] = 0.3, 0.7                         # [2.30, 2.40) and [2.40, 2.50)
    p[1, :3] = 1 / 3
    assert np.allclose(B.prob_below(p), [0.3, 1.0])
    assert B.EVENT_INDEX == 8
    y = np.array([2.40, 2.3999])
    assert (y < B.EVENT_EDGE).tolist() == [False, True] and B.to_bin(y).tolist() == [9, 8]
    with pytest.raises(ValueError):
        B.prob_below(p, 2.45)


def test_hierarchical_mean_weights_wells_then_gaps():
    v = np.array([1.0, 1.0, 1.0, 3.0, 0.0])
    wells = np.array(["A", "A", "A", "A", "B"])
    gaps = np.array(["a1", "a1", "a1", "a2", "b1"])
    assert S.per_well(v, wells, gaps) == {"A": 2.0, "B": 0.0}
    assert S.hier_mean(v, wells, gaps) == 1.0


def test_abstentions_and_wells_are_kept_apart():
    rng = np.random.default_rng(2)
    p = B.normalise(rng.random((6, 15)))
    p[1] = np.nan
    y = np.array([2.3, 2.4, 2.5, 2.2, 2.6, np.nan])
    wells = np.array(["A", "A", "B", "B", "C", "C"])
    s = S.score(p, y, wells, wells, TAILS)
    assert s["abstentions"] == 1 and s["unscored_no_truth"] == 1 and s["n_depths"] == 4
    assert set(s["_per_well_RPS"]) == {"A", "B", "C"}
    p[4] = np.nan                                        # every scorable row of well C abstains
    s = S.score(p, y, wells, wells, TAILS)
    assert set(s["_per_well_RPS"]) == {"A", "B"} and s["n_wells"] == 2 and s["abstentions"] == 2
    w = H.row_weights(wells, np.array(["a", "a", "b", "c", "d", "d"]))
    assert np.isclose(w[wells == "A"].sum(), w[wells == "B"].sum())


# ------------------------------------------------------------------------------------------------ features
def test_features_never_read_the_hidden_truth():
    depth, rhob = toy()
    a, b = 150, 250
    base = rhob + 0.05
    t1 = gap_task(depth, rhob, a, b)
    other = rhob.copy()
    other[a:b] = 1.0 + 2 * np.random.default_rng(3).random(b - a)      # a different hidden truth
    leaky = gap_task(depth, other, a, b, hide=False)                     # truth left in target_observed
    r1, r2 = FT.gap_rows(t1, base), FT.gap_rows(leaky, base)
    for f in FT.FIELDS:
        np.testing.assert_array_equal(r1[f], r2[f])
    y = rhob[a:b]
    e = H.EmpiricalHead(by_flank=True).fit(base[a:b], y, r1["flank_var"], np.ones(b - a))
    np.testing.assert_array_equal(e.predict(r1["base"], r1["flank_var"]), e.predict(r2["base"], r2["flank_var"]))
    pol = FT.FeaturePolicy(use_width=False, sonic=False)
    X1, names = pol.matrix(r1)
    X2, _ = pol.matrix(r2)
    h = H.SoftmaxHead().fit(X1, names, B.to_bin(y), np.ones(len(y)))
    np.testing.assert_array_equal(h.predict(X1), h.predict(X2))


def test_unequal_depth_spacing_uses_metres():
    rng = np.random.default_rng(4)
    depth = 1000 + np.cumsum(rng.uniform(0.10, 0.16, 400))
    rhob = np.full(400, 2.4)
    a, b = 100, 180
    r = FT.gap_rows(gap_task(depth, rhob, a, b), np.full(400, 2.4))
    j = np.arange(a, b)
    np.testing.assert_allclose(r["dist_up"], depth[j] - depth[a - 1])
    np.testing.assert_allclose(r["dist_dn"], depth[b] - depth[j])
    np.testing.assert_allclose(r["position"], (depth[j] - depth[a]) / (depth[b - 1] - depth[a]))
    np.testing.assert_array_equal(r["index"], j)


def test_missing_sonic_and_point_only_inputs():
    depth, rhob = toy()
    a, b = 0, 80                                         # gap at the top: no shallow flank
    base = rhob + 0.03
    r = FT.gap_rows(gap_task(depth, rhob, a, b), base)  # no band, no Gardner
    assert np.isnan(r["width"]).all() and np.isnan(r["sonic_diff"]).all() and (r["dtc"] == 0).all()
    assert np.isnan(r["dist_up"]).all() and np.isfinite(r["dist_dn"]).all()
    pol = FT.FeaturePolicy(use_width=False, sonic=True)
    X, names = pol.matrix(r)
    gard = np.where(np.arange(len(depth)) % 2 == 0, rhob, np.nan)       # sonic on every other train depth
    rt = FT.gap_rows(gap_task(depth, rhob, 100, 300), base, gardner=gard)
    Xt, _ = pol.matrix(rt)
    h = H.SoftmaxHead().fit(Xt, names, B.to_bin(rhob[100:300]), np.ones(200))
    p = h.predict(X)
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1.0)
    e = H.EmpiricalHead(by_flank=True).fit(rt["base"], rhob[100:300], rt["flank_var"], np.ones(200))
    q = e.predict(r["base"], np.full(len(r["base"]), np.nan))           # no flank variability: pooled
    np.testing.assert_allclose(q, e.pooled.mass(r["base"]))
    nanbase = r["base"].copy()
    nanbase[:5] = np.nan
    assert np.isnan(e.predict(nanbase, r["flank_var"])[:5]).all()       # no base value: abstain


def test_empirical_head_shifts_residuals_by_the_prediction():
    resid = np.array([-0.15, -0.05, 0.05, 0.15])
    e = H.EmpiricalHead().fit(np.full(4, 2.0), 2.0 + resid, np.full(4, np.nan), np.ones(4))
    p = e.predict(np.array([2.40]), np.array([np.nan]))[0]
    expect = np.zeros(15)
    expect[[7, 8, 9, 10]] = 0.25                          # 2.25, 2.35, 2.45, 2.55
    np.testing.assert_allclose(p, H.smooth(expect))
    assert np.isclose(p.sum(), 1.0)


def test_temperature_one_is_identity_and_fit_is_bounded():
    rng = np.random.default_rng(5)
    p = B.normalise(rng.dirichlet(np.ones(15), 50))
    np.testing.assert_allclose(H.apply_temperature(p, 1.0), p)
    T = H.fit_temperature(p, rng.integers(0, 15, 50), np.ones(50))
    assert H.T_BOUNDS[0] <= T <= H.T_BOUNDS[1]


# ------------------------------------------------------------------------------------------------ Decima
def test_decima_reads_cache_in_option_order_and_normalises(tmp_path):
    states = ["Density estimate 2.400 g/cm3.", "Density estimate missing g/cm3."]
    cache = tmp_path / "c.jsonl"
    logp = [list(np.log(np.arange(1, 16, dtype=float))), [0.0] * 15]  # unnormalised on purpose
    with cache.open("w") as f:
        for s, lp in zip(states, logp):
            f.write(json.dumps({"key": DZ.cache_key(s), "state": s, "logp": lp, "seconds": 0.1, "meta": {}}) + "\n")
    p, secs, meta = DZ.run(states, cache, tmp_path, log=lambda m: None)
    assert meta["new"] == 0 and p.shape == (2, 15) and np.allclose(p.sum(1), 1.0)
    np.testing.assert_allclose(p[0], np.arange(1, 16) / 120)
    assert DZ.cache_key(states[0]) != DZ.cache_key(states[0] + " ")


def test_decima_state_and_sample_are_deterministic():
    row = {f: 1.0 for f in FT.FIELDS}
    row["dist_up"] = np.nan
    s = DZ.state_text(row, use_width=False)
    assert "missing" in s and "P10-P90" not in s and "2.4" not in s
    assert "P10-P90 width" in DZ.state_text(row, use_width=True)
    i1 = DZ.sample_positions(330, "m", "D", "W|0:330")
    assert np.array_equal(i1, DZ.sample_positions(330, "m", "D", "W|0:330")) and len(i1) == 8
    sec = np.array_split(np.arange(330), 8)
    assert all(i in s_ for i, s_ in zip(i1, sec))


def test_inference_entry_point_matches_the_fitted_head(tmp_path):
    from workstreams.black_physics.ordinal_density.infer import Head

    depth, rhob = toy()
    base = rhob + 0.02
    tr = FT.gap_rows(gap_task(depth, rhob, 100, 300), base)
    y = rhob[100:300]
    e = H.EmpiricalHead(by_flank=True).fit(tr["base"], y, tr["flank_var"], np.ones(200))
    pol = FT.FeaturePolicy(use_width=False, sonic=False)
    X, names = pol.matrix(tr)
    s = H.SoftmaxHead().fit(X, names, B.to_bin(y), np.ones(200))
    bundle = {"tails": B.Tails.fit(y).as_dict(),
              "heads": {"empirical_flank": {**e.as_dict(), "policy": None, "temperature": 1.3},
                        "numeric": {**s.as_dict(), "policy": pol.as_dict(), "temperature": 0.8}}}
    (tmp_path / "heads_m.json").write_text(json.dumps(bundle))
    task = gap_task(depth, rhob, 320, 380)
    rows = FT.gap_rows(task, base)
    out = Head.load(tmp_path, "m", "empirical_flank|cal").predict(task, base[task.predict_mask])
    np.testing.assert_allclose(out["probs"], H.apply_temperature(e.predict(rows["base"], rows["flank_var"]), 1.3))
    out = Head.load(tmp_path, "m", "numeric|raw").predict(task, base)
    np.testing.assert_allclose(out["probs"], s.predict(pol.matrix(rows)[0]))
    assert out["probs"].shape == (60, 15) and np.allclose(out["probs"].sum(1), 1.0)
    with pytest.raises(ValueError):
        Head.load(tmp_path, "m", "numeric|raw").predict(task, base[:10])


def test_cv_folds_never_split_a_well_and_pick_from_the_grid():
    wells = np.repeat([f"W{i}" for i in range(9)], 30)
    f = H.well_folds(wells, 4)
    for w in np.unique(wells):
        assert len(set(f[wells == w])) == 1
    assert set(f) == {0, 1, 2, 3}
    rng = np.random.default_rng(6)
    y = 2.3 + 0.1 * rng.standard_normal(len(wells))
    X = np.column_stack([y + 0.03 * rng.standard_normal(len(y)), rng.random(len(y))])
    best, table = H.select_l2(X, ["base", "noise"], B.to_bin(y), wells, wells, grid=(1e-2, 1e-4), n_folds=3)
    assert best in (1e-2, 1e-4) and set(table) == {1e-2, 1e-4} and all(np.isfinite(list(table.values())))
