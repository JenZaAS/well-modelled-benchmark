"""Equations: published numeric examples, units, domains, monotonicity and invalid inputs. No data needed."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from workstreams.black_physics.models._shared.units import (km_s_to_slowness, m_s_to_slowness,  # noqa: E402
                                                            slowness_to_km_s, slowness_to_m_s)
from workstreams.black_physics.models.castagna_mudrock.equation import mudrock_dts, mudrock_vs  # noqa: E402
from workstreams.black_physics.models.gardner.equation import (PUBLISHED_A, PUBLISHED_B, fit_gardner,  # noqa: E402
                                                               gardner_density)
from workstreams.black_physics.models.greenberg_castagna.equation import (GC_COEFFICIENTS,  # noqa: E402
                                                                          endmember_vs, gc_brine_dts,
                                                                          gc_brine_gr_dts, gc_brine_vs,
                                                                          gr_shale_fraction, gr_well_endpoints)
from workstreams.black_physics.models.volumetric_density.equation import (bulk_density, matrix_density,  # noqa: E402
                                                                          mixed_density)
from workstreams.black_physics.models.vpvs_ratio.equation import fit_ratio, ratio_dts, well_ratio  # noqa: E402
from workstreams.black_physics.models.wyllie.equation import wyllie_dtc  # noqa: E402

BAD = np.array([np.nan, np.inf, -np.inf, 0.0, -50.0])


# --- units -----------------------------------------------------------------------------------------

def test_velocity_units_differ_by_exactly_1000():
    dtc = np.array([50.0, 100.0, 189.0])
    np.testing.assert_allclose(slowness_to_m_s(dtc), 1000 * slowness_to_km_s(dtc), rtol=1e-15)
    assert slowness_to_m_s(100.0) == pytest.approx(3048.0)
    assert slowness_to_km_s(100.0) == pytest.approx(3.048)
    np.testing.assert_allclose(km_s_to_slowness(slowness_to_km_s(dtc)), dtc)
    np.testing.assert_allclose(m_s_to_slowness(slowness_to_m_s(dtc)), dtc)


def test_slowness_conversion_abstains_on_invalid_values():
    for f in (slowness_to_km_s, slowness_to_m_s, km_s_to_slowness, m_s_to_slowness):
        assert np.isnan(f(BAD)).all()
    assert np.isnan(slowness_to_m_s(5e-324))           # overflow to inf is not a prediction


# --- Gardner ---------------------------------------------------------------------------------------

def test_gardner_published_example():
    assert float(gardner_density(100.0)) == pytest.approx(2.30338, abs=5e-6)
    assert (PUBLISHED_A, PUBLISHED_B) == (0.31, 0.25)


def test_gardner_uses_m_s_not_km_s():
    wrong = PUBLISHED_A * float(slowness_to_km_s(100.0)) ** PUBLISHED_B      # 0.41: the km/s mistake
    assert float(gardner_density(100.0)) == pytest.approx(wrong * 1000 ** 0.25)
    assert abs(float(gardner_density(100.0)) - wrong) > 1.5


def test_gardner_monotone_and_domain():
    dtc = np.linspace(40, 200, 50)
    rho = gardner_density(dtc)
    assert np.isfinite(rho).all() and (np.diff(rho) < 0).all()
    assert np.isnan(gardner_density(BAD)).all()
    assert np.isfinite(gardner_density([20.0, 400.0])).all()      # extrapolation outside the fit range


def test_gardner_fit_recovers_coefficients_and_falls_back():
    dtc = np.linspace(60, 150, 200)
    rho = 0.25 * slowness_to_m_s(dtc) ** 0.27
    c = fit_gardner(dtc, rho)
    assert c.fitted and c.a == pytest.approx(0.25, rel=1e-8) and c.b == pytest.approx(0.27, rel=1e-8)
    # fitting range is exclusive and separate from prediction
    assert fit_gardner(np.r_[dtc, 30.0, 250.0], np.r_[rho, 9.0, 9.0]).b == pytest.approx(0.27, rel=1e-8)
    few = fit_gardner(dtc[:10], rho[:10])                       # exactly 10 rows is not enough
    assert not few.fitted and (few.a, few.b) == (0.31, 0.25)
    assert fit_gardner(dtc[:11], rho[:11]).fitted
    flat = fit_gardner(np.full(50, 100.0), np.linspace(2.0, 2.6, 50))
    assert not flat.fitted and "distinct" in flat.note
    backwards = fit_gardner(dtc, rho[::-1])                     # density decreasing with velocity
    assert not backwards.fitted and "increasing" in backwards.note
    nonpositive = fit_gardner(dtc, -rho)
    assert not nonpositive.fitted
    empty = fit_gardner(np.array([], float), np.array([], float))
    assert not empty.fitted and empty.n_rows == 0


# --- Castagna mudrock line -------------------------------------------------------------------------

def test_mudrock_unit_conversions():
    dtc = np.array([60.0, 100.0, 150.0])
    vp = 304.8 / dtc
    expected = 304.8 / (0.8621 * vp - 1.1724)
    np.testing.assert_allclose(mudrock_dts(dtc), expected, rtol=1e-12)
    assert float(mudrock_dts(100.0)) == pytest.approx(209.44412, abs=1e-4)
    assert float(mudrock_vs(4.0)) == pytest.approx(0.8621 * 4 - 1.1724)


def test_mudrock_velocity_guard():
    floor_dtc = 304.8 / ((0.3 + 1.1724) / 0.8621)               # 178.46 us/ft
    assert np.isfinite(mudrock_dts(floor_dtc - 0.01)) and np.isnan(mudrock_dts(floor_dtc + 0.01))
    assert np.isnan(mudrock_dts([250.0, 400.0])).all()          # negative Vs region
    assert np.isnan(mudrock_dts(BAD)).all()
    dts = mudrock_dts(np.linspace(40, 178, 60))
    assert np.isfinite(dts).all() and (np.diff(dts) > 0).all()
    assert (dts > np.linspace(40, 178, 60)).all()               # Vp/Vs > 1 on the whole domain


# --- Greenberg-Castagna ----------------------------------------------------------------------------

def test_gc_published_example():
    assert float(endmember_vs(4.0, "sandstone")) == pytest.approx(2.36076, abs=1e-9)
    assert float(endmember_vs(4.0, "shale")) == pytest.approx(2.21141, abs=1e-9)
    vs = float(gc_brine_vs(4.0, {"sandstone": 0.5, "shale": 0.5}))
    assert vs == pytest.approx(2.284865, abs=5e-7)
    assert float(gc_brine_dts(304.8 / 4.0, {"sandstone": 0.5, "shale": 0.5})) == pytest.approx(133.3995, abs=5e-5)


@pytest.mark.parametrize("lith", sorted(GC_COEFFICIENTS))
def test_gc_pure_endmembers(lith):
    vp = np.array([2.0, 3.0, 4.0, 5.5])
    a2, a1, a0 = GC_COEFFICIENTS[lith]
    np.testing.assert_allclose(gc_brine_vs(vp, {lith: 1.0}), a2 * vp ** 2 + a1 * vp + a0, rtol=1e-14)
    dts = gc_brine_dts(np.linspace(45, 160, 40), {lith: 1.0})
    assert np.isfinite(dts).all() and (np.diff(dts) > 0).all()  # slower P -> slower S inside the domain


def test_gc_mixture_bounds_and_four_components():
    rng = np.random.default_rng(0)
    vp = rng.uniform(2.0, 6.0, 500)
    f = rng.dirichlet(np.ones(4), 500)
    fr = {k: f[:, i] for i, k in enumerate(sorted(GC_COEFFICIENTS))}
    vs = gc_brine_vs(vp, fr)
    speeds = np.column_stack([endmember_vs(vp, k) for k in sorted(GC_COEFFICIENTS)])
    assert np.isfinite(vs).all()
    assert (vs >= speeds.min(axis=1) - 1e-12).all() and (vs <= speeds.max(axis=1) + 1e-12).all()


def test_gc_invalid_fractions_are_masked_locally():
    vp = np.full(6, 4.0)
    sand = np.array([0.5, -0.1, np.nan, 0.6, np.inf, 0.5])
    shale = np.array([0.5, 1.1, 0.5, 0.5, 0.0, 0.5])
    vs = gc_brine_vs(vp, {"sandstone": sand, "shale": shale})
    assert np.isfinite(vs[[0, 5]]).all() and np.isnan(vs[1:5]).all()
    assert np.isnan(gc_brine_vs([0.0, -1.0, np.nan, np.inf], {"shale": 1.0})).all()
    with pytest.raises(ValueError):
        gc_brine_vs(4.0, {"granite": 1.0})
    with pytest.raises(ValueError):
        gc_brine_vs(4.0, {})


def test_gc_nonpositive_active_endmember_masks_but_zero_weight_is_skipped():
    vp = 1.0                                                     # sandstone speed < 0, dolomite > 0
    assert float(endmember_vs(vp, "sandstone")) < 0 < float(endmember_vs(vp, "dolomite"))
    assert float(gc_brine_vs(vp, {"sandstone": 0.0, "dolomite": 1.0})) == pytest.approx(0.58321 - 0.07775)
    assert np.isnan(gc_brine_vs(vp, {"sandstone": 0.1, "dolomite": 0.9}))


def test_gc_sum_tolerance():
    assert np.isfinite(gc_brine_vs(4.0, {"sandstone": 0.3, "shale": 0.7 + 1e-12}))
    assert np.isnan(gc_brine_vs(4.0, {"sandstone": 0.3, "shale": 0.69}))


def test_gr_proxy():
    np.testing.assert_allclose(gr_shale_fraction([20, 30, 75, 120, 200], 30, 120), [0, 0, 0.5, 1, 1])
    assert np.isnan(gr_shale_fraction([np.nan, np.inf], 30, 120)).all()
    for lo, hi in ((None, 120), (120, 120), (130, 120), (np.nan, 120)):
        assert np.isnan(gr_shale_fraction([50.0], lo, hi)).all()
    pure_sand = gc_brine_gr_dts([100.0], [10.0], 30, 120)
    np.testing.assert_allclose(pure_sand, gc_brine_dts([100.0], {"sandstone": 1.0}))
    assert np.isnan(gc_brine_gr_dts([100.0, 100.0], [np.nan, 50.0], None, None)).all()


def test_gr_endpoints_median_of_well_percentiles():
    wells = [np.arange(101.0), np.arange(101.0) + 20, np.array([np.nan, np.nan])]
    lo, hi, n = gr_well_endpoints(wells)
    assert n == 2 and lo == pytest.approx(20.0) and hi == pytest.approx(100.0)
    assert gr_well_endpoints([np.array([np.nan])]) is None


# --- Wyllie and volumetric density -----------------------------------------------------------------

def test_wyllie_example_endpoints_and_domain():
    assert float(wyllie_dtc(0.20)) == pytest.approx(82.2)
    np.testing.assert_allclose(wyllie_dtc([0.0, 1.0]), [55.5, 189.0])
    assert np.isnan(wyllie_dtc([-1e-9, 1 + 1e-9, np.nan, np.inf, -np.inf])).all()
    dt = wyllie_dtc(np.linspace(0, 1, 21))
    assert (np.diff(dt) > 0).all()
    with pytest.raises(ValueError):
        wyllie_dtc(0.2, dt_matrix=0.0)


def test_density_example_endpoints_and_domain():
    assert float(bulk_density(0.20)) == pytest.approx(2.32)
    np.testing.assert_allclose(bulk_density([0.0, 1.0]), [2.65, 1.00])
    assert np.isnan(bulk_density([-0.01, 1.01, np.nan, np.inf])).all()
    assert (np.diff(bulk_density(np.linspace(0, 1, 21))) < 0).all()
    with pytest.raises(ValueError):
        bulk_density(0.2, rho_fluid=-1.0)


def test_density_mineral_mixture():
    rho_ma = matrix_density({"quartz": 0.6, "calcite": 0.4}, {"quartz": 2.65, "calcite": 2.71})
    assert float(rho_ma) == pytest.approx(0.6 * 2.65 + 0.4 * 2.71)
    assert float(mixed_density(0.2, {"quartz": 0.6, "calcite": 0.4}, {"quartz": 2.65, "calcite": 2.71})) == \
        pytest.approx(0.8 * float(rho_ma) + 0.2)
    assert float(mixed_density(0.2)) == pytest.approx(2.32)
    bad = matrix_density({"quartz": [0.5, -0.1, 0.7], "calcite": [0.5, 1.1, 0.2]}, {"quartz": 2.65, "calcite": 2.71})
    assert np.isfinite(bad[0]) and np.isnan(bad[1:]).all()
    with pytest.raises(ValueError):
        matrix_density({"quartz": 1.0}, {})


# --- constant ratio --------------------------------------------------------------------------------

def test_ratio_units_and_fit():
    # DTS/DTC is Vp/Vs: the same in any slowness unit
    assert well_ratio([100.0, 80.0], [180.0, 144.0]) == pytest.approx(1.8)
    assert well_ratio(np.array([100.0, 80.0]) / 0.3048, np.array([180.0, 144.0]) / 0.3048) == pytest.approx(1.8)
    vp, vs = slowness_to_km_s(100.0), slowness_to_km_s(180.0)
    assert float(vp / vs) == pytest.approx(1.8)
    assert well_ratio([100.0, -1, np.nan, 0], [np.nan, 150, 150, 150]) is None
    r, n = fit_ratio([([100.0], [170.0]), ([100.0], [180.0]), ([100.0], [200.0]), ([np.nan], [1.0])])
    assert (r, n) == (pytest.approx(1.8), 3)
    assert fit_ratio([]) == (None, 0)
    dts = ratio_dts([50.0, 100.0, 0.0, -5.0, np.nan, np.inf], 1.8)
    np.testing.assert_allclose(dts[:2], [90.0, 180.0])
    assert np.isnan(dts[2:]).all()
    assert np.isnan(ratio_dts([100.0], None)).all() and np.isnan(ratio_dts([100.0], -1.0)).all()
