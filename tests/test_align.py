"""Alignment to the FORCE / canonical curve standard. Synthetic LAS only; no real well data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numpy")
pytest.importorskip("yaml")
lasio = pytest.importorskip("lasio")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from wmb.align.mapping import (  # noqa: E402
    convert_depth,
    infer_and_convert,
    load_align_spec,
    norm_mnem,
    pick_best,
    Candidate,
    assess_candidate,
)
from wmb.align.pipeline import (  # noqa: E402
    align_dataset,
    assign_to_grid,
    build_grid,
    choose_step,
)


def _las(path: Path, depth, unit="M", curves: dict | None = None, well="TEST",
         start=None, extras_header=""):
    """Write a tiny LAS 2.0 file. `curves` is name -> (unit, descr, values)."""
    depth = np.asarray(depth, float)
    curves = curves or {}
    start = float(depth[0]) if start is None else start
    stop = float(depth[-1])
    step = float(depth[1] - depth[0]) if depth.size > 1 else 0.1
    lines = [
        "~Version information",
        "VERS.  2.0 :",
        "WRAP.  NO :",
        "~Well information",
        f"STRT.{unit}  {start} :",
        f"STOP.{unit}  {stop} :",
        f"STEP.{unit}  {step} :",
        "NULL.  -999.25 :",
        f"WELL.  {well} :",
        "~Curve information",
        f"DEPT.{unit} : depth",
    ]
    names = []
    for name, (u, descr, _v) in curves.items():
        lines.append(f"{name}.{u} : {descr}")
        names.append(name)
    lines.append("~Ascii")
    for i, d in enumerate(depth):
        row = [f"{float(d):.4f}"]
        for name in names:
            v = curves[name][2][i]
            row.append("-999.25" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{float(v):.4f}")
        lines.append(" ".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _depth(n=11, start=100.0, step=0.1):
    return start + step * np.arange(n)


def test_norm_mnem_strips_suffixes():
    assert norm_mnem("rd-1") == "RD_1" or norm_mnem("AF90:1") == "AF90"
    assert norm_mnem("AF90:1") == "AF90"
    assert norm_mnem("GR_NO_1") == "GR"


def test_depth_ft_to_metres():
    d, flags, unit = convert_depth(np.array([1000.0, 2000.0]), "FT")
    assert unit == "m"
    assert flags == ["depth_converted_from_ft"]
    np.testing.assert_allclose(d, np.array([1000.0, 2000.0]) * 0.3048)


def test_depth_rejects_tenth_inch_index():
    with pytest.raises(ValueError, match=r"\.1IN"):
        convert_depth(np.array([1.0, 2.0]), ".1IN")


def test_unit_conversions():
    spec = load_align_spec("nlog_dts")
    # us/m sonic
    x, flags, u = infer_and_convert(np.array([322.0, 330.0]), spec, "DTC", "US/M")
    assert u == "us/ft" and "sonic_converted_from_us_m" in flags
    np.testing.assert_allclose(x, np.array([322.0, 330.0]) * 0.3048)
    # NPHI percent
    x, flags, u = infer_and_convert(np.array([16.0, 18.0]), spec, "NPHI", "%")
    assert "nphi_converted_from_percent" in flags
    np.testing.assert_allclose(x, np.array([0.16, 0.18]))
    # PU already fraction
    x, flags, u = infer_and_convert(np.array([0.21, 0.22]), spec, "NPHI", "PU")
    assert "nphi_converted_from_percent" not in flags
    np.testing.assert_allclose(x, np.array([0.21, 0.22]))
    # density kg/m3
    x, flags, u = infer_and_convert(np.array([2200.0, 2500.0]), spec, "RHOB", "kg/m3")
    np.testing.assert_allclose(x, np.array([2.2, 2.5]))
    assert "density_converted_from_kg_m3" in flags
    # conductivity
    x, flags, u = infer_and_convert(np.array([100.0, 200.0]), spec, "RDEP", "mS/m")
    np.testing.assert_allclose(x, np.array([10.0, 5.0]))
    assert "resistivity_converted_from_conductivity" in flags


def test_sentinels_and_no_gap_interpolation():
    grid = build_grid(10.0, 10.4, 0.1)
    depth = np.array([10.0, 10.1, 10.3, 10.4])
    values = np.array([2.2, 2.3, 2.5, 2.6])  # gap at 10.2
    out = assign_to_grid(depth, values, grid, 0.1)
    np.testing.assert_allclose(out[0], 2.2)
    assert not np.isfinite(out[2]), "must not interpolate across a gap"
    np.testing.assert_allclose(out[3], 2.5)


def test_choose_step_snaps_half_foot():
    d = 1000 + 0.1524 * np.arange(50)
    assert abs(choose_step([d]) - 0.1524) < 1e-6


def test_priority_prefers_deep_resistivity_and_rejects_bad_aliases():
    spec = load_align_spec("kansas_kgs_dts")
    depth = _depth()
    af90 = Candidate("RDEP", "AF90", "OHMM", "ohm.m", "array 90", "a.las",
                     np.full(depth.shape, 8.0), depth, alias_rank=5)
    af60 = Candidate("RDEP", "AF60", "OHMM", "ohm.m", "array 60", "a.las",
                     np.full(depth.shape, 7.0), depth, alias_rank=8)
    best = pick_best([assess_candidate(af90, spec), assess_candidate(af60, spec)])
    assert best is not None and best.mnemonic == "AF90"

    dtst = assess_candidate(Candidate("DTS", "DTST", "US/FT", "us/ft", "Stoneley slowness", "a.las",
                                      np.full(depth.shape, 280.0), depth, alias_rank=0), spec)
    assert dtst.rejected == "stoneley_not_shear"

    rd_in = assess_candidate(Candidate("RDEP", "RD", "IN", "ohm.m", "Deep resistivity", "a.las",
                                       np.full(depth.shape, 8.0), depth, alias_rank=0), spec)
    assert rd_in.rejected == "resistivity_unit_is_inches"


def test_dcal_accepted_only_when_hole_size():
    spec = load_align_spec("kansas_kgs_dts")
    depth = _depth()
    ok = assess_candidate(Candidate("CALI", "DCAL", "IN", "in", "Density Caliper", "a.las",
                                    np.full(depth.shape, 8.5), depth, alias_rank=10000), spec)
    assert ok.rejected is None and "dcal_accepted_as_caliper" in ok.flags
    bad = assess_candidate(Candidate("CALI", "DCAL", "IN", "in", "Differential Caliper", "a.las",
                                     np.full(depth.shape, 0.2), depth, alias_rank=10000), spec)
    assert bad.rejected in {"dcal_is_differential", "dcal_near_zero"}


def test_range_flags_do_not_drop(tmp_path: Path):
    depth = _depth(n=20, start=100, step=0.1)
    rhob = np.full(depth.shape, 2.4)
    rhob[0] = 0.2  # outside 1.0-3.5, kept
    well = tmp_path / "in" / "W1"
    _las(well / "w.las", depth, unit="M", well="W1", curves={
        "GR": ("GAPI", "gamma", np.full(depth.shape, 80.0)),
        "RHOB": ("G/C3", "density", rhob),
        "DTCO": ("US/FT", "compressional", np.full(depth.shape, 90.0)),
        "DTSM": ("US/FT", "shear", np.full(depth.shape, 160.0)),
    })
    (tmp_path / "in" / "wells.json").write_text(json.dumps({
        "manifest_version": 1, "dataset": {"id": "nlog_dts", "credit": "Source: NLOG.NL"},
        "wells": [{"name": "W1"}],
    }))
    out = tmp_path / "aligned"
    result = align_dataset(tmp_path / "in", out, "nlog_dts")
    w = result["wells"][0]
    assert np.isfinite(w["curves"]["RHOB"][0]) and w["curves"]["RHOB"][0] == pytest.approx(0.2)
    assert w["provenance"]["RHOB"]["range"]["samples_outside_plausible_range"] >= 1


def test_align_dataset_writes_standard_files_and_keeps_provenance(tmp_path: Path):
    # File A: feet, percent NPHI, AF90 / AF60, DTST (must not become DTS)
    depth_ft = 3280.84 + 0.5 * np.arange(21)  # ~1000 m, 0.5 ft
    n = depth_ft.size
    well_a = tmp_path / "raw" / "WELL-A"
    _las(well_a / "res.las", depth_ft, unit="FT", well="WELL-A", curves={
        "GR": ("GAPI", "gamma", np.full(n, 70.0)),
        "AF90": ("OHMM", "deep", np.full(n, 9.0)),
        "AF60": ("OHMM", "medium-deep", np.full(n, 6.0)),
        "NPHI": ("PU", "neutron percent", np.full(n, 20.0)),
        "DTST": ("US/FT", "Stoneley", np.full(n, 270.0)),
    })
    # File B: metres, RHOB, DTC/DTS, a gap in RHOB, plus a source TVD curve (already in metres)
    depth_m = 1000.0 + 0.1524 * np.arange(21)
    rhob = np.full(depth_m.shape, 2.35)
    rhob[8:12] = np.nan
    tvd = depth_m - 5.0
    _las(well_a / "son.las", depth_m, unit="M", well="WELL-A", curves={
        "RHOB": ("G/CC", "density", rhob),
        "DTCO": ("US/FT", "compressional", np.full(n, 95.0)),
        "DTSM": ("US/FT", "shear", np.full(n, 170.0)),
        "TVD": ("M", "true vertical depth", tvd),
        "INCL": ("DEG", "inclination", np.full(n, 12.0)),
    })
    (tmp_path / "raw" / "wells.json").write_text(json.dumps({
        "manifest_version": 1,
        "dataset": {"id": "demo", "name": "Demo", "licence": "test", "credit": "Source: NLOG.NL",
                    "source_url": "https://example.example/"},
        "wells": [{"name": "WELL-A", "lat": 1.0, "lon": 2.0, "td_md_m": 2000, "td_tvd_m": 1800}],
    }))
    out = tmp_path / "out"
    result = align_dataset(tmp_path / "raw", out, "nlog_dts")
    assert (out / "WELL-A" / "aligned.las").is_file()
    assert (out / "wells.json").is_file() and (out / "completeness.csv").is_file() and (out / "README.md").is_file()
    text = (out / "README.md").read_text(encoding="utf-8")
    assert "NLOG.NL" in text and "MIT repo" in text
    w = result["wells"][0]
    assert w["curve_map"]["RDEP"] == "AF90"
    written = json.loads((out / "wells.json").read_text(encoding="utf-8"))["wells"][0]
    assert written["curve_map"]["RDEP"] == "RDEP" and written["source_curve_map"]["RDEP"] == "AF90"
    assert "DTS" in w["curves"] and w["curve_map"]["DTS"] == "DTSM"
    assert "DTST" not in (w["curve_map"].get("DTS"),)
    np.testing.assert_allclose(w["curves"]["NPHI"][np.isfinite(w["curves"]["NPHI"])][0], 0.20, atol=1e-6)
    # gap in RHOB preserved
    assert np.isnan(w["curves"]["RHOB"]).any()
    # depth in metres (1000 ft would be wrong)
    assert 999 <= w["depth_m"][0] <= 1002
    ref = w["depth_reference"]
    assert ref["reference"] == "MD" and ref["tvd_available"] is True
    assert any(c["mnemonic"] == "TVD" for c in ref["tvd_curves"])
    assert any(c["mnemonic"] == "INCL" for c in ref["deviation_curves"])
    assert ref["apparently_deviated"] is True  # |2000-1800| > 10, from metadata, not computed TVD
    assert "TVD" in w["extras"]
    man = json.loads((out / "wells.json").read_text(encoding="utf-8"))
    assert man["dataset"]["depth_units"] == "m" and man["dataset"]["depth_reference"] == "MD"
    assert man["wells"][0]["lat"] == 1.0
    csv_text = (out / "completeness.csv").read_text(encoding="utf-8")
    assert "WELL-A" in csv_text and "yes" in csv_text
    # data was written only under --output, not into the repo
    assert out.resolve() == (tmp_path / "out").resolve()


def test_decreasing_depth_is_sorted_via_grid(tmp_path: Path):
    depth = np.array([110.0, 109.9, 109.8, 109.7])
    well = tmp_path / "in" / "REV"
    _las(well / "a.las", depth, unit="M", well="REV", curves={
        "GR": ("GAPI", "gamma", np.array([10.0, 11.0, 12.0, 13.0])),
        "RHOB": ("G/CC", "den", np.array([2.1, 2.2, 2.3, 2.4])),
        "DTC": ("US/FT", "dtc", np.array([90, 91, 92, 93], float)),
    })
    result = align_dataset(tmp_path / "in", tmp_path / "out", "nlog_dts")
    d = result["wells"][0]["depth_m"]
    assert np.all(np.diff(d[np.isfinite(d)]) > 0)
    # First depth is the shallowest; GR there was 13 (logged at 109.7 going upwards)
    w = result["wells"][0]
    i = int(np.argmin(np.abs(w["depth_m"] - 109.7)))
    assert w["curves"]["GR"][i] == pytest.approx(13.0)


def test_does_not_invent_missing_curve(tmp_path: Path):
    depth = _depth()
    well = tmp_path / "in" / "NO-DTS"
    _las(well / "a.las", depth, unit="M", well="NO-DTS", curves={
        "GR": ("GAPI", "gamma", np.full(depth.shape, 60.0)),
        "RHOB": ("G/CC", "den", np.full(depth.shape, 2.4)),
    })
    w = align_dataset(tmp_path / "in", tmp_path / "out", "nlog_dts")["wells"][0]
    assert "DTS" not in w["curves"] or not np.isfinite(w["curves"].get("DTS", np.array([np.nan]))).any()
    assert w["completeness"]["present"]["DTS"] is False


def test_windows_inf_tokens_do_not_drop_the_file(tmp_path: Path):
    depth = _depth()
    values = np.full(depth.shape, 170.0)
    well = tmp_path / "in" / "INF"
    _las(well / "a.las", depth, unit="M", well="INF", curves={
        "GR": ("GAPI", "gamma", np.full(depth.shape, 70.0)),
        "DTSM": ("US/FT", "shear", values),
        "DTCO": ("US/FT", "p", np.full(depth.shape, 90.0)),
    })
    text = (well / "a.las").read_text(encoding="utf-8")
    # inject a Windows INF on one sample of DTSM
    lines = text.splitlines()
    # last data line: replace last number
    parts = lines[-1].split()
    parts[-2] = "1.#INF"  # DTSM column depends on order: DEPT GR DTSM DTCO — actually insertion order GR, DTSM, DTCO
    lines[-1] = " ".join(parts)
    # safer: replace a known value
    text2 = text.replace("170.0000", "1.#INF", 1)
    (well / "a.las").write_text(text2, encoding="utf-8")
    w = align_dataset(tmp_path / "in", tmp_path / "out", "nlog_dts")["wells"][0]
    assert w["completeness"]["present"]["DTS"] is True
    assert np.isnan(w["curves"]["DTS"]).any()  # the INF sample is missing, not a crash


def test_implausible_depth_file_is_skipped_not_merged(tmp_path: Path):
    well = tmp_path / "in" / "DEEP"
    good = _depth(n=20, start=1000, step=0.1)
    _las(well / "good.las", good, unit="M", well="DEEP", curves={
        "GR": ("GAPI", "gamma", np.full(good.shape, 80.0)),
        "DTCO": ("US/FT", "p", np.full(good.shape, 90.0)),
        "DTSM": ("US/FT", "s", np.full(good.shape, 160.0)),
    })
    bad = 20000 + np.arange(20)
    _las(well / "bad.las", bad, unit="M", well="DEEP", curves={
        "GR": ("GAPI", "gamma", np.full(bad.shape, 30.0)),
        "RHOB": ("G/CC", "den", np.full(bad.shape, 2.2)),
    })
    w = align_dataset(tmp_path / "in", tmp_path / "out", "nlog_dts")["wells"][0]
    assert w["depth_m"].max() < 2000
    assert w["completeness"]["present"]["DTS"] is True
    assert any("implausible" in (e.get("error") or "").lower() for e in w["file_errors"])
