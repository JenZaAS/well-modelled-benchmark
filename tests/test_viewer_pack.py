"""pack_well keeps full resolution (under the cap) and leaves null gaps as null."""
from __future__ import annotations

import inspect
import sys

import numpy as np
import pandas as pd
import pytest

from wmb import viewer

sys.path.insert(0, str(viewer.VIEWER_DIR))
from build_wellmap import load_csv_groups, load_well_frame, pack_well  # noqa: E402


def test_pack_logs_default_exceeds_2000_points():
    default = inspect.signature(viewer.ViewerState.pack_logs).parameters["max_points"].default
    assert default >= 20000


def test_pack_well_keeps_nulls_and_more_than_2000_points():
    n = 5000
    gr = np.linspace(10, 100, n)
    gr[1000:1200] = np.nan
    frame = pd.DataFrame({"DEPTH": np.arange(n) * 0.5, "GR": gr})
    well = {"name": "synthetic"}
    packed = pack_well(well, {"id": "synthetic"}, frame, 20000)
    assert len(packed["d"]) == n > 2000
    arr = packed["c"]["GR"]
    assert all(v is None for v in arr[1000:1200])
    assert arr[999] is not None and arr[1200] is not None


@pytest.mark.parametrize("file_format", ["csv", "las"])
def test_packed_depths_are_ascending_for_viewport_search(tmp_path, file_format):
    if file_format == "csv":
        text = "WELL;DEPTH_MD;GR\nsynthetic;1002;52\nsynthetic;1001;51\nsynthetic;1000;50\n"
    else:
        pytest.importorskip("lasio")
        text = (
            "~Version\n VERS. 2.0 :\n WRAP. NO :\n"
            "~Well\n STRT.M 1002 :\n STOP.M 1000 :\n STEP.M -1 :\n NULL. -999.25 :\n"
            "~Curve\n DEPT.M : Depth\n GR.GAPI : Gamma\n"
            "~A\n1002 52\n1001 51\n1000 50\n"
        )
    path = tmp_path / f"synthetic.{file_format}"
    path.write_text(text, encoding="utf-8")
    well = {"name": "synthetic", "files": [{"path": path.name, "format": file_format}]}
    dataset = {"id": "synthetic", "curve_map": {"GR": "GR"}}
    frames = load_csv_groups({"_data_root": tmp_path, "dataset": dataset, "wells": [well]})
    frame = load_well_frame(well, dataset, tmp_path, frames)
    packed = pack_well(well, dataset, frame, 20000)

    assert packed["d"] == [1000, 1001, 1002]
    assert packed["c"]["GR"] == [50, 51, 52]


def test_pack_well_passes_through_display_curves_and_formations():
    """PHIE/VSH/SW/GROUP/FORMATION are packed when present; completeness stays the 9 standard curves."""
    frame = pd.DataFrame({
        "DEPTH": [1000.0, 1001.0, 1002.0],
        "GR": [40.0, 41.0, 42.0],
        "NPHI": [0.20, 0.21, 0.22],
        "PHIE": [0.15, 0.16, None],
        "PHIT": [0.18, 0.19, 0.20],
        "VSH": [0.3, 0.4, 0.5],
        "SW": [0.6, 0.7, 0.8],
        "BS": [8.5, 8.5, 8.5],
        "GROUP": ["HORDALAND GP.", "HORDALAND GP.", "NORDLAND GP."],
        "FORMATION": ["Utsira Fm.", "Utsira Fm.", None],
    })
    packed = pack_well({"name": "synthetic"}, {"id": "synthetic"}, frame, 20000)
    assert packed["c"]["GR"] == [40.0, 41.0, 42.0]
    assert packed["c"]["NPHI"] == [0.2, 0.21, 0.22]
    assert packed["c"]["PHIE"] == [0.15, 0.16, None]
    assert packed["c"]["PHIT"] == [0.18, 0.19, 0.2]
    assert packed["c"]["VSH"] == [0.3, 0.4, 0.5]
    assert packed["c"]["SW"] == [0.6, 0.7, 0.8]
    assert packed["c"]["BS"] == [8.5, 8.5, 8.5]
    assert packed["group"] == ["HORDALAND GP.", "HORDALAND GP.", "NORDLAND GP."]
    assert packed["formation"] == ["Utsira Fm.", "Utsira Fm.", None]
    # Completeness is still the 9 standard curves only — no PHIE/VSH/SW keys.
    assert set(packed["curves"]) == {"GR", "RDEP", "RMED", "CALI", "RHOB", "NPHI", "PEF", "DTC", "DTS"}
    assert "PHIE" not in packed["curves"]
    assert "GROUP" not in packed["c"]

