"""Predict a log of a new well from uploaded files (no Streamlit here).

    well, mapping = load_las(las_bytes)               LAS -> ``Well`` with the 9 standard log names
    tops = read_tops(file_bytes, "tops.xlsx", well.name)    optional formation tops table
    unmatched = attach_tops(well, tops, known_labels) puts GROUP / FORMATION per depth sample in ``well.meta``
    pred = predict_log(model_dir, well, "RHOB", "gbt")

Tops tables use the NPD layout: one column with the top's name (``Surface`` or ``HorizonName``, e.g.
``VIKING GP. Top`` or ``Draupne Fm. Top``) and one with its measured depth (``MD``), optionally a well column.
Names ending in ``GP.`` are groups, everything else is a formation. A top applies down to the next top of its level.
"""
from __future__ import annotations

import io
import os
import re
import tempfile

import numpy as np
import pandas as pd

from wmb.curves import CORE, UNITS, match_property
from wmb.io import Well, read_las

from .data import UNKNOWN
from .store import load_model

NAME_COLUMNS = ("surface", "horizonname", "horizon", "formation", "name", "top")
DEPTH_COLUMNS = ("md", "depth", "depth_md", "measured depth", "depth md")
WELL_COLUMNS = ("well identifier", "well", "wellbore", "well name", "well_name")


def load_las(data: bytes) -> tuple[Well, dict[str, str]]:
    """Read LAS bytes into a ``Well`` (depth in metres, increasing) and report which mnemonic became which standard log."""
    import lasio
    fd, tmp = tempfile.mkstemp(suffix=".las")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        well = read_las(tmp, dataset="upload")
        las = lasio.read(tmp)
    finally:
        os.unlink(tmp)
    mapping: dict[str, str] = {}
    for c in las.curves[1:]:
        std = match_property(c.mnemonic)
        if std and std not in mapping:
            mapping[std] = c.mnemonic
    if "WELL" in las.well and str(las.well["WELL"].value).strip():
        well.name = str(las.well["WELL"].value).strip()
    ok = np.isfinite(well.depth)
    order = np.argsort(well.depth[ok], kind="stable")
    well.depth = well.depth[ok][order]
    well.curves = {k: v[ok][order] for k, v in well.curves.items()}
    return well, mapping


def sample_spacing(well: Well) -> float:
    """Median depth step in metres (0 if the well has fewer than two samples)."""
    return float(np.median(np.diff(well.depth))) if len(well.depth) > 1 else 0.0


def well_warnings(well: Well, ranges: dict[str, list[float]] | None) -> list[str]:
    """Things that make a prediction unreliable: no input logs, unusual units, odd sample spacing."""
    out = []
    if not any(well.has(c) for c in CORE):
        out.append("None of the 9 standard logs were recognised in this file.")
    for c in CORE:
        if well.has(c) and ranges and c in ranges:
            med, (lo, hi) = float(np.nanmedian(well.curves[c])), ranges[c]
            if not lo <= med <= hi:
                out.append(f"{c}: median {med:.3g} is outside the usual range {lo:.3g} to {hi:.3g} {UNITS[c]} of the "
                           "training wells. Check the unit (for example kg/m3 vs g/cm3, % vs v/v, us/m vs us/ft).")
    step = sample_spacing(well)
    if step and not 0.02 <= step <= 1.0:
        out.append(f"Sample spacing is {step:.3g} m; the models were trained on 0.152 m.")
    return out


def _norm(text: object) -> str:
    return re.sub(r"[\s/_-]+", "", str(text)).lower()


def _match_well(wells: list[str], name: str | None) -> str | None:
    """The well id of a tops table that belongs to a LAS well name such as '15/9-23 Skardkollen'.

    Exact match first, then the longest id that the name starts with, where the id must end at a word boundary
    ('15/9-1' must not match '15/9-13')."""
    if not name:
        return None
    exact = [w for w in wells if _norm(w) == _norm(name)]
    if exact:
        return exact[0]
    starts = [w for w in wells if name.lower().startswith(w.lower().strip())
              and not re.match(r"[\w-]", name[len(w.strip()):len(w.strip()) + 1] or " ")]
    return max(starts, key=len) if starts else None


def _pick(columns: list[str], wanted: tuple[str, ...]) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    return next((lower[w] for w in wanted if w in lower), None)


def read_tops(data: bytes, filename: str, well_name: str | None = None) -> pd.DataFrame:
    """Read a tops table (csv or xlsx) into columns ``name`` and ``md``; pick the rows of ``well_name`` if there is a well column."""
    buf = io.BytesIO(data)
    df = pd.read_excel(buf) if filename.lower().endswith((".xlsx", ".xls")) else pd.read_csv(buf, sep=None, engine="python")
    name_col, depth_col = _pick(list(df.columns), NAME_COLUMNS), _pick(list(df.columns), DEPTH_COLUMNS)
    if name_col is None or depth_col is None:
        raise ValueError(f"need a name column ({'/'.join(NAME_COLUMNS)}) and a depth column ({'/'.join(DEPTH_COLUMNS)}); "
                         f"found {list(df.columns)}")
    well_col = _pick(list(df.columns), WELL_COLUMNS)
    if well_col is not None:
        wells = list(df[well_col].astype(str).unique())
        if len(wells) > 1:
            hit = _match_well(wells, well_name)
            if hit is None:
                raise ValueError(f"the table has several wells ({', '.join(wells[:8])}...) and none matches "
                                 f"'{well_name}'. Upload the tops of this well only.")
            df = df[df[well_col].astype(str) == hit]
    out = pd.DataFrame({"name": df[name_col].astype(str), "md": pd.to_numeric(df[depth_col], errors="coerce")})
    return out.dropna().reset_index(drop=True)


def _clean_name(name: str) -> str:
    return re.sub(r"\s+top$", "", name.strip(), flags=re.IGNORECASE)


def tops_to_labels(depth: np.ndarray, tops: pd.DataFrame, known: dict[str, list[str]] | None = None
                   ) -> tuple[dict[str, np.ndarray], list[str]]:
    """GROUP / FORMATION label per depth sample from a tops table, plus the names that the models do not know.

    Names are matched to the training labels ignoring case; samples above the first top are ``UNKNOWN``."""
    canon = {lvl: {x.lower(): x for x in labels} for lvl, labels in (known or {}).items()}
    names = tops["name"].map(_clean_name)
    levels = np.where(names.str.contains(r"\bGP\.?$", case=False, regex=True), "group", "formation")
    labels, unmatched = {}, []
    for level in ("group", "formation"):
        sel = levels == level
        order = np.argsort(tops["md"].to_numpy()[sel], kind="stable")
        md = tops["md"].to_numpy()[sel][order]
        nm = np.array([canon.get(level, {}).get(n.lower(), n) for n in names[sel].to_numpy()[order]], dtype=object)
        unmatched += [n for n in nm if known and n not in set(known.get(level, []))]
        idx = np.searchsorted(md, depth, side="right") - 1
        labels[level] = np.where(idx >= 0, nm[np.clip(idx, 0, max(len(nm) - 1, 0))] if len(nm) else UNKNOWN, UNKNOWN)
    return labels, sorted(set(unmatched))


def attach_tops(well: Well, tops: pd.DataFrame, known: dict[str, list[str]] | None = None) -> list[str]:
    """Store GROUP / FORMATION labels in ``well.meta`` (needed by the stratigraphy variants). Returns unknown names."""
    labels, unmatched = tops_to_labels(well.depth, tops, known)
    well.meta.update(group=labels["group"].astype(str), formation=labels["formation"].astype(str))
    return unmatched


def has_strat(well: Well) -> bool:
    """True if the well carries stratigraphy labels."""
    return "group" in well.meta and "formation" in well.meta


def usable_variants(saved: dict[str, list[str]], target: str, with_strat: bool) -> list[str]:
    """Saved variants for ``target`` that can run on this well (stratigraphy variants need tops)."""
    return [v for v in saved.get(target, []) if with_strat or not v.endswith("_strat")]


def input_counts(well: Well, target: str) -> np.ndarray:
    """Number of input logs (the other standard logs) that have a value at each depth sample."""
    present = [np.isfinite(well.curves[c]) for c in CORE if c != target and c in well.curves]
    return np.sum(present, axis=0).astype(int) if present else np.zeros(len(well.depth), dtype=int)


def mask_sparse(pred: np.ndarray, counts: np.ndarray, min_inputs: int) -> np.ndarray:
    """NaN where fewer than ``min_inputs`` input logs exist. The models saw such depths rarely (under 10% of training
    rows had 4 or fewer inputs) and predict them badly."""
    return np.where(counts >= min_inputs, pred, np.nan)


def predict_log(model_dir, well: Well, target: str, variant: str, model=None) -> np.ndarray:
    """Predict ``target`` along ``well`` with a saved model; the well's own ``target`` curve (if any) is not used."""
    model = model or load_model(model_dir, target, variant)
    return model.predict(well.without(target), target).mean


def filled_curve(measured: np.ndarray | None, predicted: np.ndarray) -> np.ndarray:
    """Measured values where they exist, predictions elsewhere (gap filling)."""
    return predicted.copy() if measured is None else np.where(np.isfinite(measured), measured, predicted)
