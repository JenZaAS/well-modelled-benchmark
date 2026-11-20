"""Per-depth state for the probability heads, built from a masked task and a base model's fill only.

Nothing here reads the hidden measurement: the flanks come from `Task.target_observed` (RHOB with the hidden
interval already removed), the fill and band from the base model, and the sonic check from DTC through the
fixed Gardner equation. Distances are in metres, so unequal depth spacing is handled.

Fields per hidden depth (all g/cm3 unless noted):
    base         the base model's point prediction at this depth
    position     0 at the top hidden sample of the gap, 1 at the bottom one
    dist_up/dn   metres to the nearest observed RHOB above / below (NaN if that side has none)
    flank_up/dn  that nearest observed RHOB value above / below
    flank_var    mean |value - side median| over up to 512 contiguous observed samples on each side
    width        the base model's native P10-P90 width here (only for models whose own band is used)
    sonic_diff   base - fixed Gardner density from DTC at this depth (NaN where DTC is unusable)
    dtc          1 where DTC gives a finite Gardner estimate, else 0
"""
from __future__ import annotations

import numpy as np

from workstreams.black_physics.granite import gaps as G

CTX = 512
FIELDS = ("base", "position", "dist_up", "dist_dn", "flank_up", "flank_dn", "flank_var", "width", "sonic_diff", "dtc")


def gap_rows(task, p50, lo=None, hi=None, gardner=None) -> dict:
    """One row per hidden depth of `task` (in depth order). p50/lo/hi/gardner: full-depth arrays or None."""
    depth = np.asarray(task.depth, float)
    obs = np.asarray(task.target_observed, float).copy()
    mask = np.asarray(task.predict_mask, bool)
    obs[mask] = np.nan                                   # never trust a caller to have blanked the hidden values
    n = len(depth)
    full = lambda x: np.full(n, np.nan) if x is None else np.asarray(x, float)   # noqa: E731
    p50, lo, hi, gardner = full(p50), full(lo), full(hi), full(gardner)
    cols = {f: [] for f in FIELDS}
    idx, piece_of = [], []
    for pi, (a, b, s, e) in enumerate(G.mask_pieces(depth, mask)):
        up = G.flank_run(obs, a, b, s, e, CTX, "above")
        dn = G.flank_run(obs, a, b, s, e, CTX, "below")
        dev = [np.abs(obs[i] - np.median(obs[i])) for i in (up, dn) if len(i)]
        fvar = float(np.concatenate(dev).mean()) if dev else np.nan
        j = np.arange(a, b)
        span = depth[b - 1] - depth[a]
        cols["base"].append(p50[j])
        cols["position"].append((depth[j] - depth[a]) / span if span > 0 else np.full(len(j), 0.5))
        cols["dist_up"].append(depth[j] - depth[up[-1]] if len(up) else np.full(len(j), np.nan))
        cols["dist_dn"].append(depth[dn[0]] - depth[j] if len(dn) else np.full(len(j), np.nan))
        cols["flank_up"].append(np.full(len(j), obs[up[-1]] if len(up) else np.nan))
        cols["flank_dn"].append(np.full(len(j), obs[dn[0]] if len(dn) else np.nan))
        cols["flank_var"].append(np.full(len(j), fvar))
        cols["width"].append(hi[j] - lo[j])
        cols["sonic_diff"].append(p50[j] - gardner[j])
        cols["dtc"].append(np.isfinite(gardner[j]).astype(float))
        idx.append(j)
        piece_of.append(np.full(len(j), pi))
    out = {f: np.concatenate(v) if v else np.zeros(0) for f, v in cols.items()}
    out["index"] = np.concatenate(idx) if idx else np.zeros(0, int)
    out["piece"] = np.concatenate(piece_of) if piece_of else np.zeros(0, int)
    out["depth"] = depth[out["index"]]
    return out


class FeaturePolicy:
    """Which state fields a numeric head reads, and the derived spread products.

    The products base x spread let a linear softmax head widen or narrow its distribution where a spread field is
    large or small (with logits linear in base, the curvature that sets the width is the coefficient of base).
    """

    def __init__(self, use_width: bool, sonic: bool, products: bool = True):
        self.use_width, self.sonic, self.products = use_width, sonic, products

    @property
    def raw(self) -> list[str]:
        f = ["base", "position", "dist_up", "dist_dn", "flank_up", "flank_dn", "flank_var"]
        if self.use_width:
            f.append("width")
        if self.sonic:
            f += ["sonic_diff", "dtc"]
        return f

    @property
    def spreads(self) -> list[str]:
        s = ["flank_var"] + (["width"] if self.use_width else []) + (["abs_sonic"] if self.sonic else [])
        return s if self.products else []

    def matrix(self, rows: dict) -> tuple[np.ndarray, list[str]]:
        cols, names = [], []
        for f in self.raw:
            cols.append(np.asarray(rows[f], float))
            names.append(f)
        for s in self.spreads:
            v = np.abs(np.asarray(rows["sonic_diff"], float)) if s == "abs_sonic" else np.asarray(rows[s], float)
            cols.append(np.asarray(rows["base"], float) * v)
            names.append(f"base*{s}")
        return np.column_stack(cols) if cols else np.zeros((0, 0)), names

    def as_dict(self) -> dict:
        return {"use_width": self.use_width, "sonic": self.sonic, "products": self.products, "raw": self.raw,
                "spreads": self.spreads}
