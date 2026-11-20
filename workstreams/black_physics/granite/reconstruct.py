"""Gap reconstruction with a Granite backbone: the inference code shared by frozen and tuned rows.

The backend is injected (`forecast(contexts) -> [n, 9, horizon]` for TTM, `reconstruct(windows) -> [n, W]` for
TSPulse), so the geometry below is plain NumPy and is unit-tested with fake backends.

Every piece (one hidden run inside one acquisition segment) gets one status:
    two_sided         model output from measured context on both sides
    one_sided_above   model output from the shallow side only (TTM: forward forecast; TSPulse: shifted window)
    one_sided_below   model output from the deep side only
    fallback_interp   the shared straight-line interpolation, because the model cannot cover the piece
                      (TTM horizon > 336 samples; TSPulse hole >= the 512-sample window; or both flanks
                      too short - TTM: < `ttm_min_context` contiguous samples, TSPulse: < `min_flank_samples`
                      observed in the window - while measured values exist on both sides)
    abstain           no usable target context; prediction is NaN (whole missing curve, missing_curve tasks)

The hidden target is never read: inputs come from `task.target_observed`, which is additionally blanked on
`predict_mask` here before any statistic, padding or window is formed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from workstreams.black_physics.granite import gaps as G
from workstreams.black_physics.granite.config import CONFIG

MODEL_STATUSES = ("two_sided", "one_sided_above", "one_sided_below")


@dataclass
class Piece:
    a: int
    b: int
    s: int
    e: int
    status: str
    p50: np.ndarray
    q_raw: np.ndarray | None = None          # TTM: blended raw quantiles [9, H] (0.1..0.9)
    dir_above: np.ndarray | None = None      # TTM: directional medians in increasing depth
    dir_below: np.ndarray | None = None
    ctx_above: np.ndarray = field(default_factory=lambda: np.zeros(0, int))   # indices of the context used
    ctx_below: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    ctx_above_m: float = 0.0
    ctx_below_m: float = 0.0
    flank_med_above: float = float("nan")
    flank_med_below: float = float("nan")
    flank_var: float = float("nan")          # mean |value - side median| over the context used
    reason: str = ""

    @property
    def n(self) -> int:
        return self.b - self.a

    @property
    def model_made(self) -> bool:
        return self.status in MODEL_STATUSES

    @property
    def raw_lo(self):
        return None if self.q_raw is None else self.q_raw[0]

    @property
    def raw_hi(self):
        return None if self.q_raw is None else self.q_raw[-1]


def _blank(task) -> np.ndarray | None:
    if task.target_observed is None:
        return None
    obs = np.asarray(task.target_observed, float).copy()
    obs[np.asarray(task.predict_mask, bool)] = np.nan
    return obs


def _span_m(depth, idx) -> float:
    if not len(idx):
        return 0.0
    d = np.asarray(depth, float)
    step = float(np.median(np.diff(d))) if len(d) > 1 else 0.0
    return float(d[idx[-1]] - d[idx[0]] + step)


def _flank_stats(piece: Piece, depth, obs) -> None:
    vals = []
    for side in ("above", "below"):
        idx = getattr(piece, f"ctx_{side}")
        setattr(piece, f"ctx_{side}_m", _span_m(depth, idx))
        if len(idx):
            v = obs[idx]
            med = float(np.median(v))
            setattr(piece, f"flank_med_{side}", med)
            vals.append(np.abs(v - med))
    piece.flank_var = float(np.concatenate(vals).mean()) if vals else float("nan")


def _distance_weight(depth, a: int, b: int, top: int, bot: int) -> np.ndarray:
    """u in [0, 1]: fractional depth position between the two measured boundaries (0 = shallow boundary)."""
    d = np.asarray(depth, float)
    return np.clip((d[a:b] - d[top]) / (d[bot] - d[top]), 0.0, 1.0)


def _fallback(depth, obs, a, b, s, e, why: str) -> Piece:
    line = G.linear_fill(depth, obs, a, b, s, e)
    if line is None:
        return Piece(a, b, s, e, "abstain", np.full(b - a, np.nan), reason=f"{why}; no measured target on both sides")
    return Piece(a, b, s, e, "fallback_interp", line, reason=why)


class TTMReconstructor:
    """Two directional TTM forecasts, the deeper one reversed, blended by physical distance."""

    arm = "ttm"

    def __init__(self, forecast, horizon: int = CONFIG.ttm_horizon, context: int = CONFIG.ttm_context,
                 min_flank: int = CONFIG.ttm_min_context):
        self.forecast, self.horizon, self.context, self.min_flank = forecast, horizon, context, min_flank

    def run(self, tasks) -> list[list[Piece]]:
        plans, contexts = [], []
        for t in tasks:
            obs = _blank(t)
            depth = np.asarray(t.depth, float)
            out = []
            if obs is None:      # whole-curve task: there is no RHOB context at all
                out = [Piece(a, b, s, e, "abstain", np.full(b - a, np.nan), reason="no target context (missing curve)")
                       for a, b, s, e in G.mask_pieces(depth, t.predict_mask)]
                plans.append((t, depth, obs, out, []))
                continue
            jobs = []
            for a, b, s, e in G.mask_pieces(depth, t.predict_mask):
                H = b - a
                up = G.flank_run(obs, a, b, s, e, self.context, "above")
                dn = G.flank_run(obs, a, b, s, e, self.context, "below")
                use_up, use_dn = len(up) >= self.min_flank, len(dn) >= self.min_flank
                if H > self.horizon:
                    p = _fallback(depth, obs, a, b, s, e, f"gap of {H} samples exceeds the TTM horizon {self.horizon}")
                elif not (use_up or use_dn):
                    p = _fallback(depth, obs, a, b, s, e, f"flanks shorter than {self.min_flank} samples")
                else:
                    status = "two_sided" if use_up and use_dn else ("one_sided_above" if use_up else "one_sided_below")
                    p = Piece(a, b, s, e, status, np.full(H, np.nan), ctx_above=up if use_up else up[:0],
                              ctx_below=dn if use_dn else dn[:0])
                    if use_up:
                        jobs.append((len(out), "above", len(contexts)))
                        contexts.append(obs[up])
                    if use_dn:
                        jobs.append((len(out), "below", len(contexts)))
                        contexts.append(obs[dn][::-1])          # deeper flank, reversed: forecast runs upward
                out.append(p)
            plans.append((t, depth, obs, out, jobs))
        q_all = self.forecast(contexts) if contexts else np.zeros((0, 9, self.horizon))
        results = []
        for t, depth, obs, out, jobs in plans:
            got: dict[int, dict] = {}
            for i, side, j in jobs:
                p = out[i]
                q = np.asarray(q_all[j][:, :p.n], float)
                got.setdefault(i, {})[side] = q if side == "above" else q[:, ::-1]   # back to increasing depth
            for i, sides in got.items():
                p = out[i]
                if "above" in sides and "below" in sides:
                    u = _distance_weight(depth, p.a, p.b, int(p.ctx_above[-1]), int(p.ctx_below[0]))
                    q = (1 - u) * sides["above"] + u * sides["below"]
                else:
                    q = sides.get("above", sides.get("below"))
                p.q_raw = q
                p.p50 = q[len(q) // 2].copy()
                p.dir_above = sides["above"][len(q) // 2].copy() if "above" in sides else None
                p.dir_below = sides["below"][len(q) // 2].copy() if "below" in sides else None
            for p in out:
                _flank_stats(p, depth, obs if obs is not None else np.full(len(depth), np.nan))
            results.append(out)
        return results


class TSPulseReconstructor:
    """One 512-sample window around the gap, the gap kept intact and masked; read the reconstruction."""

    arm = "tspulse"

    def __init__(self, reconstruct, window: int = CONFIG.tspulse_window, min_flank: int = CONFIG.min_flank_samples):
        self.reconstruct, self.window, self.min_flank = reconstruct, window, min_flank

    def place(self, a: int, b: int, s: int, e: int) -> int:
        """First index of the window: gap centred, shifted to stay inside [s, e); may extend past e if the segment is short."""
        W, H = self.window, b - a
        w0 = a - (W - H) // 2
        if e - s >= W:
            w0 = min(max(w0, s), e - W)
        else:
            w0 = s
        return int(w0)

    def window_values(self, obs, a, b, s, e) -> tuple[np.ndarray, int]:
        W = self.window
        w0 = self.place(a, b, s, e)
        idx = np.arange(w0, w0 + W)
        inside = (idx >= s) & (idx < e)
        vals = np.full(W, np.nan)
        vals[inside] = obs[idx[inside]]
        vals[(idx >= a) & (idx < b)] = np.nan       # the gap is never observed
        return vals, w0

    def run(self, tasks) -> list[list[Piece]]:
        plans, windows = [], []
        for t in tasks:
            obs = _blank(t)
            depth = np.asarray(t.depth, float)
            out, jobs = [], []
            for a, b, s, e in G.mask_pieces(depth, t.predict_mask):
                H = b - a
                if obs is None:
                    out.append(Piece(a, b, s, e, "abstain", np.full(H, np.nan), reason="no target context (missing curve)"))
                    continue
                if H >= self.window:
                    out.append(_fallback(depth, obs, a, b, s, e, f"gap of {H} samples fills the {self.window}-sample window"))
                    continue
                vals, w0 = self.window_values(obs, a, b, s, e)
                idx = np.arange(w0, w0 + self.window)
                seen = np.isfinite(vals)
                up, dn = idx[seen & (idx < a)], idx[seen & (idx >= b)]
                use_up, use_dn = len(up) >= self.min_flank, len(dn) >= self.min_flank
                if not (use_up or use_dn):
                    out.append(_fallback(depth, obs, a, b, s, e, f"fewer than {self.min_flank} observed samples per side in the window"))
                    continue
                status = "two_sided" if use_up and use_dn else ("one_sided_above" if use_up else "one_sided_below")
                out.append(Piece(a, b, s, e, status, np.full(H, np.nan), ctx_above=up, ctx_below=dn))
                jobs.append((len(out) - 1, w0, len(windows)))
                windows.append(vals)
            plans.append((depth, obs, out, jobs))
        rec = self.reconstruct(np.asarray(windows)) if windows else np.zeros((0, self.window))
        results = []
        for depth, obs, out, jobs in plans:
            for i, w0, j in jobs:
                p = out[i]
                p.p50 = np.asarray(rec[j][p.a - w0:p.b - w0], float).copy()
            for p in out:
                _flank_stats(p, depth, obs if obs is not None else np.full(len(depth), np.nan))
            results.append(out)
        return results


def make_reconstructor(backend):
    if backend.arm == "ttm":
        return TTMReconstructor(backend.forecast, backend.horizon, backend.context)
    return TSPulseReconstructor(backend.reconstruct, backend.window)
