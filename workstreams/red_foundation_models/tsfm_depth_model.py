"""Masked-depth forecasting adapter for the TTM and MOMENT backbones (``red_ttm``, ``red_moment``).

Same gap/flank logic and quantile contract as ``depth_model.DepthForecaster``, plus
quantile-band support for fine-tuned TTM. Kept as a separate module so the
Chronos-2 / TimesFM / Timestral adapter is untouched.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

from wmb.contender import BaseContender, Prediction
from wmb.curves import CORE


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def _segments(depth):
    delta = np.diff(depth)
    if not np.all(np.isfinite(depth)) or np.any(delta <= 0):
        raise ValueError("Depth must be finite and strictly increasing (metres MD)")
    cuts = np.flatnonzero(delta > 3 * np.median(delta)) + 1 if delta.size else []
    return zip([0, *cuts], [*cuts, len(depth)])


def prediction_from_band(q: np.ndarray, levels=(0.1, 0.5, 0.9)) -> Prediction:
    """Turn a (3, n) quantile stack into a Prediction the benchmark can score.

    When ``levels`` is (0.05, 0.5, 0.95), p05/p95 are set so cover_90 / sharpness
    use the model's 90% band. Otherwise only std is set and the contract fills a
    Gaussian 90% interval (the zero-shot point-forecast path).
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] != 3:
        raise ValueError(f"expected quantile stack of shape (3, n), got {q.shape}")
    levels = tuple(float(x) for x in levels)
    lo, mid, hi = levels
    mass = hi - lo
    z = NormalDist().inv_cdf(0.5 + mass / 2.0) if 0 < mass < 1 else 0.0
    std = (q[2] - q[0]) / (2.0 * z) if z > 0 else np.zeros_like(q[1])
    quantiles = {lo: q[0], mid: q[1], hi: q[2]}
    p05 = q[0] if abs(lo - 0.05) < 1e-9 else None
    p95 = q[2] if abs(hi - 0.95) < 1e-9 else None
    return Prediction(mean=q[1], std=std, quantiles=quantiles, p50=q[1], p05=p05, p95=p95)


class DepthForecaster(BaseContender):
    workstream = "red_foundation_models"
    targets = ["RHOB", "DTC", "DTS"]
    required_inputs = ["DEPTH"]
    optional_inputs = list(CORE)
    inputs = [*CORE, "LITH"]
    context = 512
    horizon = 96
    # Point models wrap a Gaussian p10/p90 around p50. Quantile-head models
    # set this to (0.05, 0.5, 0.95) so the scored 90% band is the raw q05/q95.
    band_levels = (0.1, 0.5, 0.9)

    def __init__(self, *, shots=1):
        if shots < 0:
            raise ValueError("shots must be nonnegative")
        self.shots = shots
        self.sigma = 0.0

    def fit(self, train_wells, target):
        self.target = target
        self.examples = []
        vals = []
        for w in sorted(train_wells, key=lambda w: w.name):
            if target not in w.curves:
                continue
            y = np.asarray(w.curves[target], float)
            finite = y[np.isfinite(y)]
            if finite.size:
                vals.append(finite)
            if len(self.examples) < self.shots:
                runs = ((start + i, start + j) for start, end in _segments(w.depth)
                        for i, j in _runs(np.isfinite(y[start:end])))
                a, b = max(runs, key=lambda r: r[1] - r[0], default=(0, 0))
                if b > a:
                    self.examples.append((w, np.arange(a, min(b, a + self.context))))
        if vals:
            stacked = np.concatenate(vals)
            # Scale for p10/p90 around a point forecast (TTM/MOMENT are point models).
            d = np.diff(stacked)
            self.sigma = float(np.nanstd(d)) if d.size else float(np.nanstd(stacked))
        self.forecast = self._quantiles(self.load_backend())

    def _quantiles(self, forecast):
        sigma = float(self.sigma or 0.0)
        z = 1.281551565  # ~80% half-width, so p10/p90 flank a Gaussian around p50

        def wrapped(history, past, future, horizon):
            out = np.asarray(forecast(history, past, future, horizon), dtype=float)
            if out.ndim == 1:
                q50 = out
            elif out.shape[0] == 3:
                return out
            else:
                raise ValueError(f"{self.name} backend returned shape {out.shape}")
            width = z * sigma if sigma > 0 else np.zeros_like(q50)
            return np.stack([q50 - width, q50, q50 + width])

        return wrapped

    def _features(self, well, ix):
        x = {"depth_md": well.depth[ix]}
        x.update({key: well.curves[key][ix] for key in CORE
                  if key != self.target and key in well.curves})
        return x

    def _direction(self, history, features, query):
        h = history.copy()
        past, future = {}, {}
        for key in sorted(features.keys() & query.keys()):
            a, b = features[key], query[key]
            valid, known = np.isfinite(a), np.isfinite(b)
            if not valid.any() or not known.any():
                continue
            fill = np.median(a[valid])
            past[key], future[key] = np.where(valid, a, fill), np.where(known, b, fill)
        n = len(query["depth_md"])
        out = np.empty((3, n))
        for i in range(0, n, self.horizon):
            end = min(i + self.horizon, n)
            block = {k: v[i:end] for k, v in future.items()}
            q = np.asarray(self.forecast(h, past, block, end - i), dtype=float)
            if q.shape != (3, end - i) or not np.isfinite(q).all() or np.any(np.diff(q, axis=0) < 0):
                raise ValueError(f"{self.name} returned invalid forecast quantiles")
            out[:, i:end] = q
            h = np.r_[h, q[1]][-self.context:]
            past = {k: np.r_[v, block[k]][-self.context:] for k, v in past.items()}
        return out

    def predict(self, well, target):
        if target != self.target:
            raise ValueError(f"Fitted for {self.target}, not {target}")
        y = well.curves.get(target, np.full(len(well.depth), np.nan))
        obs = np.isfinite(y)
        q = np.tile(y, (3, 1))
        for start, end in _segments(well.depth):
            for i, j in _runs(~obs[start:end]):
                a, b = start + i, start + j
                ix = np.arange(a, b)
                left = np.arange(a - 1, max(start - 1, a - self.context - 1), -1)
                right = np.arange(b, min(end, b + self.context))
                flanks = []
                for ctx, direction in ((left, 1), (right, -1)):
                    ctx = ctx[:np.sum(np.logical_and.accumulate(obs[ctx]))][::-1]
                    if ctx.size:
                        pred = self._direction(y[ctx], self._features(well, ctx),
                                               self._features(well, ix[::direction]))
                        weight = 1 / np.abs(well.depth[ix] - well.depth[ctx[-1]])
                        flanks.append((pred[:, ::direction], weight))
                if flanks:
                    q[:, ix] = sum(v * w for v, w in flanks) / sum(w for _, w in flanks)
                elif self.examples:
                    for pos in range(0, ix.size, self.horizon):
                        chunk = ix[pos:pos + self.horizon]
                        query = self._features(well, chunk)
                        q[:, chunk] = np.mean([
                            self._direction(w.curves[target][ctx], self._features(w, ctx), query)
                            for w, ctx in self.examples], axis=0)
        return prediction_from_band(q, getattr(self, "band_levels", (0.1, 0.5, 0.9)))

