"""Use the same masked-depth forecasting task for all frozen backbones."""
from __future__ import annotations

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


class DepthForecaster(BaseContender):
    workstream = "red_foundation_models"
    targets = ["RHOB", "DTC", "DTS"]
    # Depth is the only hard requirement: other core logs are covariates when present.
    # LITH is used only with facies=True, which the benchmark does not enable.
    required_inputs = ["DEPTH"]
    optional_inputs = list(CORE)
    inputs = [*CORE, "LITH"]
    context = 256
    horizon = 32

    def __init__(self, *, facies=False, shots=1):
        if shots < 0:
            raise ValueError("shots must be nonnegative")
        self.facies = facies
        self.shots = shots

    def fit(self, train_wells, target):
        self.target = target
        self.examples = []
        self.labels = sorted({label for w in train_wells if self.facies and "LITH" in w.curves
                              for label in np.unique(w.curves["LITH"]) if np.isfinite(label)})
        # At most one contiguous demonstration per training well. Never concatenate wells.
        for w in sorted(train_wells, key=lambda w: w.name):
            if len(self.examples) == self.shots:
                break
            if target not in w.curves:
                continue
            runs = ((start + i, start + j) for start, end in _segments(w.depth)
                    for i, j in _runs(np.isfinite(w.curves[target][start:end])))
            a, b = max(runs, key=lambda r: r[1] - r[0], default=(0, 0))
            if b > a:
                ix = np.arange(a, min(b, a + self.context))
                self.examples.append((w, ix))
        self.forecast = self.load_backend()

    def _features(self, well, ix):
        x = {"depth_md": well.depth[ix]}
        x.update({key: well.curves[key][ix] for key in CORE
                  if key != self.target and key in well.curves})
        if self.facies and "LITH" in well.curves:
            lith = well.curves["LITH"][ix]
            for label in self.labels:
                x[f"facies_{label:g}"] = (lith == label).astype(float)
            x["facies_unknown"] = (~np.isin(lith, self.labels)).astype(float)
        return x

    def _direction(self, history, features, query):
        # Forecast in native depth order (or its reverse) without crossing an acquisition break.
        h = history.copy()
        past = {}
        future = {}
        for key in sorted(features.keys() & query.keys()):
            a, b = features[key], query[key]
            valid = np.isfinite(a)
            known = np.isfinite(b)
            if not valid.any() or not known.any():
                continue
            fill = np.median(a[valid])
            past[key], future[key] = np.where(valid, a, fill), np.where(known, b, fill)
            if not (valid.all() and known.all()):
                past[key + "_missing"], future[key + "_missing"] = (~valid).astype(float), (~known).astype(float)
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
                        # Inverse distance to the observation anchoring this forecast.
                        weight = 1 / np.abs(well.depth[ix] - well.depth[ctx[-1]])
                        flanks.append((pred[:, ::direction], weight))
                if flanks:
                    q[:, ix] = sum(v * w for v, w in flanks) / sum(w for _, w in flanks)
                elif self.examples:
                    # The whole target is absent: each training example prompts a separate forecast.
                    # Restart each block from the demonstration, not a long synthetic target history.
                    for pos in range(0, ix.size, self.horizon):
                        chunk = ix[pos:pos + self.horizon]
                        query = self._features(well, chunk)
                        q[:, chunk] = np.mean([
                            self._direction(w.curves[target][ctx], self._features(w, ctx), query)
                            for w, ctx in self.examples], axis=0)
        return Prediction(q[1], (q[2] - q[0]) / 2.563103, dict(zip((0.1, 0.5, 0.9), q)))
