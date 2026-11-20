"""Blue team starter: gradient-boosted trees (XGBoost / LightGBM). One depth sample = one row.

Starter: XGBoost if installed, else scikit-learn HistGradientBoosting. Features = the other
input curves (log10 for resistivity) + depth. Uncertainty: two extra quantile models
(16% and 84%) give sigma = (q84 - q16) / 2.
See experiments/poc-ml/ for the earlier PoC (window features, lithology, binned density).
"""
from __future__ import annotations

import numpy as np

from wmb.contender import BaseContender, Prediction
from wmb.curves import LOG_SCALE

FEATURES = ["GR", "RDEP", "NPHI", "DTC", "RHOB", "PEF", "CALI", "DTS"]


def _x(well, feats):
    cols = []
    for c in feats:
        v = well.curves.get(c, np.full_like(well.depth, np.nan, dtype=float))
        cols.append(np.log10(np.clip(v, 1e-3, None)) if c in LOG_SCALE else v)
    return np.column_stack(cols + [well.depth])


class GBT(BaseContender):
    name, workstream = "blue_gbt", "blue_boosted_trees"
    inputs = FEATURES
    licence_name = "MIT"
    licence_url = "https://opensource.org/license/mit"
    licence_restrictions = "none for this code; uses XGBoost (Apache-2.0), or scikit-learn (BSD-3-Clause) as fallback"

    def _model(self, quantile=None):
        try:
            import xgboost as xgb
            kw = {"objective": "reg:quantileerror", "quantile_alpha": quantile} if quantile else {}
            return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1, **kw)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            kw = {"loss": "quantile", "quantile": quantile} if quantile else {}
            return HistGradientBoostingRegressor(max_iter=200, **kw)

    def fit(self, train_wells, target):
        self.feats = [f for f in FEATURES if f != target]
        X = np.vstack([_x(w, self.feats) for w in train_wells if target in w.curves])
        y = np.concatenate([w.curves[target] for w in train_wells if target in w.curves])
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
        self.m = self._model().fit(X, y)
        self.lo = self._model(0.16).fit(X, y)
        self.hi = self._model(0.84).fit(X, y)

    def predict(self, well, target):
        X = _x(well, self.feats)
        mean = self.m.predict(X)
        sigma = np.abs(self.hi.predict(X) - self.lo.predict(X)) / 2
        return Prediction(mean, sigma)
