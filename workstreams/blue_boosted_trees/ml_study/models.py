"""The three ML models behind one contender class.

    gbt     XGBoost gradient-boosted trees (handles missing values natively)
    linear  multilinear regression (median imputation + missing-value flags)
    rf      scikit-learn random forest (handles missing values natively)

Each model can be built with or without context features (the logs 5 m above/below and 5/20 m moving averages) or
trend features (10 m mean above/below, their difference and the local spread; see ``features.py``) and with or
without stratigraphy. All implement ``wmb.contender.BaseContender``,
so they can also be used with the shared benchmark code. Training rows are thinned evenly to
``max_train_rows`` (depth samples are strongly autocorrelated, so little is lost); use 0 to keep all.
Resistivity (RDEP, RMED) is trained in log10 and predicted back in ohm.m.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from wmb.contender import BaseContender, Prediction
from wmb.curves import CORE, LOG_SCALE
from wmb.io import Well

from .features import FeatureBuilder
from .metrics import target_space

MODEL_KINDS = ("gbt", "linear", "rf")
MODEL_LABELS = {"gbt": "Gradient boosting (XGBoost)", "linear": "Multilinear regression", "rf": "Random forest"}


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip the first ``n_columns`` columns to their training percentiles (NaN stays NaN).

    Logs have spikes (PEF up to 380, GR over 1000) that dominate a least-squares fit.
    The remaining columns (0/1 stratigraphy flags) are left alone.
    """

    def __init__(self, n_columns: int, lower: float = 0.5, upper: float = 99.5) -> None:
        self.n_columns, self.lower, self.upper = n_columns, lower, upper

    def fit(self, X: np.ndarray, y=None) -> "Winsorizer":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns
            self.low_, self.high_ = np.nanpercentile(X[:, :self.n_columns], [self.lower, self.upper], axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        X[:, :self.n_columns] = np.clip(X[:, :self.n_columns], self.low_, self.high_)
        return X


class MedianImputer(BaseEstimator, TransformerMixin):
    """Fill NaN with the training median and append a 0/1 flag for every column that had NaN in training.

    Same result as ``SimpleImputer(strategy="median", add_indicator=True)`` but works column by column:
    scikit-learn's version needs several full-size copies, which ran out of memory with ~120 columns on 8 GB.
    """

    def fit(self, X: np.ndarray, y=None) -> "MedianImputer":
        missing = np.isnan(X)
        self.keep_ = ~missing.all(axis=0)                                     # drop columns with no data at all
        self.median_ = np.array([np.nanmedian(X[:, j]) for j in np.flatnonzero(self.keep_)])
        self.flag_cols_ = np.flatnonzero(missing.any(axis=0) & self.keep_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        flags = np.isnan(X[:, self.flag_cols_]).astype(np.float64)
        out = np.array(X[:, self.keep_], dtype=np.float64, copy=True)
        for j, med in enumerate(self.median_):
            out[np.isnan(out[:, j]), j] = med
        return np.hstack([out, flags])


def make_estimator(kind: str, seed: int = 0, n_continuous: int = 0):
    """Build an unfitted scikit-learn style regressor. ``n_continuous`` = leading non-flag columns (linear model only)."""
    if kind == "gbt":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                                n_jobs=-1, random_state=seed)
    if kind == "linear":
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        # float64 + standardised columns: with raw float32 columns (depth in thousands, flags 0/1)
        # the least-squares solve silently ignored the small stratigraphy columns.
        return make_pipeline(Winsorizer(n_continuous), MedianImputer(), StandardScaler(), LinearRegression())
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=60, min_samples_leaf=5, max_features=0.5, n_jobs=-1,
                                     random_state=seed)
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


def variant_name(kind: str, use_strat: bool = False, context: bool = False, trend: bool = False) -> str:
    """Short id used in result files, e.g. ``rf``, ``rf_strat``, ``gbt_ctx``, ``gbt_trend_strat``."""
    return kind + ("_ctx" if context else "") + ("_trend" if trend else "") + ("_strat" if use_strat else "")


class MLContender(BaseContender):
    """One of the three models, optionally with context or trend (neighbouring-depth) features and/or stratigraphy."""

    workstream = "blue_boosted_trees"

    def __init__(self, kind: str, use_strat: bool = False, max_train_rows: int = 250_000, seed: int = 0,
                 context: bool = False, trend: bool = False) -> None:
        if kind not in MODEL_KINDS:
            raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")
        self.kind, self.use_strat, self.max_train_rows, self.seed = kind, use_strat, max_train_rows, seed
        self.context, self.trend = context, trend
        if self.name == BaseContender.name:   # registered wrappers (models/<name>/) set their own name
            self.name = f"blue_{variant_name(kind, use_strat, context, trend)}"
        self.n_train_rows = 0

    def fit(self, train_wells: list[Well], target: str) -> None:
        """Fit on every training well that has ``target``, using the other core curves as inputs."""
        self.inputs = [c for c in CORE if c != target]
        wells = [w for w in train_wells if w.has(target)]
        if not wells:
            raise ValueError(f"no training well has {target}")
        self.features = FeatureBuilder(self.inputs, self.use_strat, "onehot" if self.kind == "linear" else "ordinal",
                                       self.context, self.trend).fit(wells)
        total = sum(int(np.isfinite(w.curves[target]).sum()) for w in wells)
        step = max(1, int(np.ceil(total / self.max_train_rows))) if self.max_train_rows else 1
        X_parts, y_parts = [], []
        for w in wells:
            ok = np.isfinite(w.curves[target])
            X_parts.append(self.features.transform(w)[ok][::step])
            y_parts.append(w.curves[target][ok][::step])
        X, y = np.vstack(X_parts), target_space(target, np.concatenate(y_parts))
        self.n_train_rows = len(y)
        self.model = make_estimator(self.kind, self.seed, self.features.n_continuous).fit(X, y)

    def predict(self, well: Well, target: str) -> Prediction:
        """Predict ``target`` at every depth of ``well`` (the target curve itself is never used)."""
        mean = np.asarray(self.model.predict(self.features.transform(well)), dtype=float)
        if target in LOG_SCALE:   # trained on log10: back to ohm.m, within 0.001 to 10,000
            mean = 10 ** np.clip(mean, -3, 4)
        return Prediction(mean)
