"""Blue team registration file: the runner finds the blue models through the CONTENDERS list below.

Each model lives in its own folder under models/ (wrapper, model.yaml, LICENSE.md, README.md):

    models/xgboost_quantile/    blue_gbt              XGBoost starter with quantile-based uncertainty
    models/xgboost_gbt/         blue_ml_gbt, blue_ml_gbt_trend
    models/linear_regression/   blue_ml_linear, blue_ml_linear_trend
    models/random_forest/       blue_ml_rf

The last three are the ML-study models; their shared code (features, estimators, saved-model store, the app) is in
ml_study/. See the README of this folder. Earlier PoC: experiments/poc-ml/.
"""
from __future__ import annotations

from .models.linear_regression.contender import BlueMLLinear, BlueMLLinearTrend
from .models.random_forest.contender import BlueMLRf
from .models.xgboost_gbt.contender import BlueMLGbt, BlueMLGbtTrend
from .models.xgboost_quantile.contender import GBT


CONTENDERS = [GBT, BlueMLGbt, BlueMLGbtTrend, BlueMLLinear, BlueMLLinearTrend, BlueMLRf]
