"""XGBoost gradient boosting from the ML study, registered for the benchmark (base and trend features)."""
from workstreams.blue_boosted_trees.ml_study.registered import RegisteredML


class BlueMLGbt(RegisteredML):
    """XGBoost on the other logs + depth; one depth sample = one row. Point predictions only."""
    name = "blue_ml_gbt"
    kind = "gbt"
    licence_restrictions = "none for this code; uses XGBoost (Apache-2.0)"


class BlueMLGbtTrend(BlueMLGbt):
    """As ``blue_ml_gbt`` plus trend features (10 m mean above/below, their difference, local spread)."""
    name = "blue_ml_gbt_trend"
    trend = True
