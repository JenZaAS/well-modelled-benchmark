"""Multilinear regression from the ML study, registered for the benchmark (base and trend features)."""
from workstreams.blue_boosted_trees.ml_study.registered import RegisteredML


class BlueMLLinear(RegisteredML):
    """Multilinear regression on the other logs + depth (clipped, standardised, missing values filled with the median
    plus a missing-value flag). Point predictions only."""
    name = "blue_ml_linear"
    kind = "linear"
    licence_restrictions = "none for this code; uses scikit-learn (BSD-3-Clause)"


class BlueMLLinearTrend(BlueMLLinear):
    """As ``blue_ml_linear`` plus trend features (10 m mean above/below, their difference, local spread)."""
    name = "blue_ml_linear_trend"
    trend = True
