"""Base class for the ML-study models that are registered in the benchmark (``models/<name>/contender.py``).

A registered wrapper is an ``MLContender`` with fixed settings: no stratigraphy (benchmark wells carry no formation
labels) and a fixed feature set. It declares the benchmark metadata (name, targets, licence) and nothing else, so each
``models/<name>/contender.py`` stays a few lines long. The model code itself lives in this package (shared by the three
models and by the saved-model pickles, which record this module path).
"""
from __future__ import annotations

from wmb.curves import CORE

from .models import MLContender


class RegisteredML(MLContender):
    """Benchmark-ready variant of one ML-study model. Subclasses set ``name``, ``kind`` and ``trend``."""

    workstream = "blue_boosted_trees"
    kind = ""
    trend = False
    targets = ["RHOB", "DTC", "DTS"]                      # the benchmark's target curves
    required_inputs: list[str] = []                       # NaN-tolerant: runs with whatever inputs exist
    optional_inputs = list(CORE)                          # the target itself is dropped per target
    licence_name = "MIT"
    licence_url = "https://opensource.org/license/mit"
    licence_restrictions = "none for this code"

    def __init__(self) -> None:
        super().__init__(self.kind, use_strat=False, context=False, trend=self.trend)
