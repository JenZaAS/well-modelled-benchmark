"""wmb: shared code for the well-modelled-benchmark.

Stable interface (see src/wmb/INTERFACE.md). Everyone imports this package;
changes are announced in the team chat before they are pushed.

Heavy imports (numpy, pandas) are loaded lazily so `python -m wmb.paths`
and `python -m wmb.viewer` work with only the packages those commands need.
"""
from __future__ import annotations

__all__ = ["Contender", "Prediction", "Well", "CORE", "match_property"]
__version__ = "0.1.0"

_LAZY = {
    "Contender": ("wmb.contender", "Contender"),
    "Prediction": ("wmb.contender", "Prediction"),
    "Well": ("wmb.io", "Well"),
    "CORE": ("wmb.curves", "CORE"),
    "match_property": ("wmb.curves", "match_property"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    import importlib

    value = getattr(importlib.import_module(module), attr)
    globals()[name] = value
    return value
