"""wmb: shared code for the well-modelled-benchmark.

Stable interface (see src/wmb/INTERFACE.md). Everyone imports this package;
changes are announced in the team chat before they are pushed.
"""
from wmb.contender import Contender, Prediction
from wmb.curves import CORE, match_property
from wmb.io import Well

__all__ = ["Contender", "Prediction", "Well", "CORE", "match_property"]
__version__ = "0.1.0"
