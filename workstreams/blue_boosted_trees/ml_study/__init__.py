"""Blue-team ML study: gradient boosting vs multilinear regression vs random forest.

Every model predicts each of the 9 core logs from the other logs (+ depth), with and
without stratigraphy (GROUP / FORMATION). See the workstream README for how to run it.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "src", _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
