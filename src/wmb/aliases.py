"""Legacy ids for models that moved workstreams.

Results written before the move live under ``results/<old id>/`` and carry the old
``contender`` / ``workstream`` values. Look-ups by old name and loading of old
results go through this one table; the registry never lists the old names.
"""
from __future__ import annotations

MODEL_ALIASES = {
    "white_ttm": "red_ttm",
    "white_moment": "red_moment",
}
WORKSTREAM_ALIASES = {
    "white_finetuned_tsfm": "red_foundation_models",
}


def canonical_model(name):
    return MODEL_ALIASES.get(name, name)


def canonical_workstream(name):
    return WORKSTREAM_ALIASES.get(name, name)


def legacy_model_names(name: str) -> list[str]:
    """Old ids whose result folders may hold runs of ``name``."""
    return [old for old, new in MODEL_ALIASES.items() if new == name]


def canonical_row(row: dict) -> dict:
    """Copy of a scores row with the new model and workstream ids; ``legacy_contender`` records the old id."""
    out = dict(row)
    new = canonical_model(out.get("contender"))
    if new != out.get("contender"):
        out["legacy_contender"] = out["contender"]
        out["contender"] = new
    if isinstance(out.get("workstream"), str):
        out["workstream"] = canonical_workstream(out["workstream"])
    return out
