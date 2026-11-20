"""Legacy ids for models that moved workstreams.

Results written before the move live under ``results/<old id>/`` and carry the old
``contender`` / ``workstream`` values. Look-ups by old name and loading of old
results go through this one table; the registry never lists the old names.
"""
from __future__ import annotations

MODEL_ALIASES = {
    "white_ttm": "red_ttm",
    "white_moment": "red_moment",
    # Granite density-gap foundation-model rows, moved from black_physics to red_foundation_models
    "black_ttm_r3_frozen": "red_ttm_r3_frozen",
    "black_ttm_r3_tuned": "red_ttm_r3_quicktune",
    "black_ttm_r3_ft": "red_ttm_r3_finetune",
    "black_tspulse_frozen": "red_tspulse_frozen",
    "black_tspulse_tuned": "red_tspulse_quicktune",
    "black_tspulse_ft": "red_tspulse_finetune",
    "black_timestral_target_only": "red_timestral_target_only",
    # ... then renamed in red: _tuned (the small fine-tune) -> _quicktune, _ft (the stronger one) -> _finetune
    "red_ttm_r3_tuned": "red_ttm_r3_quicktune",
    "red_ttm_r3_ft": "red_ttm_r3_finetune",
    "red_tspulse_tuned": "red_tspulse_quicktune",
    "red_tspulse_ft": "red_tspulse_finetune",
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
