"""Dataset packs: the shared canonical curve dictionary and per-dataset loaders.

Per-dataset packs (README, mapping, checksums, fetch script) live in ``datasets/<id>/``
at the repo root; loaders live here.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

__all__ = ["load_canonical_spec", "load_force2020"]


@lru_cache(maxsize=1)
def _load_canonical_spec() -> dict:
    import yaml

    text = resources.files(__package__).joinpath("canonical_curves.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_canonical_spec() -> dict:
    """The shared canonical dictionary (names, units, aliases, guardrails) from package data."""
    return _load_canonical_spec()


def load_force2020(*args, **kwargs):
    """Load FORCE 2020 wells; see ``wmb.datasets.force2020.load_force2020``."""
    from wmb.datasets.force2020 import load_force2020 as _load

    return _load(*args, **kwargs)
