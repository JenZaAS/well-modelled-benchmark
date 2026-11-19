"""Wellbore name normalisation for matching FORCE / LAS names to Sodir."""

from __future__ import annotations

import re

_DASHES = re.compile(r"[–—−‐-]+")
_SPACES = re.compile(r"\s+")
# "7S" -> "7 S", "1ST3" stays? we only insert space between digit and a single trailing letter group
_DIGIT_LETTER = re.compile(r"(\d)([A-Z]+)$")
_DIGIT_LETTER_MID = re.compile(r"(\d)([A-Z]+)(\s|$)")


def normalize_well_name(name: str) -> str:
    s = str(name).strip().upper()
    s = _DASHES.sub("-", s)
    s = _SPACES.sub(" ", s)
    return s


def name_aliases(name: str) -> list[str]:
    """Ordered unique aliases used when looking up a wellbore name."""
    n = normalize_well_name(name)
    # folder-safe 15_9-F-1 -> 15/9-F-1
    n = re.sub(r"^(\d+)_(\d+)(?=-)", r"\1/\2", n)
    out: list[str] = []

    def add(x: str) -> None:
        x = normalize_well_name(x)
        if x and x not in out:
            out.append(x)

    add(n)
    add(n.replace(" ", ""))
    # 35/3-7S <-> 35/3-7 S  (letter suffix after the well number)
    spaced = re.sub(r"(\d)([A-Z]+)$", r"\1 \2", n.replace(" ", ""))
    add(spaced)
    # 25/2-13T4 <-> 25/2-13 T4
    add(re.sub(r"(\d)([A-Z]+\d*)$", r"\1 \2", n.replace(" ", "")))
    return out


def parent_well_name(name: str) -> str | None:
    """Best-effort parent name if *name* looks like a sidetrack (T4, ST3, A, S…).

    Only used as a hint in notes, never as a silent coordinate source.
    """
    n = normalize_well_name(name)
    # strip a trailing sidetrack token: T4, ST3, A, S, B, ...
    m = re.match(r"^(.+?)(?:\s+)(ST\d+|T\d+|[A-Z]\d*)$", n)
    if m:
        return m.group(1)
    m = re.match(r"^(.+\d)(ST\d+|T\d+)$", n.replace(" ", ""))
    if m:
        # re-space if original had a slash form
        return normalize_well_name(m.group(1))
    return None
