"""Where data, work files and results live. Nothing here points inside the repo
except results/ (gitignored apart from small summaries).

Environment variables (all optional):
    FH2026_DATA     data root (default ~/fh2026-cache/data). On Windows: %USERPROFILE%\\fh2026-cache\\data
    FH2026_WORK     scratch/intermediate files (default ~/fh2026-cache/work)
    WMB_RESULTS     benchmark outputs (default <repo>/results)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("FH2026_CACHE", Path.home() / "fh2026-cache")).expanduser()


def data_root() -> Path:
    return Path(os.environ.get("FH2026_DATA", CACHE / "data")).expanduser()


def work_root() -> Path:
    return Path(os.environ.get("FH2026_WORK", CACHE / "work")).expanduser()


def results_root() -> Path:
    return Path(os.environ.get("WMB_RESULTS", REPO / "results")).expanduser()
