"""Resolve the red foundation-model weight cache (TTM / MOMENT base checkpoints and fine-tunes)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

WORKSTREAM = Path(__file__).resolve().parent
REPO = WORKSTREAM.parents[1]
DEFAULT_CACHE = WORKSTREAM / "weights"
# Folder used before TTM / MOMENT moved here; only consulted while it still exists.
LEGACY_CACHE = REPO / "workstreams" / "white_finetuned_tsfm" / "weights"


def _has_content(folder: Path) -> bool:
    return any((folder / sub).is_dir() for sub in ("hf", "ttm"))


def red_cache_dir() -> Path:
    """Weight folder, in order: ``WMB_RED_CACHE``, legacy ``WMB_WHITE_CACHE``,
    ``workstreams/red_foundation_models/weights/``, then the old
    ``workstreams/white_finetuned_tsfm/weights/`` if it still holds ``hf/`` or
    ``ttm/`` and the new default holds none, so a half-finished move keeps loading.
    """
    for var in ("WMB_RED_CACHE", "WMB_WHITE_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser().resolve()
    if not _has_content(DEFAULT_CACHE) and _has_content(LEGACY_CACHE):
        return LEGACY_CACHE
    return DEFAULT_CACHE


def red_hf_home() -> Path:
    """Hugging Face cache for base-model downloads, under the weight dir."""
    return red_cache_dir() / "hf"


white_cache_dir = red_cache_dir
white_hf_home = red_hf_home


def ttm_finetuned_dir(target: str) -> Path:
    return red_cache_dir() / "ttm" / str(target).upper()


def ttm_probabilistic_dir(target: str) -> Path:
    """Quantile-head fine-tune (q05/q50/q95), sibling of the point-forecast folder."""
    return red_cache_dir() / "ttm" / f"{str(target).upper()}-probabilistic"


def has_finetuned_ttm(target: str) -> bool:
    folder = ttm_finetuned_dir(target)
    return folder.is_dir() and (folder / "config.json").is_file()


def has_probabilistic_ttm(target: str) -> bool:
    folder = ttm_probabilistic_dir(target)
    return folder.is_dir() and (folder / "config.json").is_file()


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO.resolve())
        return True
    except ValueError:
        return False


def assert_gitignored(path: Path) -> Path:
    """If ``path`` is inside the repo, require ``git check-ignore`` to match.

    Paths outside the repo (a ``WMB_RED_CACHE`` override) are allowed.
    """
    path = path.resolve()
    if not _inside_repo(path):
        return path
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", str(path)],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{path} is inside the repo but is not git-ignored. "
            "Write weights only under workstreams/red_foundation_models/weights/ "
            "(see that folder's .gitignore) or set WMB_RED_CACHE outside the repo."
        )
    return path
