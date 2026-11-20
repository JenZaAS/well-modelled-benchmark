"""Unit tests that do not import granite-tsfm or download weights."""
import numpy as np
import pytest

from workstreams.red_foundation_models import cache
from workstreams.red_foundation_models.models.ttm.backend import _vector
from workstreams.red_foundation_models.cache import (
    assert_gitignored,
    has_finetuned_ttm,
    has_probabilistic_ttm,
    red_cache_dir,
    red_hf_home,
    ttm_finetuned_dir,
    ttm_probabilistic_dir,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("WMB_RED_CACHE", raising=False)
    monkeypatch.delenv("WMB_WHITE_CACHE", raising=False)
    monkeypatch.setattr(cache, "LEGACY_CACHE", tmp_path / "no-legacy")


def test_vector_ok():
    v = _vector([1.0, 2.0, 3.0], "history")
    assert v.dtype == np.float32 and v.shape == (3,)


def test_vector_rejects_empty():
    with pytest.raises(ValueError):
        _vector([], "history")


def test_default_cache_is_workstream_weights():
    path = ttm_finetuned_dir("RHOB")
    root = red_cache_dir()
    assert root.name == "weights"
    assert root.parent.name == "red_foundation_models"
    assert path == root / "ttm" / "RHOB"
    assert red_hf_home() == root / "hf"
    assert ttm_probabilistic_dir("RHOB") == root / "ttm" / "RHOB-probabilistic"


def test_has_finetuned_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("WMB_RED_CACHE", str(tmp_path))
    assert has_finetuned_ttm("RHOB") is False
    assert has_probabilistic_ttm("RHOB") is False
    folder = ttm_finetuned_dir("RHOB")
    folder.mkdir(parents=True)
    assert has_finetuned_ttm("RHOB") is False
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert has_finetuned_ttm("RHOB") is True
    assert has_probabilistic_ttm("RHOB") is False
    pfolder = ttm_probabilistic_dir("RHOB")
    pfolder.mkdir(parents=True)
    (pfolder / "config.json").write_text("{}", encoding="utf-8")
    assert has_probabilistic_ttm("RHOB") is True


def test_assert_gitignored_allows_override_outside_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("WMB_RED_CACHE", str(tmp_path))
    # outside the repo: no check-ignore required
    assert assert_gitignored(ttm_finetuned_dir("RHOB")) == ttm_finetuned_dir("RHOB")
    assert assert_gitignored(ttm_probabilistic_dir("RHOB")) == ttm_probabilistic_dir("RHOB")


def test_legacy_white_cache_env_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("WMB_WHITE_CACHE", str(tmp_path))
    assert red_cache_dir() == tmp_path.resolve()
    other = tmp_path / "red"
    monkeypatch.setenv("WMB_RED_CACHE", str(other))
    assert red_cache_dir() == other.resolve()


def test_old_weights_folder_used_only_while_new_default_is_empty(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "ttm").mkdir(parents=True)
    new.mkdir()
    monkeypatch.setattr(cache, "LEGACY_CACHE", old)
    monkeypatch.setattr(cache, "DEFAULT_CACHE", new)
    assert red_cache_dir() == old
    (new / "hf").mkdir()
    assert red_cache_dir() == new
