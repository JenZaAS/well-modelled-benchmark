"""Granite checks through the real pinned networks. Skip unless torch, granite-tsfm and the cached weights exist.

Run in the Granite environment (see granite/README.md):
    <granite-python> -m pytest workstreams/black_physics/tests/test_granite_torch.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch = pytest.importorskip("torch")
pytest.importorskip("tsfm_public")

from workstreams.black_physics.granite import backends as B  # noqa: E402
from workstreams.black_physics.granite import config as C  # noqa: E402
from workstreams.black_physics.granite import gaps as G  # noqa: E402
from workstreams.black_physics.granite.reconstruct import make_reconstructor  # noqa: E402


def _cached(repo, rev):
    try:
        B.snapshot_path(repo, rev)
        return True
    except Exception:
        return False


needs_ttm = pytest.mark.skipif(not _cached(C.TTM_REPO, C.TTM_REVISION), reason="TTM-R3 weights not cached")
needs_tsp = pytest.mark.skipif(not _cached(C.TSPULSE_REPO, C.TSPULSE_REVISION), reason="TSPulse weights not cached")


def _well(n=3000, seed=0):
    from wmb.io import Well

    rng = np.random.default_rng(seed)
    depth = 1000 + 0.152 * np.arange(n)
    y = 2.35 + 0.08 * np.sin(depth / 9) + rng.normal(0, 0.01, n)
    return Well("T-1", depth, {"RHOB": y}, "synthetic", "train")


def _task(w, a, b):
    return G.gap_task(w, G._make_gap(w.name, "B", 0, w.depth, a, b))


@pytest.fixture(scope="module")
def ttm():
    return B.make_backend("ttm", "cpu")


@pytest.fixture(scope="module")
def tsp():
    return B.make_backend("tspulse", "cpu")


@needs_ttm
def test_ttm_hidden_values_do_not_change_output_and_units_are_raw(ttm):
    w = _well()
    a, b = 1200, 1530
    p1 = make_reconstructor(ttm).run([_task(w, a, b)])[0][0]
    w.curves["RHOB"][a:b] += 0.4
    p2 = make_reconstructor(ttm).run([_task(w, a, b)])[0][0]
    np.testing.assert_array_equal(p1.p50, p2.p50)
    assert p1.status == "two_sided" and p1.q_raw.shape == (9, b - a)
    assert np.all(np.diff(p1.q_raw, axis=0) >= -1e-6)                       # quantiles ordered
    assert 2.0 < np.median(p1.p50) < 2.7                                    # g/cm3, not scaled units


@needs_tsp
def test_tspulse_user_mask_hides_the_gap_and_units_are_raw(tsp):
    w = _well()
    a, b = 1200, 1530
    p1 = make_reconstructor(tsp).run([_task(w, a, b)])[0][0]
    w.curves["RHOB"][a:b] = 9.9                                             # absurd hidden truth
    p2 = make_reconstructor(tsp).run([_task(w, a, b)])[0][0]
    np.testing.assert_array_equal(p1.p50, p2.p50)
    assert 2.0 < np.median(p1.p50) < 2.7
    x, m = tsp.batch(np.array([np.where((np.arange(512) >= 100) & (np.arange(512) < 300), np.nan, 2.4)]))
    assert not bool(m[0, 100:300, 0].any()) and bool(m[0, :100, 0].all())


@pytest.mark.parametrize("arm", ["ttm", "tspulse"])
def test_tuned_checkpoint_changes_only_the_head(arm):
    repo, rev = (C.TTM_REPO, C.TTM_REVISION) if arm == "ttm" else (C.TSPULSE_REPO, C.TSPULSE_REVISION)
    if not _cached(repo, rev):
        pytest.skip("weights not cached")
    f = C.work_dir() / "checkpoints" / f"{arm}_tuned_best.pt"
    if not f.is_file():
        pytest.skip("no tuned checkpoint (run the train stage)")
    head = torch.load(f, map_location="cpu", weights_only=True)
    frozen = B.make_backend(arm, "cpu")
    tuned = B.make_backend(arm, "cpu", head)
    fs, ts = frozen.model.state_dict(), tuned.model.state_dict()
    changed = [k for k in fs if not torch.equal(fs[k], ts[k])]
    assert changed and set(changed) <= set(head)
    assert all(any(h in f".{k}" for h in frozen.head_keys) for k in changed)


@needs_ttm
def test_finetune_blended_loss_scores_exactly_what_inference_produces(ttm):
    from workstreams.black_physics.granite import finetune as FTN
    from workstreams.black_physics.granite.backends import TTM_QUANTILES

    w = _well()
    a, b = 1200, 1530
    pc = make_reconstructor(ttm).run([_task(w, a, b)])[0][0]
    e = FTN.ttm_pair_examples({w.name: w}, [G._make_gap(w.name, "A", 0, w.depth, a, b)])[0]
    with torch.no_grad():
        loss = float(FTN.ttm_blended_loss(ttm, [e]))
    d = e.y[None, :] - pc.q_raw
    tau = np.array(TTM_QUANTILES)[:, None]
    assert loss == pytest.approx(float(np.maximum(tau * d, (tau - 1) * d).mean()), rel=1e-5)
