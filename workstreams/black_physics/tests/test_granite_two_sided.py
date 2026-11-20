"""Two-sided TTM bridge (granite/two_sided.py). Skips unless torch is installed; the end-to-end checks also need
granite-tsfm and the cached TTM-R3 weights. Run in the Granite environment:
    <granite-python> -m pytest workstreams/black_physics/tests/test_granite_two_sided.py
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

from workstreams.black_physics.granite import gaps as G  # noqa: E402
from workstreams.black_physics.granite import two_sided as TS  # noqa: E402


def _random_bridge(seed=1):
    torch.manual_seed(seed)
    br = TS.make_bridge()
    with torch.no_grad():
        for p in br.parameters():
            p.normal_(0, 0.3)
    return br.eval()


def _inputs(H=40, seed=0):
    rng = np.random.default_rng(seed)
    qa = np.sort(2.3 + 0.05 * rng.normal(size=(9, H)), axis=0)
    qb = np.sort(2.4 + 0.05 * rng.normal(size=(9, H)), axis=0)
    u = np.linspace(0.02, 0.98, H)
    return qa, qb, u


def test_bridge_starts_as_the_fixed_blend():
    qa, qb, u = _inputs()
    out = TS.apply_bridge(TS.make_bridge().eval(), qa, qb, u, 2.31, 2.42)
    np.testing.assert_allclose(out, (1 - u) * qa + u * qb, atol=1e-6)


def test_bridge_is_level_free_and_reads_the_far_flank():
    qa, qb, u = _inputs()
    br = _random_bridge()
    out = TS.apply_bridge(br, qa, qb, u, 2.31, 2.42)
    shifted = TS.apply_bridge(br, qa + 0.2, qb + 0.2, u, 2.51, 2.62)       # every density moved by +0.2
    np.testing.assert_allclose(shifted, out + 0.2, atol=1e-5)
    moved = TS.apply_bridge(br, qa, qb, u, 2.31, 2.52)                       # only the deep measured value moves
    assert not np.allclose(moved, out)


def _cached():
    try:
        pytest.importorskip("tsfm_public")
        from workstreams.black_physics.granite import backends as B
        from workstreams.black_physics.granite import config as C

        B.snapshot_path(C.TTM_REPO, C.TTM_REVISION)
        return True
    except BaseException:
        return False


needs_ttm = pytest.mark.skipif(not _cached(), reason="granite-tsfm or the TTM-R3 weights are not available")


def _well(n=3000, seed=0):
    from wmb.io import Well

    rng = np.random.default_rng(seed)
    depth = 1000 + 0.152 * np.arange(n)
    y = 2.35 + 0.08 * np.sin(depth / 9) + rng.normal(0, 0.01, n)
    return Well("T-1", depth, {"RHOB": y}, "synthetic", "train")


@needs_ttm
def test_training_loss_scores_exactly_what_inference_produces():
    from workstreams.black_physics.granite import finetune as FTN
    from workstreams.black_physics.granite.backends import TTM_QUANTILES, make_backend

    ttm = make_backend("ttm", "cpu")
    br = _random_bridge()
    w = _well()
    a, b = 1200, 1530
    task = G.gap_task(w, G._make_gap(w.name, "B", 0, w.depth, a, b))
    pc = TS.TwoSidedReconstructor(ttm, br).run([task])[0][0]
    assert pc.status == "two_sided"
    plain = TS.TwoSidedReconstructor(ttm, TS.make_bridge().eval()).run([task])[0][0]
    from workstreams.black_physics.granite.reconstruct import make_reconstructor

    np.testing.assert_allclose(plain.p50, make_reconstructor(ttm).run([task])[0][0].p50, atol=1e-6)
    e = FTN.ttm_pair_examples({w.name: w}, [G._make_gap(w.name, "A", 0, w.depth, a, b)])[0]
    with torch.no_grad():
        loss = float(TS.bridged_loss(ttm, br, [e], train_ttm=False))
    d = e.y[None, :] - pc.q_raw
    tau = np.array(TTM_QUANTILES)[:, None]
    assert loss == pytest.approx(float(np.maximum(tau * d, (tau - 1) * d).mean()), rel=1e-5)
