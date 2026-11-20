"""Reusable inference: a masked task plus any base model's fill on it -> 15 bin probabilities per hidden depth.

    from workstreams.black_physics.ordinal_density.infer import Head
    head = Head.load(run_dir, "red_timestral")           # the head chosen on D, with its temperature
    head = Head.load_frozen("red_timestral", sha256)      # or: the frozen copy (cli `freeze`), checked by hash
    out = head.predict(task, p50, lo, hi)                 # p50/lo/hi: full-depth arrays or hidden-depth arrays

The head was fitted to one base model's errors on C; using it with another base model's fills is allowed but
uncalibrated. Only the masked task and the fill are read: the hidden values are never needed. NumPy only (no SciPy),
so a frozen head runs inside a model's own environment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from workstreams.black_physics.ordinal_density import bins as B
from workstreams.black_physics.ordinal_density import features as FT
from workstreams.black_physics.ordinal_density import heads as H


CODE = ("bins.py", "features.py", "heads.py", "infer.py")     # the inference code a frozen head depends on


class StaleHead(RuntimeError):
    """The frozen head is missing, was changed, or was made with different inference code."""


def frozen_dir() -> Path:
    from wmb.paths import work_root

    return work_root() / "black_ordinal_density" / "frozen"


def code_sha256() -> str:
    """Hash of the inference code (file names and bytes, line endings normalised)."""
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in CODE:
        h.update(name.encode())
        h.update(hashlib.sha256((here / name).read_bytes().replace(b"\r\n", b"\n")).digest())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _full(task, x):
    if x is None:
        return None
    x = np.asarray(x, float)
    n, m = len(task.depth), np.asarray(task.predict_mask, bool)
    if len(x) == n:
        return x
    if len(x) == int(m.sum()):
        out = np.full(n, np.nan)
        out[m] = x
        return out
    raise ValueError(f"expected {n} (all depths) or {int(m.sum())} (hidden depths) values, got {len(x)}")


class Head:
    def __init__(self, spec: dict, tails: B.Tails, name: str, calibrated: bool, model: str):
        self.name, self.calibrated, self.model, self.tails = name, calibrated, model, tails
        self.temperature = float(spec["temperature"]) if calibrated else 1.0
        if spec["kind"] == "empirical":
            self.head, self.policy = H.EmpiricalHead.from_dict(spec), None
        else:
            pol = spec["policy"]
            self.head = H.SoftmaxHead.from_dict(spec)
            self.policy = FT.FeaturePolicy(pol["use_width"], pol["sonic"], pol["products"])

    @classmethod
    def load(cls, run_dir, model: str, head: str | None = None) -> "Head":
        """head: '<name>|raw' or '<name>|cal'; default the head selected on D in scores.json."""
        run_dir = Path(run_dir)
        bundle = json.loads((run_dir / f"heads_{model}.json").read_text(encoding="utf-8"))
        if head is None:
            head = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))["models"][model]["selected"]
        name, var = head.split("|")
        return cls(bundle["heads"][name], B.Tails.from_dict(bundle["tails"]), name, var == "cal", model)

    @classmethod
    def load_frozen(cls, model: str, sha256: str, folder: Path | None = None) -> tuple["Head", dict]:
        """The frozen head for `model` (cli `freeze`), refused unless its hash and the inference code still match."""
        path = (Path(folder) if folder is not None else frozen_dir()) / f"{model}.json"
        how = ("run `python -m workstreams.black_physics.ordinal_density all` then "
               "`python -m workstreams.black_physics.ordinal_density freeze` (granite env)")
        if not path.is_file():
            raise StaleHead(f"no frozen head for {model} at {path}; {how}")
        if file_sha256(path) != sha256:
            raise StaleHead(f"{path} is not the frozen head this row was registered with (sha256 differs); {how}")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("code_sha256") != code_sha256():
            raise StaleHead(f"the ordinal_density inference code changed after {model}'s head was frozen; {how}")
        name, var = rec["head"].split("|")
        return cls(rec["spec"], B.Tails.from_dict(rec["tails"]), name, var == "cal", model), rec

    @property
    def needs_sonic(self) -> bool:
        return self.policy is not None and self.policy.sonic

    def predict(self, task, p50, lo=None, hi=None, gardner=None, edges=None) -> dict:
        """edges: None for the 15 fixed bins, or a nested finer grid (bins.fine_edges) for an uncalibrated empirical
        head, whose finer probabilities sum back to its 15-bin ones. Readouts always use the 15 bins."""
        fine = edges is not None and not (len(edges) == len(B.EDGES) and np.allclose(edges, B.EDGES))
        if fine and (self.policy is not None or self.temperature != 1.0):
            raise ValueError("only an uncalibrated empirical head has finer bins (a softmax head or a temperature "
                             "is fitted on the 15 fixed bins)")
        e = np.asarray(edges, float) if fine else B.EDGES
        if self.needs_sonic and gardner is None:
            from workstreams.black_physics.granite.pipeline import gardner_on

            gardner = gardner_on(task)
        rows = FT.gap_rows(task, _full(task, p50), _full(task, lo), _full(task, hi), _full(task, gardner))
        if self.policy is None:
            p = self.head.predict(rows["base"], rows["flank_var"], e)
        else:
            p = self.head.predict(self.policy.matrix(rows)[0])
        p = H.apply_temperature(p, self.temperature) if self.temperature != 1.0 else p
        coarse = B.coarsen(p, e) if fine else p
        return {"index": rows["index"], "depth": rows["depth"], "probs": p, "edges": e.copy(),
                "readouts": B.readouts(coarse, self.tails), "p_below_2.40": B.prob_below(coarse),
                "head": f"{self.name}|{'cal' if self.calibrated else 'raw'}", "fitted_on_base_model": self.model}
