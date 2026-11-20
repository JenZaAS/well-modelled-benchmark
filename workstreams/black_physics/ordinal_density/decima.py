"""The matched Decima-small comparison: state text, the deterministic depth sample, and a keyed cache.

Decima reads exactly the base-state fields of the numeric head, in fixed units and three decimals; the sonic
fields are not given (the numeric-versus-Decima comparison stays matched on the base state). It answers one
ordinal question with the 15 density ranges as options, low to high.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from workstreams.black_physics.ordinal_density import bins as B

REPO = "amyrmahdy/decima-small"
REVISION = "480c4ea9e54f2946b947fb3ceb3c7cc7ece77522"
PRECISION = "int8"
KIND = "score"
QUESTION = "At the marked depth, which range contains the measured bulk density?"
CHOICES = B.option_labels("g/cm3")
PER_GAP = 8          # one deterministic draw from each of eight equal-depth sections of a gap
SEED = 2026


def python() -> Path:
    return Path(os.environ.get("WMB_DECIMA_PYTHON") or
                Path(os.environ.get("LOCALAPPDATA", "")) / "wmb-envs" / "decima-cpu" / "Scripts" / "python.exe")


def _v(x, nd=3) -> str:
    return f"{x:.{nd}f}" if np.isfinite(x) else "missing"


def state_text(row: dict, use_width: bool) -> str:
    """One depth's state. row: the state fields at that depth (see features.FIELDS)."""
    s = (f"Density estimate {_v(row['base'])} g/cm3. Relative position in gap {_v(row['position'])}. "
         f"Nearest shallow and deep density readings {_v(row['flank_up'])} and {_v(row['flank_dn'])} g/cm3, "
         f"at distances {_v(row['dist_up'])} and {_v(row['dist_dn'])} m. "
         f"Observed flank variability {_v(row['flank_var'])} g/cm3.")
    if use_width:
        s += f" Model P10-P90 width {_v(row['width'])} g/cm3."
    return s


def sample_positions(n: int, model: str, group: str, gap_id: str, per_gap: int = PER_GAP, seed: int = SEED):
    """Indices (within the gap's hidden depths) of the sampled depths: one per equal section, reproducible."""
    h = hashlib.sha256(f"{seed}|{model}|{group}|{gap_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return np.array([int(rng.choice(sec)) for sec in np.array_split(np.arange(n), min(per_gap, n))])


def cache_key(state: str) -> str:
    blob = json.dumps({"state": state, "repo": REPO, "revision": REVISION, "precision": PRECISION, "kind": KIND,
                       "question": QUESTION, "choices": CHOICES}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def run(states: list[str], cache_file: Path, hub: Path, log=print) -> tuple[np.ndarray, np.ndarray, dict]:
    """Probabilities [n, 15] and per-state seconds for `states`, reading/writing the keyed cache."""
    cache = {}
    if cache_file.is_file():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cache[r["key"]] = r
    keys = [cache_key(s) for s in states]
    todo = {k: s for k, s in zip(keys, states) if k not in cache}
    meta = {"cached": len(states) - len(todo), "new": len(todo)}
    if todo:
        py = python()
        if not py.is_file():
            raise FileNotFoundError(f"Decima environment not found at {py} (set WMB_DECIMA_PYTHON)")
        with tempfile.TemporaryDirectory() as td:
            job, out = Path(td) / "job.json", Path(td) / "out.json"
            job.write_text(json.dumps({"repo": REPO, "revision": REVISION, "precision": PRECISION, "cache": str(hub),
                                       "question": QUESTION, "choices": CHOICES, "kind": KIND,
                                       "items": [{"key": k, "state": s} for k, s in todo.items()]}), encoding="utf-8")
            script = Path(__file__).resolve().parent / "decima_runner.py"
            log(f"decima: {len(todo)} new states ({meta['cached']} cached) in {py}")
            t0 = time.perf_counter()
            res = subprocess.run([str(py), "-I", str(script), str(job), str(out)], capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Decima runner failed: {res.stderr.strip()[-800:]}")
            got = json.loads(out.read_text(encoding="utf-8"))
            meta.update(got["meta"], wall_s=time.perf_counter() - t0)
            with cache_file.open("a", encoding="utf-8") as f:
                for it in got["items"]:
                    rec = {"key": it["key"], "state": todo[it["key"]], "logp": it["logp"], "seconds": it["seconds"],
                           "meta": {k: got["meta"][k] for k in ("repo", "revision", "precision", "versions")}}
                    cache[it["key"]] = rec
                    f.write(json.dumps(rec) + "\n")
    lp = np.array([cache[k]["logp"] for k in keys], float)
    p = np.exp(lp)
    p /= p.sum(axis=1, keepdims=True)
    secs = np.array([cache[k]["seconds"] for k in keys], float)
    used = {json.dumps(cache[k]["meta"], sort_keys=True) for k in keys}
    meta["checkpoints_used"] = [json.loads(u) for u in sorted(used)]      # one entry unless the cache mixes runs
    meta["_scores"] = lp                         # Decima's own scores per option, in option order
    return p, secs, meta


if __name__ == "__main__":                       # pragma: no cover - convenience
    print(json.dumps({"question": QUESTION, "choices": CHOICES}, indent=1))
    sys.exit(0)
