"""Score depth states with Decima (INT8 ONNX, CPU, one thread) in its own environment; no repository imports.

    <decima-env python> -I workstreams/black_physics/ordinal_density/decima_runner.py JOB.json OUT.json

JOB.json: {"repo", "revision", "precision", "cache", "question", "choices", "kind", "items": [{"key", "state"}]}.
OUT.json: {"meta": {...}, "items": [{"key", "logp": [one per choice, in the given order], "seconds"}]}.
The option order is preserved exactly; logp are Decima's normalised log-probabilities over the choices.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    import numpy as np
    from decima import Decima, Question
    from huggingface_hub import snapshot_download

    prec = job["precision"]
    root = Path(snapshot_download(job["repo"], revision=job["revision"], cache_dir=job.get("cache"),
                                  local_files_only=True,
                                  allow_patterns=["config.json", "LICENSE", "README.md", f"onnx/{prec}/*",
                                                  f"onnx/{prec}/tokenizer/*"]))
    t0 = time.perf_counter()
    model = Decima(root / "onnx" / prec, threads=1)
    load_s = time.perf_counter() - t0
    q = Question(job["question"], list(job["choices"]), kind=job["kind"], lang="en")
    model.decide_logits("Density estimate 2.410 g/cm3.", q)          # warm-up, not timed
    out = []
    for it in job["items"]:
        t = time.perf_counter()
        lp = np.asarray(model.decide_logits(it["state"], q), float)
        dt = time.perf_counter() - t
        if lp.shape != (len(job["choices"]),) or not np.isfinite(lp).all():
            raise RuntimeError(f"bad Decima output for {it['key']}: shape {lp.shape}")
        out.append({"key": it["key"], "logp": lp.tolist(), "seconds": dt})
    vers = {}
    for p in ("decima-ai", "onnxruntime", "tokenizers", "transformers", "numpy"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    meta = {"repo": job["repo"], "revision": job["revision"], "precision": prec, "threads": 1, "load_s": load_s,
            "snapshot": str(root), "versions": vers, "python": sys.version.split()[0]}
    Path(argv[1]).write_text(json.dumps({"meta": meta, "items": out}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
