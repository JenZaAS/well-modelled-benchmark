"""Score gap summaries with Decima-small (INT8 ONNX, CPU) in its own environment.

Runs without this repository's packages or torch. Input: JSON lines {"key", "state"}; output: JSON lines
{"key", "p_pass", "logit", "p_yes", "p_no"}, plus a provenance header row {"_meta": ...}.

    <decima-env python> -I workstreams/black_physics/granite/decima_score.py IN.jsonl OUT.jsonl --cache <hf hub dir>

The question and option order are fixed: kind "verify", choices ["yes", "no"] in that order, and
p_pass = P("yes"). `logit` = log P(yes) - log P(no), the input to the later intercept/slope recalibration.
Install (Python 3.12): pip install decima-ai==2.5.1 --no-deps; pip install onnxruntime tokenizers
huggingface_hub numpy transformers==4.57.6   (the runtime imports no torch).
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import time
from pathlib import Path

REPO_ID = "amyrmahdy/decima-small"
REVISION = "480c4ea9e54f2946b947fb3ceb3c7cc7ece77522"
PRECISION = "int8"
QUESTION = "Will the mean absolute error across this 50 m density gap be at most 0.05 g/cm³?"
CHOICES = ["yes", "no"]          # p_pass = probability of CHOICES[0]


def load(cache: str | None, download: bool):
    from huggingface_hub import snapshot_download
    from decima import Decima

    root = Path(snapshot_download(REPO_ID, revision=REVISION, cache_dir=cache, local_files_only=not download,
                                  allow_patterns=["config.json", "LICENSE", "README.md", f"onnx/{PRECISION}/*",
                                                  f"onnx/{PRECISION}/tokenizer/*"]))
    return Decima(root / "onnx" / PRECISION, threads=1), root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--cache", help="Hugging Face hub cache directory")
    ap.add_argument("--download", action="store_true", help="allow fetching the pinned revision once")
    a = ap.parse_args(argv)
    import numpy as np
    from decima import Question

    model, root = load(a.cache, a.download)
    q = Question(QUESTION, list(CHOICES), kind="verify", lang="en")
    rows = [json.loads(line) for line in Path(a.inp).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    t0 = time.perf_counter()
    for r in rows:
        lp = np.asarray(model.decide_logits(r["state"], q), float)       # log-probabilities aligned with CHOICES
        p = np.exp(lp)
        out.append({"key": r["key"], "p_pass": float(p[0]), "p_yes": float(p[0]), "p_no": float(p[1]),
                    "logit": float(lp[0] - lp[1])})
    sec = time.perf_counter() - t0
    meta = {"_meta": {"model": f"{REPO_ID}@{REVISION}", "precision": PRECISION, "question": QUESTION,
                      "choices": CHOICES, "p_pass": "P(choices[0])", "decima_ai": md.version("decima-ai"),
                      "onnxruntime": md.version("onnxruntime"), "n": len(out), "seconds": round(sec, 3),
                      "ms_per_state": round(1000 * sec / max(len(out), 1), 2), "snapshot": str(root),
                      "python": sys.version.split()[0]}}
    Path(a.out).write_text("\n".join(json.dumps(x) for x in [meta, *out]) + "\n", encoding="utf-8")
    print(json.dumps(meta["_meta"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
