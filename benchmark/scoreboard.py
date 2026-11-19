"""Collect every results/<contender>/<dataset>/<target>/scores.json into one Markdown table.

    python -m benchmark.scoreboard            # print
    python -m benchmark.scoreboard --write    # also write results/summary/scoreboard.md (small, may be committed)
"""
from __future__ import annotations

import argparse
import json

from benchmark.registry import REPO  # noqa: F401  (sets sys.path)
from wmb import paths

COLS = ["dataset", "target", "contender", "workstream", "MAE", "RMSE", "R2", "cover_1sigma", "cover_2sigma",
        "sharpness_mean_sigma", "CRPS", "fit_s", "predict_s", "predict_s_per_1000"]


def collect() -> list[dict]:
    rows = [json.loads(p.read_text()) for p in paths.results_root().glob("*/*/*/scores.json")]
    return sorted(rows, key=lambda r: (r["dataset"], r["target"], r.get("MAE", 1e9)))


def to_markdown(rows: list[dict]) -> str:
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in COLS) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    md = to_markdown(collect()); print(md)
    if a.write:
        out = paths.results_root() / "summary"; out.mkdir(parents=True, exist_ok=True)
        (out / "scoreboard.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
