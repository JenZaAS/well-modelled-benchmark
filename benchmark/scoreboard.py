"""Collect every results/<contender>/<dataset>/<target>/<scenario>/scores.json into one Markdown table.

    python -m benchmark.scoreboard            # print
    python -m benchmark.scoreboard --write    # also write results/summary/scoreboard.md (small, may be committed)
"""
from __future__ import annotations

import argparse
import json

from benchmark.registry import REPO  # noqa: F401  (sets sys.path)
from benchmark.test_def import generated_name
from wmb import paths
from wmb.aliases import canonical_row
from wmb.viewer import benchmark_run_order

COLS = ["test", "dataset", "target", "scenario", "contender", "workstream", "MAE", "RMSE", "R2", "cover_90", "cover_50",
        "interval_width_90", "has_uncertainty", "cover_1sigma", "cover_2sigma", "sharpness_mean_sigma", "CRPS",
        "load_s", "fit_s", "predict_s", "predict_s_per_well", "samples_per_s", "predict_s_per_1000"]


def collect() -> list[dict]:
    """All layouts: <model>/<dataset>/<target>/<scenario>/<test>/scores.json, .../<scenario>/scores.json, and the oldest without <scenario>."""
    root = paths.results_root()
    rows: dict[tuple, dict] = {}
    for pattern in ("*/*/*/scores.json", "*/*/*/*/scores.json", "*/*/*/*/*/scores.json"):
        for p in sorted(root.glob(pattern)):
            r = canonical_row(json.loads(p.read_text()))
            r.setdefault("scenario", "missing_curve")
            r["test"] = r.get("test") or ""
            key = (r["contender"], r["test"], r["dataset"], r["target"], r["scenario"])
            previous = rows.get(key)
            if (previous is None or r.get("legacy_contender") == previous.get("legacy_contender")
                    or benchmark_run_order(r) >= benchmark_run_order(previous)):
                rows[key] = r
    # Same rule as the viewer: a row without a test name is superseded by a named test of the same combination.
    named = {(k[0], *k[2:]) for k in rows if k[1]}
    rows = {k: r for k, r in rows.items() if k[1] or (k[0], *k[2:]) not in named}
    for r in rows.values():
        r["test"] = r["test"] or generated_name(r["dataset"], r["target"], r["scenario"])
    return sorted(rows.values(), key=lambda r: (r["test"], r["dataset"], r["target"], r["scenario"], r.get("MAE", 1e9)))


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
