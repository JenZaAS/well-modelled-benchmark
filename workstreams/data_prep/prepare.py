"""Data prep starter: load a dataset through wmb and print per-well completeness.

    python -m workstreams.data_prep.prepare --dataset synthetic
    python -m workstreams.data_prep.prepare --dataset nlog         # folder under FH2026_DATA with wells.json

Output contract (for the other teams): every dataset folder under FH2026_DATA has a
valid wells.json (viewer/manifest.schema.json) whose curve_map maps to the 9 core names.
Cleaned / resampled copies go to FH2026_WORK/prepared/<dataset>/, never into the repo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wmb import paths, synthetic  # noqa: E402
from wmb.curves import CORE  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="synthetic")
    a = ap.parse_args(argv)
    if a.dataset == "synthetic":
        wells = synthetic.make_wells()
    else:
        from wmb.io import iter_manifest_wells
        wells = list(iter_manifest_wells(paths.data_root() / a.dataset))
    print("well".ljust(14) + " ".join(c.rjust(5) for c in CORE))
    for w in wells:
        c = w.completeness()
        print(w.name.ljust(14) + " ".join(f"{100 * c[k]:5.0f}" for k in CORE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
