"""``python -m wmb.datasets --check <dataset>``."""
from __future__ import annotations

import argparse
import sys

CHECKERS = {"force2020": "wmb.datasets.force2020:check_force2020"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wmb.datasets", description="Dataset pack tools.")
    ap.add_argument("--check", metavar="DATASET", required=True,
                    help="verify files and checksums; one of: " + ", ".join(CHECKERS))
    args = ap.parse_args(argv)
    from wmb.paths import normalise_dataset

    key = normalise_dataset(args.check)
    if key not in CHECKERS:
        print(f"No checker for {args.check!r}. Available: {', '.join(CHECKERS)}", file=sys.stderr)
        return 2
    import importlib

    mod, func = CHECKERS[key].split(":")
    return getattr(importlib.import_module(mod), func)()


if __name__ == "__main__":
    sys.exit(main())
