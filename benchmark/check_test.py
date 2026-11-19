"""Validate benchmark test definitions (schema, plus data checks when the dataset can be loaded).

    python -m benchmark.check_test rhob_missing
    python -m benchmark.check_test all
    python -m benchmark.check_test path\\to\\my_test.yaml

Exit code 0 when every checked def is valid (warnings are allowed), 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark import test_def as td


def check_file(path: Path) -> tuple[list[str], list[str]]:
    """(errors, warnings) for one def file: schema first, then data checks."""
    try:
        tdef = td.load_def(path)
    except td.TestDefError as exc:
        return [str(exc)], []
    return td.check_def(tdef)


def resolve_paths(name: str, defs_dir: Path | None = None) -> list[Path]:
    if name == "all":
        return td.def_paths(defs_dir)
    p = Path(name)
    if p.suffix == ".yaml" and p.is_file():
        return [p]
    return [(Path(defs_dir) if defs_dir else td.DEFS_DIR) / f"{name}.yaml"]


def check_names(names: list[str], defs_dir: Path | None = None) -> int:
    paths_ = [p for n in names for p in resolve_paths(n, defs_dir)]
    if not paths_:
        print("no test definitions found", file=sys.stderr)
        return 1
    failed = 0
    for p in paths_:
        if not p.is_file():
            print(f"FAILED {p.stem}: file not found: {p}", file=sys.stderr)
            failed += 1
            continue
        errors, warnings = check_file(p)
        for w in warnings:
            print(f"warning: {p.stem}: {w}")
        if errors:
            failed += 1
            print(f"FAILED {p.stem}:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
        else:
            print(f"OK {p.stem}")
    return 1 if failed else 0


def main(argv=None, defs_dir: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate benchmark test definitions")
    ap.add_argument("names", nargs="*", default=["all"], help="test names, YAML paths, or 'all' (default)")
    a = ap.parse_args(argv)
    return check_names(a.names, defs_dir)


if __name__ == "__main__":
    sys.exit(main())
