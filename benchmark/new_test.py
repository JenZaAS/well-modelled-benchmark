"""Create a new benchmark test definition from the template, then validate it.

    python -m benchmark.new_test my_test            # writes benchmark/test_defs/my_test.yaml
    python -m benchmark.new_test my_test --force    # overwrite an existing file
    python -m benchmark.new_test --check [name|all] # same as python -m benchmark.check_test

Exit codes: 0 schema-valid (data checks may warn), 1 schema-invalid, 2 usage error.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from benchmark import test_def as td
from benchmark.check_test import check_names


def create(name: str, defs_dir: Path | None = None, force: bool = False) -> Path:
    if not td.NAME_RE.fullmatch(name):
        raise ValueError(f"invalid name '{name}': use letters, digits, '_' and '-'")
    if name == "all" or name.upper() == "TEMPLATE":
        raise ValueError(f"'{name}' is reserved; choose another test name")
    out = (Path(defs_dir) if defs_dir else td.DEFS_DIR) / f"{name}.yaml"
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass --force to overwrite")
    text = (td.DEFS_DIR / td.TEMPLATE_NAME).read_text(encoding="utf-8")
    text = re.sub(r"(?m)^name: \S+", f"name: '{name}'", text, count=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


def main(argv=None, defs_dir: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create a benchmark test definition from the template")
    ap.add_argument("name", nargs="?", help="test name (file stem)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    ap.add_argument("--check", nargs="?", const="all", metavar="NAME", help="only validate NAME (default all)")
    a = ap.parse_args(argv)
    if a.check is not None:
        return check_names([a.check], defs_dir)
    if not a.name:
        print("usage: python -m benchmark.new_test <name> [--force]\n"
              f"template: {td.DEFS_DIR / td.TEMPLATE_NAME}\nguide: docs/tests.md")
        return 2
    try:
        out = create(a.name, defs_dir, a.force)
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out}; edit it, then run: python -m benchmark.check_test {a.name}")
    try:
        tdef = td.load_def(out)
    except td.TestDefError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    errors, warnings = td.check_def(tdef)
    for w in warnings + errors:
        print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
