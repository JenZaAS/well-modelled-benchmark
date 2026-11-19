"""Find contenders: every workstreams/<name>/contender.py that defines CONTENDERS
(a list of classes or factories). Adding a model = adding it to your own file;
the benchmark code doesn't need to change."""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def discover(verbose: bool = False) -> dict:
    found = {}
    for f in sorted((REPO / "workstreams").glob("*/contender.py")):
        mod_name = f"workstreams.{f.parent.name}.contender"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # one broken workstream must not stop the others
            print(f"[registry] skipped {mod_name}:\n{traceback.format_exc(limit=2)}", file=sys.stderr)
            continue
        for factory in getattr(mod, "CONTENDERS", []):
            try:
                c = factory()
            except Exception:  # a model that cannot even be built is skipped, not fatal
                print(f"[registry] skipped a contender in {mod_name}:\n{traceback.format_exc(limit=2)}", file=sys.stderr)
                continue
            found[c.name] = factory
            if verbose:
                print(f"  {c.name:28s} workstream={c.workstream}")
    return found
