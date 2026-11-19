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


def discover_detailed(verbose: bool = False, *, call=lambda fn: fn()) -> tuple[dict, list[dict]]:
    """(found, failures): failures are [{"name", "workstream", "error"}] for workstreams or contenders that could not load."""
    found, failures = {}, []
    for f in sorted((REPO / "workstreams").glob("*/contender.py")):
        ws = f.parent.name
        mod_name = f"workstreams.{ws}.contender"
        try:
            mod = call(lambda mod_name=mod_name: importlib.import_module(mod_name))
        except (Exception, SystemExit) as exc:  # one broken workstream must not stop the others
            print(f"[registry] skipped {mod_name}:\n{traceback.format_exc(limit=2)}", file=sys.stderr)
            failures.append({"name": ws, "workstream": ws, "error": f"import failed: {type(exc).__name__}: {exc}"})
            continue
        for factory in getattr(mod, "CONTENDERS", []):
            try:
                c = call(factory)
            except (Exception, SystemExit) as exc:  # a model that cannot even be built is skipped, not fatal
                print(f"[registry] skipped a contender in {mod_name}:\n{traceback.format_exc(limit=2)}", file=sys.stderr)
                name = getattr(factory, "name", None) or getattr(factory, "__name__", ws)
                failures.append({"name": str(name), "workstream": ws, "error": f"could not be built: {type(exc).__name__}: {exc}"})
                continue
            found[c.name] = factory
            if verbose:
                print(f"  {c.name:28s} workstream={c.workstream}")
    return found, failures


def discover(verbose: bool = False) -> dict:
    return discover_detailed(verbose)[0]
