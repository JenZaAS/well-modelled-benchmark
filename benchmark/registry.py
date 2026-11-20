"""Find contenders: every workstreams/<name>/contender.py that defines CONTENDERS
(a list of classes or factories). Adding a model = adding it to your own file;
the benchmark code doesn't need to change. A contender.py may also define OPT_IN
({factory: "WMB_FLAG=value"}) for models it registers only when a flag is set."""
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


class Registry(dict):
    """name -> factory. Legacy ids (``wmb.aliases``) resolve on lookup but are never listed or iterated."""

    def __missing__(self, key):
        from wmb.aliases import canonical_model
        new = canonical_model(key)
        if new != key and new in self:
            return self[new]
        raise KeyError(key)

    def __contains__(self, key):
        from wmb.aliases import canonical_model
        return dict.__contains__(self, key) or dict.__contains__(self, canonical_model(key))

    def get(self, key, default=None):
        return self[key] if key in self else default


def _contender_modules() -> list[tuple[str, str]]:
    """(workstream, module name) for every workstreams/<name>/contender.py."""
    return [(f.parent.name, f"workstreams.{f.parent.name}.contender") for f in sorted((REPO / "workstreams").glob("*/contender.py"))]


def discover_detailed(verbose: bool = False, *, call=lambda fn: fn()) -> tuple[dict, list[dict]]:
    """(found, failures): failures are [{"name", "workstream", "error"}] for workstreams or contenders that could not load."""
    found, failures = Registry(), []
    for ws, mod_name in _contender_modules():
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


def discover_opt_in(*, call=lambda fn: fn()) -> list[dict]:
    """Models a workstream leaves unregistered until a flag is set: [{"name", "workstream", "enable", "factory"}].

    A contender.py declares them as ``OPT_IN = {factory: "WMB_FLAG=value", ...}``. They are never run from here; the
    viewer's Prediction tab lists them, disabled, with the flag that registers them. Workstreams that fail to import
    are skipped (discover_detailed already reports them)."""
    out: list[dict] = []
    for ws, mod_name in _contender_modules():
        try:
            mod = call(lambda mod_name=mod_name: importlib.import_module(mod_name))
        except (Exception, SystemExit):
            continue
        for factory, enable in (getattr(mod, "OPT_IN", None) or {}).items():
            name = getattr(factory, "name", None) or getattr(factory, "__name__", ws)
            out.append({"name": str(name), "workstream": ws, "enable": str(enable), "factory": factory})
    return out


def discover(verbose: bool = False) -> dict:
    return discover_detailed(verbose)[0]
