"""Lightweight import checks so the Prediction tab can grey out models that lack a Python package."""
from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

# Import name -> pip install name, for the inline message in the viewer.
PIP_NAME = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
}


def pip_name(module: str) -> str:
    root = (module or "").split(".")[0]
    return PIP_NAME.get(root, root)


def needs_install_message(module: str) -> str:
    pkg = pip_name(module)
    return f"needs {pkg} — pip install {pkg}"


def friendly_module_error(exc: BaseException) -> str | None:
    """Rewrite ModuleNotFoundError / ImportError into a short install hint, else None."""
    name = getattr(exc, "name", None)
    if not name and isinstance(exc, (ModuleNotFoundError, ImportError)):
        msg = str(exc)
        if "No module named" in msg:
            name = msg.rsplit(" ", 1)[-1].strip(chr(39) + chr(34))
        elif msg:
            name = msg.strip(chr(39) + chr(34))
    if not name:
        return None
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return needs_install_message(name)
    return None


def _is_import_error_handler(handler: ast.ExceptHandler) -> bool:
    typ = handler.type
    names: list[str] = []
    if typ is None:
        return False
    if isinstance(typ, ast.Name):
        names = [typ.id]
    elif isinstance(typ, ast.Tuple):
        names = [e.id for e in typ.elts if isinstance(e, ast.Name)]
    return any(n in ("ImportError", "ModuleNotFoundError") for n in names)


def _alias_roots(aliases: list[ast.alias], prefix: str | None = None) -> set[str]:
    out: set[str] = set()
    for a in aliases:
        if prefix is None:
            if a.name != "*":
                out.add(a.name.split(".")[0])
        elif prefix:
            out.add(prefix.split(".")[0])
        else:
            if a.name != "*":
                out.add(a.name.split(".")[0])
    return out


def _imports_in(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            found |= _alias_roots(n.names)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            found.add(n.module.split(".")[0])
    return found


def import_groups(source: str) -> tuple[set[str], list[set[str]]]:
    """(required modules, alternative groups). A group is ok if any member is installed
    (e.g. 	ry: import xgboost / except ImportError: import sklearn)."""
    tree = ast.parse(source)
    required: set[str] = set()
    groups: list[set[str]] = []
    skip_import_nodes: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(_is_import_error_handler(h) for h in node.handlers):
            continue
        names: set[str] = set()
        for stmt in node.body:
            names |= _imports_in(stmt)
            skip_import_nodes.add(id(stmt))
            for child in ast.walk(stmt):
                skip_import_nodes.add(id(child))
        for h in node.handlers:
            if _is_import_error_handler(h):
                for stmt in h.body:
                    names |= _imports_in(stmt)
                    skip_import_nodes.add(id(stmt))
                    for child in ast.walk(stmt):
                        skip_import_nodes.add(id(child))
        if names:
            groups.append(names)

    for node in ast.walk(tree):
        if id(node) in skip_import_nodes:
            continue
        if isinstance(node, ast.Import):
            required |= _alias_roots(node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            required.add(node.module.split(".")[0])
    required -= {n for g in groups for n in g}
    return required, groups


def module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


def missing_from_source(source: str) -> list[str]:
    """Root module names that are required (or whose whole fallback group is missing)."""
    required, groups = import_groups(source)
    missing: list[str] = []
    for name in sorted(required):
        if not module_installed(name):
            missing.append(name)
    for g in groups:
        if not any(module_installed(n) for n in g):
            # Prefer a pip-friendly fallback name (sklearn over the first try).
            missing.append("sklearn" if "sklearn" in g else sorted(g)[0])
    return missing


def missing_for_factory(factory) -> list[str]:
    """Modules named in the factory's source file that are not installed."""
    try:
        path = inspect.getsourcefile(factory)
        if not path and inspect.isclass(factory):
            path = inspect.getsourcefile(factory)
        if not path:
            mod = inspect.getmodule(factory)
            path = getattr(mod, "__file__", None) if mod else None
        if not path:
            return []
        source = Path(path).read_text(encoding="utf-8")
    except (OSError, TypeError, UnicodeError):
        return []
    return missing_from_source(source)


def missing_reason(factory) -> str | None:
    missing = missing_for_factory(factory)
    if not missing:
        return None
    return needs_install_message(missing[0])
