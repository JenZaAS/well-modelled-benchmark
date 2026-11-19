"""Where data, work files and results live.

Primary config: ``data_paths.local.yaml`` at the repo root (gitignored).
Override the file location with ``WMB_DATA_PATHS``. See
``docs/SETUP_DATA_PATHS.md``.

Environment variables still work as a fallback:

    FH2026_DATA     data root, if the yaml file is missing
    FH2026_WORK     scratch / intermediate files
    FH2026_CACHE    cache root (default ``~/fh2026-cache``)
    WMB_RESULTS     benchmark outputs (default ``<repo>/results``)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Canonical keys -> default subfolder names under data_root (first is preferred).
DATASETS: dict[str, tuple[str, ...]] = {
    "force2020": ("FORCE2020", "force2020"),
    "sodir_factpages": ("Sodir-FactPages", "sodir_factpages"),
    "nlog_dts": ("NLOG-DTS", "NLOG", "nlog"),
    "kansas_kgs_dts": ("Kansas-KGS-DTS", "kansas_kgs_dts"),
    "volve": ("Volve", "volve"),
}

_FOLDER_TO_KEY: dict[str, str] = {}
for _key, _folders in DATASETS.items():
    _FOLDER_TO_KEY[_key] = _key
    for _folder in _folders:
        _FOLDER_TO_KEY[_folder.lower()] = _key

_CONFIG: dict | None = None
_CONFIG_PATH: Path | None = None


class DataPathsError(FileNotFoundError):
    """Raised when the local data-paths file is missing and no fallback is set."""


def _reset_for_tests() -> None:
    global _CONFIG, _CONFIG_PATH
    _CONFIG = None
    _CONFIG_PATH = None


def repo_root() -> Path:
    return REPO


def config_path() -> Path:
    """Path of the yaml file that will be / was loaded."""
    override = os.environ.get("WMB_DATA_PATHS")
    if override:
        return Path(override).expanduser()
    return REPO / "data_paths.local.yaml"


def _missing_message() -> str:
    return (
        "No data_paths.local.yaml found and FH2026_DATA is not set.\n"
        "This file says where the well data lives on this machine and is not committed.\n"
        "Create it by following docs/SETUP_DATA_PATHS.md, then run: python -m wmb.paths --check"
    )


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required to read data_paths.local.yaml. Install it with: pip install pyyaml"
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise DataPathsError(f"{path}: expected a YAML mapping, not {type(data).__name__}")
    return data


def _as_path(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def load_config() -> dict:
    """Load and cache the local config (yaml and/or env fallbacks)."""
    global _CONFIG, _CONFIG_PATH
    path = config_path()
    if _CONFIG is not None and _CONFIG_PATH == path:
        return _CONFIG

    cfg: dict = {}
    if path.is_file():
        cfg = dict(_load_yaml(path))
    elif os.environ.get("FH2026_DATA"):
        cfg = {"data_root": os.environ["FH2026_DATA"]}
    else:
        raise DataPathsError(_missing_message())

    root = _as_path(cfg.get("data_root"))
    if root is None:
        raise DataPathsError(
            f"{path}: 'data_root' is missing or empty. See docs/SETUP_DATA_PATHS.md"
        )
    cfg["_path"] = path
    cfg["_data_root"] = root
    cache = _as_path(cfg.get("cache_dir"))
    if cache is None:
        cache = Path(os.environ.get("FH2026_CACHE", Path.home() / "fh2026-cache")).expanduser()
    cfg["_cache_dir"] = cache
    _CONFIG, _CONFIG_PATH = cfg, path
    return cfg


def data_root() -> Path:
    return load_config()["_data_root"]


def cache_dir() -> Path:
    return load_config()["_cache_dir"]


def work_root() -> Path:
    env = os.environ.get("FH2026_WORK")
    if env:
        return Path(env).expanduser()
    return cache_dir() / "work"


def results_root() -> Path:
    return Path(os.environ.get("WMB_RESULTS", REPO / "results")).expanduser()


def normalise_dataset(name: str) -> str:
    """Map a folder name or alias to a canonical dataset key; unknown names pass through."""
    key = _FOLDER_TO_KEY.get(name.strip().lower())
    return key or name.strip()


def get_dataset_path(name: str) -> Path:
    """Resolve a dataset folder.

    Looks at an optional per-dataset key in the yaml, then at the default
    subfolder names under ``data_root``. The returned path may not exist;
    use ``dataset_status`` or ``python -m wmb.paths --check`` to see what
    is present.
    """
    cfg = load_config()
    key = normalise_dataset(name)
    override = _as_path(cfg.get(key)) if key in DATASETS else None
    if override is not None:
        return override
    root = data_root()
    folders = DATASETS.get(key)
    if not folders:
        return root / name
    for folder in folders:
        candidate = root / folder
        if candidate.exists():
            return candidate
    return root / folders[0]


@dataclass(frozen=True)
class DatasetStatus:
    key: str
    path: Path
    exists: bool
    n_las: int = 0
    n_csv: int = 0
    has_manifest: bool = False

    @property
    def found(self) -> bool:
        return self.exists and (self.n_las + self.n_csv + int(self.has_manifest)) > 0


def _count_files(folder: Path) -> tuple[int, int, bool]:
    n_las = n_csv = 0
    has_manifest = False
    if not folder.is_dir():
        return 0, 0, False
    for dirpath, dirnames, filenames in os.walk(folder):
        # skip huge or irrelevant trees if someone pointed at a cache by mistake
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".venv"}]
        for name in filenames:
            lower = name.lower()
            if lower in {"wells.json", "wells.manifest.json"}:
                has_manifest = True
            elif lower.endswith(".las"):
                n_las += 1
            elif lower.endswith(".csv"):
                n_csv += 1
    return n_las, n_csv, has_manifest


def dataset_status(name: str | None = None) -> list[DatasetStatus]:
    """Report which known datasets are present. ``name`` limits the report to one key."""
    keys = [normalise_dataset(name)] if name else list(DATASETS)
    out: list[DatasetStatus] = []
    for key in keys:
        if key not in DATASETS:
            path = get_dataset_path(key)
            n_las, n_csv, has_manifest = _count_files(path)
            out.append(DatasetStatus(key, path, path.exists(), n_las, n_csv, has_manifest))
            continue
        path = get_dataset_path(key)
        n_las, n_csv, has_manifest = _count_files(path)
        out.append(DatasetStatus(key, path, path.exists(), n_las, n_csv, has_manifest))
    return out


def write_local_paths(data_root_path: Path | str, dest: Path | None = None) -> Path:
    """Write a minimal ``data_paths.local.yaml``. Does not overwrite without dest existing check
    by the caller; this always writes ``dest``."""
    dest = Path(dest) if dest is not None else REPO / "data_paths.local.yaml"
    root = Path(data_root_path).expanduser()
    text = (
        "# Local data locations for this machine. Do not commit this file.\n"
        f"data_root: {_yaml_string(root)}\n"
        "\n"
        "# Optional overrides and cache — uncomment only if needed:\n"
        "# force2020: \"/path/to/FORCE2020\"\n"
        "# sodir_factpages: \"/path/to/Sodir-FactPages\"\n"
        "# nlog_dts: \"/path/to/NLOG-DTS\"\n"
        "# kansas_kgs_dts: \"/path/to/Kansas-KGS-DTS\"\n"
        "# volve: \"/path/to/Volve\"\n"
        "# cache_dir: \"/path/to/fh2026-cache\"\n"
    )
    dest.write_text(text, encoding="utf-8")
    _reset_for_tests()
    return dest


def _yaml_string(path: Path) -> str:
    # Forward slashes work on Windows and avoid a flood of backslashes in YAML.
    text = path.as_posix()
    return '"' + text.replace("\\", "/").replace('"', '\\"') + '"'


def run_check() -> int:
    try:
        cfg = load_config()
    except DataPathsError as exc:
        print(exc, file=sys.stderr)
        return 1
    root = data_root()
    print(f"config:    {cfg['_path']}")
    print(f"data_root: {root}" + ("  (ok)" if root.is_dir() else "  (MISSING)"))
    print(f"cache_dir: {cache_dir()}")
    print(f"work_root: {work_root()}")
    print()
    rows = dataset_status()
    width = max(len(r.key) for r in rows)
    found_n = 0
    for row in rows:
        if row.found:
            found_n += 1
            bits = []
            if row.n_las:
                bits.append(f"{row.n_las} LAS")
            if row.n_csv:
                bits.append(f"{row.n_csv} CSV")
            if row.has_manifest:
                bits.append("wells.json")
            detail = ", ".join(bits) or "present"
            print(f"  {row.key:<{width}}  found   {row.path}  ({detail})")
        elif row.exists:
            print(f"  {row.key:<{width}}  empty   {row.path}")
        else:
            print(f"  {row.key:<{width}}  missing {row.path}")
    print()
    print(f"{found_n}/{len(rows)} datasets found")
    return 0 if root.is_dir() else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wmb.paths",
        description="Show or check local data paths (see docs/SETUP_DATA_PATHS.md).",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="list each dataset and whether it was found (default)",
    )
    ap.parse_args(argv)
    return run_check()


if __name__ == "__main__":
    sys.exit(main())
