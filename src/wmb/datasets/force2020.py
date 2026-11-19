"""FORCE 2020 loader and file check. Pack: ``datasets/force2020/`` (mapping.yaml, checksums)."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np

from wmb import paths
from wmb.curves import CORE
from wmb.io import Well, _clean

DATASET_ID = "force2020"
SPLITS = ("train", "leaderboard", "hidden")
LITH_COL = "FORCE_2020_LITHOFACIES_LITHOLOGY"
LITH_CONF_COL = "FORCE_2020_LITHOFACIES_CONFIDENCE"
COORDS = ("X_LOC", "Y_LOC", "Z_LOC")
LABEL_COLUMNS = {"LITH": LITH_COL, "LITH_CONF": LITH_CONF_COL}


class ChecksumError(RuntimeError):
    """A data file does not match its published SHA-256."""


def load_mapping() -> dict:
    import yaml

    path = paths.repo_root() / "datasets" / DATASET_ID / "mapping.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_root(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else paths.get_dataset_path(DATASET_ID)


def _file_entries(mapping: dict) -> dict[str, dict]:
    return {f["name"]: f for f in mapping["files"]}


def ensure_train_csv(root: Path, mapping: dict) -> Path | None:
    """Return train.csv, unzipping a checksum-valid train.zip if needed. Never overwrites.

    Returns None when neither file is present. Raises ChecksumError on a mismatch.
    """
    entries = _file_entries(mapping)
    csv_path, zip_path = root / "train.csv", root / "train.zip"
    if csv_path.is_file():
        return csv_path
    if not zip_path.is_file():
        return None
    got = sha256_file(zip_path)
    expected_zip = entries["train.zip"]["sha256"]
    if got != expected_zip:
        raise ChecksumError(f"{zip_path}: SHA-256 mismatch (got {got}); file left unchanged")
    expected_csv = entries["train.csv"]["sha256"]
    tmp = csv_path.with_name("train.csv.part")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if Path(n).name == "train.csv"]
            if not members:
                raise ChecksumError(f"{zip_path}: no train.csv inside the zip")
            with zf.open(members[0]) as src, tmp.open("wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        digest = sha256_file(tmp)
        if digest != expected_csv:
            raise ChecksumError(
                f"{zip_path}: extracted train.csv SHA-256 mismatch (got {digest}); not written"
            )
        tmp.replace(csv_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return csv_path


def _first_finite(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    return float(v[0]) if v.size else float("nan")


def _plausibility_flags(curves: dict[str, np.ndarray], spec: dict) -> dict:
    out_of_range: dict[str, int] = {}
    median_flags: list[str] = []
    for name, x in curves.items():
        cs = spec["curves"].get(name)
        if not cs:
            continue
        good = x[np.isfinite(x)]
        if good.size == 0:
            continue
        lo, hi = cs["plausible_range"]
        n_bad = int(np.sum((good < lo) | (good > hi)))
        if n_bad:
            out_of_range[name] = n_bad
        mlo, mhi = cs["plausible_median"]
        if not (mlo <= float(np.median(good)) <= mhi):
            median_flags.append(name)
    meta: dict = {}
    if out_of_range:
        meta["samples_outside_plausible_range"] = out_of_range
    if median_flags:
        meta["median_outside_plausible_range"] = median_flags
    return meta


def _frame_to_wells(df, split: str, source: str, spec: dict) -> list[Well]:
    wells = []
    for name, g in df.groupby("WELL", sort=True):
        g = g.sort_values("DEPTH_MD", kind="stable")
        curves: dict[str, np.ndarray] = {}
        for std in CORE:
            if std in g:
                curves[std] = _clean(g[std].to_numpy(float))
        for std, col in LABEL_COLUMNS.items():
            if col in g:
                curves[std] = _clean(g[col].to_numpy(float))
        meta: dict = {"file": source}
        for c in COORDS:
            if c in g:
                meta[c] = _first_finite(_clean(g[c].to_numpy(float)))
        for c in ("GROUP", "FORMATION"):
            if c in g:
                meta[c] = g[c].to_numpy(object)
        meta.update(_plausibility_flags({k: v for k, v in curves.items() if k in CORE}, spec))
        wells.append(Well(str(name), g["DEPTH_MD"].to_numpy(float), curves, DATASET_ID, split, meta))
    return wells


def _read(path: Path):
    import pandas as pd

    return pd.read_csv(path, sep=";", low_memory=False)


def load_force2020(root: str | Path | None = None, splits: list[str] | None = None) -> list[Well]:
    """Load FORCE 2020 wells with canonical curve names (depth in metres MD).

    ``splits`` picks from train / leaderboard / hidden (default all). Sentinels
    become NaN; guardrail flags go on ``Well.meta``, values are not otherwise changed.
    """
    from wmb.datasets import load_canonical_spec

    splits = list(splits) if splits else list(SPLITS)
    unknown = set(splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}; choose from {list(SPLITS)}")
    root = resolve_root(root)
    mapping, spec = load_mapping(), load_canonical_spec()
    wells: list[Well] = []
    for split in SPLITS:
        if split not in splits:
            continue
        if split == "train":
            path = ensure_train_csv(root, mapping)
            if path is None:
                raise FileNotFoundError(f"{root / 'train.csv'} not found (and no train.zip). See datasets/force2020/README.md")
            df, source = _read(path), path.name
        elif split == "leaderboard":
            cfg = mapping["splits"]["leaderboard"]
            feats, target = root / cfg["features_file"], root / cfg["target_file"]
            for p in (feats, target):
                if not p.is_file():
                    raise FileNotFoundError(f"{p} not found. See datasets/force2020/README.md")
            tdf = _read(target)
            keep = ["WELL", "DEPTH_MD"] + [c for c in tdf.columns if c in (LITH_COL, LITH_CONF_COL)]
            df = _read(feats).merge(tdf[keep], on=["WELL", "DEPTH_MD"], how="left")
            source = f"{feats.name}+{target.name}"
        else:
            path = root / mapping["splits"]["hidden"]["file"]
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found. See datasets/force2020/README.md")
            df, source = _read(path), path.name
        wells.extend(_frame_to_wells(df, split, source, spec))
    return wells


def curve_availability(wells: list[Well]) -> dict[str, dict[str, int]]:
    """Wells with at least one valid sample per curve: {split or 'all': {'wells': n, curve: n}}."""
    names = CORE + ["LITH", "LITH_CONF"]
    table: dict[str, dict[str, int]] = {}
    for w in wells:
        for key in (w.split or "unknown", "all"):
            row = table.setdefault(key, {"wells": 0, **{c: 0 for c in names}})
            row["wells"] += 1
            for c in names:
                x = w.curves.get(c)
                if x is not None and np.isfinite(x).any():
                    row[c] += 1
    return table


def format_availability(table: dict[str, dict[str, int]]) -> str:
    cols = ["wells"] + CORE + ["LITH"]
    order = [s for s in (*SPLITS, "all") if s in table]
    lines = ["  " + f"{'split':<12}" + "".join(f"{c:>7}" for c in cols)]
    for s in order:
        lines.append("  " + f"{s:<12}" + "".join(f"{table[s][c]:>7}" for c in cols))
    return "\n".join(lines)


def check_force2020(root: str | Path | None = None, mapping: dict | None = None,
                    show_table: bool = True, out=None) -> int:
    """Verify required/optional files and checksums. Returns 0 only if all required files pass."""
    import sys

    out = out or sys.stdout
    mapping = mapping or load_mapping()
    try:
        root = resolve_root(root)
    except paths.DataPathsError as exc:
        print(f"{exc}\nRun: python -m wmb.onboarding  (or see docs/SETUP_DATA_PATHS.md)", file=sys.stderr)
        return 1
    print(f"force2020 folder: {root}", file=out)
    if not root.is_dir():
        print("  folder not found. See datasets/force2020/README.md and fetch.py", file=sys.stderr)
        return 1

    failures: list[str] = []
    entries = _file_entries(mapping)
    for name, entry in entries.items():
        role, path = entry["role"], root / name
        if name == "train.zip":
            continue  # handled together with train.csv
        if name == "train.csv":
            try:
                path = ensure_train_csv(root, mapping)
            except ChecksumError as exc:
                failures.append(str(exc))
                print(f"  MISMATCH  train.zip ({exc})", file=out)
                continue
            if path is None:
                failures.append("train.csv missing (and no train.zip)")
                print("  MISSING   train.csv (required; or provide a valid train.zip)", file=out)
                continue
        if not path.is_file():
            if role == "required":
                failures.append(f"{name} missing")
                print(f"  MISSING   {name} (required)", file=out)
            else:
                print(f"  warning   {name} missing ({role})", file=out)
            continue
        got = sha256_file(path)
        if got == entry["sha256"]:
            print(f"  ok        {name}", file=out)
        else:
            failures.append(f"{name}: SHA-256 mismatch")
            print(f"  MISMATCH  {name} (left unchanged; expected {entry['sha256'][:12]}..., got {got[:12]}...)", file=out)
    zip_path = root / "train.zip"
    if zip_path.is_file() and (root / "train.csv").is_file():
        if sha256_file(zip_path) != entries["train.zip"]["sha256"]:
            failures.append("train.zip: SHA-256 mismatch")
            print("  MISMATCH  train.zip (left unchanged)", file=out)
        else:
            print("  ok        train.zip", file=out)

    if failures:
        print(f"\nFAILED: {len(failures)} problem(s): " + "; ".join(failures), file=out)
        return 1
    if show_table:
        print("\nWells with at least one valid sample per curve:", file=out)
        print(format_availability(curve_availability(load_force2020(root))), file=out)
    print("\nOK: all required files present and checksums match.", file=out)
    return 0
