#!/usr/bin/env python3
"""Download official FORCE 2020 files into a local FORCE2020 data folder.

Reads URLs and SHA-256 checksums from mapping.yaml in this folder. Writes only
to --dest (or to FORCE2020 under data_root). Never writes data into the git
repository.

A file whose checksum already matches is left untouched. A file with the wrong
checksum is reported and left unchanged unless you pass --force. Downloads and
extracted files are written to a temporary name and only published after the
checksum matches.

    python datasets/force2020/fetch.py --dest /path/to/data_root/FORCE2020
    python datasets/force2020/fetch.py              # uses data_paths.local.yaml
    python datasets/force2020/fetch.py --required   # CSVs via train.zip only
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

PACK_DIR = Path(__file__).resolve().parent
CHUNK = 1024 * 1024
USER_AGENT = "wmb-force2020-fetch/1"

REQUIRED_DOWNLOADS = {
    "train.zip",
    "leaderboard_test_features.csv",
    "leaderboard_test_target.csv",
    "hidden_test.csv",
}


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a mapping")
    return data


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _dest_from_paths() -> Path:
    try:
        from wmb.paths import get_dataset_path
    except ImportError as exc:
        raise SystemExit(
            "Cannot import wmb.paths. Run from the repo after `pip install -e .`, "
            "or pass --dest explicitly."
        ) from exc
    return get_dataset_path("force2020")


def _download_to_temp(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        return tmp
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _publish(tmp: Path, dest: Path, expected: str | None) -> None:
    """Move tmp into dest only after the checksum matches. dest is unchanged on failure."""
    try:
        digest = _sha256(tmp)
        if expected and digest != expected:
            raise ValueError(f"SHA-256 mismatch (got {digest})")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _unzip_train_to_temp(zpath: Path, dest_csv: Path) -> Path:
    tmp = dest_csv.with_name("train.csv.part")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(zpath) as zf:
        members = [n for n in zf.namelist() if Path(n).name == "train.csv"]
        if not members:
            raise ValueError(f"{zpath}: no train.csv inside the zip")
        with zf.open(members[0]) as src, tmp.open("wb") as out:
            while True:
                chunk = src.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
    return tmp


def _has_hash(path: Path, expected: str | None) -> bool:
    return bool(expected) and path.is_file() and _sha256(path) == expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, help="FORCE2020 folder under your data root")
    parser.add_argument(
        "--required",
        action="store_true",
        help="Download only train.zip and the other required CSVs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a local file whose checksum does not match (default: leave it and fail).",
    )
    args = parser.parse_args(argv)

    mapping = _load_yaml(PACK_DIR / "mapping.yaml")
    dest = (args.dest.expanduser() if args.dest else _dest_from_paths())
    dest.mkdir(parents=True, exist_ok=True)
    print(f"destination: {dest}")

    failed = 0
    zip_spec = next((f for f in mapping.get("files") or [] if f["name"] == "train.zip"), None)
    expected_zip = (zip_spec or {}).get("sha256")

    for spec in mapping.get("files") or []:
        name = spec["name"]
        if name == "train.csv":
            continue
        if args.required and name not in REQUIRED_DOWNLOADS:
            continue

        path = dest / name
        expected = spec.get("sha256")
        url = spec.get("source") or ""

        if _has_hash(path, expected):
            print(f"  ok        {name}")
            continue
        if path.is_file() and expected and not args.force:
            print(f"  MISMATCH  {name} (left unchanged; pass --force to replace)", file=sys.stderr)
            failed += 1
            continue
        if not url.startswith("http"):
            print(f"  MISSING   {name} (no download URL)", file=sys.stderr)
            failed += 1
            continue

        print(f"  download  {name}")
        try:
            tmp = _download_to_temp(url, path)
            _publish(tmp, path, expected)
        except Exception as exc:
            print(f"  ERROR     {name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"  ok        {name}")

    csv_spec = next((f for f in mapping.get("files") or [] if f["name"] == "train.csv"), None)
    csv_path = dest / "train.csv"
    expected_csv = (csv_spec or {}).get("sha256")
    zip_path = dest / "train.zip"
    zip_ok = _has_hash(zip_path, expected_zip)
    if _has_hash(csv_path, expected_csv):
        print("  ok        train.csv")
    elif zip_ok:
        if csv_path.is_file() and expected_csv and not args.force:
            print("  MISMATCH  train.csv (left unchanged; pass --force to replace from train.zip)", file=sys.stderr)
            failed += 1
        else:
            print("  unzip     train.zip -> train.csv")
            try:
                tmp = _unzip_train_to_temp(zip_path, csv_path)
                _publish(tmp, csv_path, expected_csv)
            except Exception as exc:
                print(f"  ERROR     train.csv: {exc}", file=sys.stderr)
                failed += 1
            else:
                print("  ok        train.csv")
    else:
        print("  MISSING   train.csv (need a checksum-valid train.zip first)", file=sys.stderr)
        failed += 1

    if failed:
        print(f"{failed} file(s) failed.", file=sys.stderr)
        return 1
    print("all requested files ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
