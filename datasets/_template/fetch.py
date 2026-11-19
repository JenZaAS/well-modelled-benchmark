#!/usr/bin/env python3
"""Download official files for this dataset into the local data subfolder.

Copy this script into datasets/<id>/fetch.py and point it at the real URLs
and at checksums.sha256 in the same folder. It must never write data into
the git repository. Prefer verifying checksums after each download and
refusing to overwrite a file whose checksum already matches.

Example:

    python datasets/<id>/fetch.py --dest /path/to/data_root/<Folder>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        help="Destination folder (the dataset subfolder under data_root).",
    )
    parser.parse_args(argv)
    print(
        "This is a template. Copy it into a real dataset folder and implement "
        "the download. See datasets/force2020/fetch.py for a working example.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
