"""Align a local well-log dataset to the FORCE / canonical standard.

    python -m wmb.align --input <raw folder> --output <aligned folder> --dataset nlog_dts

Data is read from --input and written to --output. Neither path should be
inside the git repository's tracked tree; this command never stages or commits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wmb.align.pipeline import align_dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wmb.align", description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="folder with one subfolder per well (raw LAS)")
    ap.add_argument("--output", required=True, type=Path, help="folder to write aligned LAS, wells.json, README")
    ap.add_argument("--dataset", required=True, help="dataset key, e.g. nlog_dts or kansas_kgs_dts")
    ap.add_argument("--alias-set", default=None,
                    help="alias list in canonical_curves.yaml (default: from datasets/<id>/mapping.yaml)")
    ap.add_argument("--wells", default=None, help="comma-separated well folder names (default: every subfolder)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="source wells.json to copy coordinates and credit from (default: <input>/wells.json)")
    args = ap.parse_args(argv)
    if not args.input.is_dir():
        print(f"input folder not found: {args.input}", file=sys.stderr)
        return 2
    names = [s.strip() for s in args.wells.split(",") if s.strip()] if args.wells else None
    result = align_dataset(args.input, args.output, args.dataset, alias_set=args.alias_set,
                           well_names=names, source_manifest=args.manifest)
    s = result["summary"]
    print(f"aligned {s['n_wells_aligned']}/{s['n_wells_in']} wells -> {result['output']}")
    print("raw mnemonics used:", s["n_raw_mnemonics"])
    if s["flag_counts"]:
        print("flag counts:", s["flag_counts"])
    if s["well_errors"]:
        print(f"well errors: {s['n_well_errors']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
