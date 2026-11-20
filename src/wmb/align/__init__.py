"""Align a well-log dataset to the FORCE / canonical curve standard.

Data never enters the git repo. Run the CLI against a local folder:

    python -m wmb.align --input <raw> --output <aligned> --dataset nlog_dts

See docs/data-alignment.md.
"""
from __future__ import annotations

from wmb.align.mapping import load_align_spec, norm_mnem, norm_unit
from wmb.align.pipeline import align_dataset, align_well, write_dataset

__all__ = [
    "align_dataset",
    "align_well",
    "load_align_spec",
    "norm_mnem",
    "norm_unit",
    "write_dataset",
]
