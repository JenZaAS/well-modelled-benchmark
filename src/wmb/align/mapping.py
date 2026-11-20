"""Canonical mapping, unit conversion and guardrails for dataset alignment.

The shared dictionary is ``src/wmb/datasets/canonical_curves.yaml``. A
per-dataset ``datasets/<id>/mapping.yaml`` may select an alias set and add
extras. Matching is case-insensitive; ``-`` counts as ``_``; a trailing
``:N`` or ``_NO_N`` is stripped and that copy is deprioritised.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wmb.curves import CORE
from wmb.io import SENTINELS
from wmb.paths import repo_root

# Output index name in aligned files (FORCE 2020 CSV / LAS convention).
DEPTH_OUT = "DEPTH_MD"
NULL_OUT = -999.25
GRID_HALF = 0.5  # snap a sample to a grid node if |d - node| <= step * this

# Extra curves preserved when present; not part of wmb.curves.CORE.
EXTRA_CURVES = {
    "BS": {
        "standard_name": "BS",
        "standard_unit": "in",
        "aliases": ["BS", "BIT", "BITSIZE", "BS_RM"],
        "unit_aliases": {"in": ["in", "IN", "inches", "inch", "H.IN"]},
        "conversions_to_standard": {"mm": 0.03937007874, "cm": 0.3937007874},
        "plausible_range": [4, 30],
        "plausible_median": [4.5, 28],
        "exclude": ["CALI", "HCAL", "DCAL"],
    },
    "RSHA": {
        "standard_name": "RSHA",
        "standard_unit": "ohm.m",
        "aliases": ["RSHA", "MSFL", "SFL"],
        "unit_aliases": {"ohm.m": ["ohm.m", "OHMM", "OHM-M", "Ohm-m", "ohmm"]},
        "conversions_to_standard": {"mS/m": "conductivity"},
        "plausible_range": [0.01, 100000],
        "plausible_median": [0.05, 5000],
        "exclude": ["RDEP", "RMED", "RT", "LLD", "ILD"],
    },
}

TVD_MNEMONICS = ("TVD", "TVDSS", "TVD_M", "TVDFT", "TVDEPT", "DVER", "SSDEPT")
DEV_MNEMONICS = ("DEVI", "DEVIATION", "INCL", "INCLINATION", "AZIM", "AZIMUTH", "SDEV", "DEVIA")
# DEV is left out: too many false hits (DEVICE, etc.). Accept it only with a survey description.

COMPUTED_TOKENS = ("LFP", "CPI", "PETRO", "APPENDNTR", "_EDITED", "COMPUTED")
CONDUCTIVITY_UNITS = {"MS/M", "MS/M", "MM/M", "MMHO", "MMHOS", "MSCM", "MHO/M"}
# normalised forms
_COND_UNITS = {"MS/M", "MM/M", "MMHO", "MMHOS", "MHO/M", "MSCM"}
_JUNK_UNIT_RE = re.compile(r"^:|^\d+$", re.I)
_NO_N_RE = re.compile(r"_NO_\d+$", re.I)
_TRAILING_NUM_RE = re.compile(r":\d+$")

# Sentinels used in the wild, in addition to wmb.io.SENTINELS.
_EXTRA_SENTINELS = (-999.95, -999.9, 999.95, -99999.25)


def all_sentinels() -> tuple[float, ...]:
    return tuple(dict.fromkeys((*SENTINELS, *_EXTRA_SENTINELS)))


def clean_values(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(x)
    for s in all_sentinels():
        x[np.isclose(x, s, rtol=0, atol=1e-6)] = np.nan
    # some files use very large placeholders
    x[finite & (np.abs(x) >= 1e30)] = np.nan
    return x


def norm_mnem(m: str) -> str:
    m = str(m).strip().upper().replace("-", "_").replace(" ", "_")
    if ":" in m:
        m = m.split(":", 1)[0]
    return _NO_N_RE.sub("", m)


def is_deprioritized_copy(raw: str) -> bool:
    s = str(raw).strip().upper()
    return bool(_TRAILING_NUM_RE.search(s) or _NO_N_RE.search(s.replace("-", "_")))


def norm_unit(u: str | None) -> str:
    if u is None:
        return ""
    s = str(u).strip()
    if not s or s.upper() in {"UNKNOWN", "NONE", "UNITLESS", "NA", "N/A"}:
        return ""
    if _JUNK_UNIT_RE.match(s) or s.startswith(":"):
        return ""
    s = s.replace(" ", "").replace("_", "")
    return s.upper()


def load_canonical_spec() -> dict:
    from wmb.datasets import load_canonical_spec as _load

    return _load()


def load_dataset_mapping(dataset_id: str | None) -> dict:
    if not dataset_id:
        return {}
    path = repo_root() / "datasets" / dataset_id / "mapping.yaml"
    if not path.is_file():
        return {}
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_align_spec(dataset_id: str | None = None, alias_set: str | None = None) -> dict:
    """Merge the canonical dictionary with an optional per-dataset mapping file."""
    canon = load_canonical_spec()
    pack = load_dataset_mapping(dataset_id) if dataset_id else {}
    align = dict(pack.get("alignment") or {})
    if alias_set:
        align["alias_set"] = alias_set
    elif "alias_set" not in align and dataset_id:
        defaults = {"nlog_dts": "NLOG-DTS", "nlog": "NLOG-DTS",
                    "kansas_kgs_dts": "Kansas-KGS-DTS", "kansas_kgs": "Kansas-KGS-DTS",
                    "force2020": "FORCE-2020", "volve": "Volve"}
        align["alias_set"] = defaults.get(dataset_id, "all")
    return {"canonical": canon, "pack": pack, "alignment": align, "dataset_id": dataset_id}


def curve_spec(spec: dict, name: str) -> dict | None:
    if name in EXTRA_CURVES:
        return EXTRA_CURVES[name]
    curves = spec["canonical"]["curves"]
    return curves.get(name)


def standard_curves(spec: dict) -> list[str]:
    extras = list(spec["alignment"].get("extra_curves") or ["BS", "RSHA"])
    out = list(CORE)
    for e in extras:
        if e not in out:
            out.append(e)
    return out


def aliases_for(spec: dict, name: str) -> list[str]:
    """Priority-ordered aliases for a standard curve in this dataset."""
    cs = curve_spec(spec, name)
    if not cs:
        return [name]
    raw = cs.get("aliases", [name])
    alias_set = spec["alignment"].get("alias_set", "all")
    if isinstance(raw, dict):
        ordered: list[str] = []
        if alias_set and alias_set != "all" and alias_set in raw:
            ordered.extend(raw[alias_set])
        # Always accept the standard name first if the dataset already uses it.
        if isinstance(raw.get("FORCE-2020"), list):
            for a in raw["FORCE-2020"]:
                if a not in ordered:
                    ordered.insert(0, a) if a == name else ordered.append(a)
        if name not in {norm_mnem(a) for a in ordered}:
            ordered.insert(0, name)
        # Last-resort: other datasets' aliases, after the preferred set.
        for key, vals in raw.items():
            if key in {alias_set, "FORCE-2020"}:
                continue
            for a in vals or []:
                if a not in ordered:
                    ordered.append(a)
        aliases = ordered
    else:
        aliases = list(raw)
        if name not in aliases:
            aliases.insert(0, name)
    # Dedup by normalised mnemonic, keep first (highest priority).
    seen: set[str] = set()
    out = []
    for a in aliases:
        n = norm_mnem(a)
        if n and n not in seen:
            seen.add(n)
            out.append(a)
    return out


def excluded(spec: dict, name: str) -> set[str]:
    cs = curve_spec(spec, name) or {}
    return {norm_mnem(x) for x in cs.get("exclude") or []}


def _unit_groups(cs: dict) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for key, aliases in (cs.get("unit_aliases") or {}).items():
        groups[key] = {norm_unit(a) for a in aliases} | {norm_unit(key)}
    return groups


def lookup_unit(cs: dict, unit: str) -> str | None:
    """Return the group name (standard unit or conversion source) for a raw unit string."""
    n = norm_unit(unit)
    if not n:
        return None
    # conductivity first (shared)
    if n in _COND_UNITS or n in {"MS/M"}:
        return "mS/m"
    for key, names in _unit_groups(cs).items():
        if n in names:
            return key
    return None


def _as_factor(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def infer_and_convert(values: np.ndarray, spec: dict, name: str, unit: str | None,
                      description: str = "") -> tuple[np.ndarray, list[str], str]:
    """Convert a curve to the standard unit. Returns (values, flags, unit_used)."""
    cs = curve_spec(spec, name)
    if cs is None:
        return values, [], unit or ""
    flags: list[str] = []
    std = cs.get("standard_unit", "")
    x = clean_values(values)
    good = x[np.isfinite(x)]
    median = float(np.median(good)) if good.size else float("nan")
    group = lookup_unit(cs, unit or "")

    conversions = cs.get("conversions_to_standard") or {}

    if name == "NPHI":
        x, f, used = _convert_nphi(x, median, group, unit or "", cs)
        return x, f, used
    if name in {"DTC", "DTS"}:
        x, f, used = _convert_sonic(x, median, group, unit or "", cs, name)
        return x, f, used
    if name == "RHOB":
        x, f, used = _convert_rhob(x, median, group, unit or "", cs)
        return x, f, used

    if group == std or group is None and not norm_unit(unit or ""):
        if group is None and good.size:
            flags.append("unit_inferred_from_values")
        # resistivity conductivity if unit says so
        if name in {"RDEP", "RMED", "RSHA"} and group == "mS/m":
            x, flags = _convert_conductivity(x, flags)
            return x, flags, std
        return x, flags, std

    if group == "mS/m" and name in {"RDEP", "RMED", "RSHA"}:
        x, flags = _convert_conductivity(x, flags)
        return x, flags, std

    factor = _as_factor(conversions.get(group)) if group else None
    if factor is not None:
        x = x * factor
        if group == "ft" or name == "DEPTH":
            flags.append("depth_converted_from_ft")
        return x, flags, std

    if group is None and good.size:
        flags.append("unit_inferred_from_values")
    return x, flags, std or (unit or "")


def _convert_nphi(x: np.ndarray, median: float, group: str | None, unit: str, cs: dict):
    flags: list[str] = []
    if group == "percent" or (np.isfinite(median) and median > 1.5):
        # PU labelled but already v/v (median in -0.2..0.7) is handled by the median check.
        if group == "percent" and np.isfinite(median) and -0.2 <= median <= 0.7:
            flags.append("unit_inferred_from_values")
            return x, flags, "v/v"
        x = x * 0.01
        flags.append("nphi_converted_from_percent")
        return x, flags, "v/v"
    if group in {None, "v/v"}:
        if group is None and np.isfinite(median):
            flags.append("unit_inferred_from_values")
        return x, flags, "v/v"
    return x, flags, "v/v"


def _convert_sonic(x: np.ndarray, median: float, group: str | None, unit: str, cs: dict, name: str):
    flags: list[str] = []
    factor = 0.3048
    if group == "us/m" or (group == "us/ft" and np.isfinite(median) and median > 250):
        x = x * factor
        flags.append("sonic_converted_from_us_m")
        if group == "us/ft":
            flags.append("unit_inferred_from_values")
        return x, flags, "us/ft"
    if group == "us/m":
        x = x * factor
        flags.append("sonic_converted_from_us_m")
        return x, flags, "us/ft"
    # labelled us/m but values already look like us/ft
    if group == "us/m" and np.isfinite(median) and 40 <= median <= 200:
        flags.append("unit_inferred_from_values")
        return x, flags, "us/ft"
    if group is None and np.isfinite(median):
        if median > 250:
            x = x * factor
            flags.append("sonic_converted_from_us_m")
            flags.append("unit_inferred_from_values")
        else:
            flags.append("unit_inferred_from_values")
    return x, flags, "us/ft"


def _convert_rhob(x: np.ndarray, median: float, group: str | None, unit: str, cs: dict):
    flags: list[str] = []
    if group == "kg/m3" or (np.isfinite(median) and median > 50):
        x = x * 0.001
        flags.append("density_converted_from_kg_m3")
        if group != "kg/m3":
            flags.append("unit_inferred_from_values")
        return x, flags, "g/cm3"
    if group is None and np.isfinite(median):
        flags.append("unit_inferred_from_values")
    return x, flags, "g/cm3"


def _convert_conductivity(x: np.ndarray, flags: list[str]) -> tuple[np.ndarray, list[str]]:
    y = np.asarray(x, float).copy()
    good = np.isfinite(y) & (y != 0)
    y[good] = 1000.0 / y[good]
    y[np.isfinite(x) & (x == 0)] = np.nan
    flags = list(flags) + ["resistivity_converted_from_conductivity"]
    return y, flags


def convert_depth(depth: np.ndarray, unit: str | None, values_hint: np.ndarray | None = None) -> tuple[np.ndarray, list[str], str]:
    """Convert a depth index to metres. Rejects a ``.1IN`` index."""
    flags: list[str] = []
    n = norm_unit(unit)
    x = clean_values(depth)
    if n in {".1IN", "0.1IN", "0.1INCH"}:
        raise ValueError("depth index unit .1IN is not a well depth (not ft or m)")
    if n in {"FT", "F", "FEET", "FOOT"}:
        return x * 0.3048, ["depth_converted_from_ft"], "m"
    if n in {"M", "METRE", "METER", "METRES", "METERS", "I.M", "HM", "H.M", "IM"}:
        return x, flags, "m"
    # Infer from values: Kansas / imperial wells are typically < 30,000 ft; metre wells rarely exceed 8000 m.
    good = x[np.isfinite(x)]
    if good.size:
        mx, med = float(np.nanmax(np.abs(good))), float(np.median(good))
        if mx > 8000 or (med > 2500 and mx > 4000):
            # likely feet
            flags += ["depth_converted_from_ft", "unit_inferred_from_values"]
            return x * 0.3048, flags, "m"
        flags.append("unit_inferred_from_values")
    return x, flags, "m"


def nphi_matrix_flag(mnemonic: str, spec: dict) -> str | None:
    cs = curve_spec(spec, "NPHI") or {}
    matrix = cs.get("matrix") or {}
    n = norm_mnem(mnemonic)
    for kind, names in matrix.items():
        if n in {norm_mnem(a) for a in names} and kind in {"sandstone", "limestone", "dolomite"}:
            return f"nphi_{kind}_matrix"
    return None


def description_mentions(desc: str, *words: str) -> bool:
    d = (desc or "").lower()
    return any(w.lower() in d for w in words)


@dataclass
class Candidate:
    standard: str
    mnemonic: str
    unit_in: str
    unit_out: str
    description: str
    path: str
    values: np.ndarray
    depth_m: np.ndarray
    flags: list[str] = field(default_factory=list)
    rejected: str | None = None
    alias_rank: int = 999
    deprioritized: bool = False
    computed_file: bool = False

    @property
    def n_valid(self) -> int:
        return int(np.isfinite(self.values).sum())

    @property
    def median(self) -> float:
        good = self.values[np.isfinite(self.values)]
        return float(np.median(good)) if good.size else float("nan")

    @property
    def span_m(self) -> float:
        d = self.depth_m[np.isfinite(self.values) & np.isfinite(self.depth_m)]
        return float(d.max() - d.min()) if d.size else 0.0


def is_computed_source(path: str, mnemonic: str) -> bool:
    blob = f"{path} {mnemonic}".upper().replace("-", "_")
    return any(tok in blob for tok in COMPUTED_TOKENS)


def assess_candidate(cand: Candidate, spec: dict) -> Candidate:
    """Accept or reject a mapped curve after unit conversion and guardrails."""
    name, cs = cand.standard, curve_spec(spec, cand.standard)
    desc = cand.description or ""
    unit_n = norm_unit(cand.unit_in)
    mnem = norm_mnem(cand.mnemonic)

    if cand.n_valid == 0:
        cand.rejected = "empty_or_all_null"
        return cand
    if name == "DTS" and (mnem == "DTST" or description_mentions(desc, "stoneley")):
        cand.rejected = "stoneley_not_shear"
        return cand
    if name in {"RDEP", "RMED", "RSHA"}:
        if unit_n in {"IN", "INCH", "INCHES"}:
            cand.rejected = "resistivity_unit_is_inches"
            return cand
        if unit_n in {"CPS", "CPM"}:
            cand.rejected = "resistivity_unit_is_cps"
            return cand
        if mnem == "RLL" and unit_n in {"CPS", "CPM"}:
            cand.rejected = "rll_cps"
            return cand
        # conductivity without conversion is rejected later only if we refuse to convert;
        # we do convert and flag.
    if name == "GR" and unit_n in {"2.1/S", "H2.1/S", "21/S"}:
        if not (5 <= cand.median <= 250):
            cand.rejected = "gr_unit_not_api"
            return cand
        cand.flags.append("gr_unit_unusual_accepted")
    if name == "CALI" and mnem == "DCAL":
        if description_mentions(desc, "differential"):
            cand.rejected = "dcal_is_differential"
            return cand
        if np.isfinite(cand.median) and abs(cand.median) < 3:
            cand.rejected = "dcal_near_zero"
            return cand
        # accept only if it looks like hole size (or the description says density caliper)
        if description_mentions(desc, "density caliper") or (5 <= cand.median <= 20):
            cand.flags.append("dcal_accepted_as_caliper")
        else:
            cand.rejected = "dcal_not_hole_size"
            return cand
    if name == "CALI" and np.isfinite(cand.median) and cand.median > 80:
        cand.rejected = "caliper_median_too_large"
        return cand

    # Values that look like a caliper sitting on a resistivity alias (unit missing or IN already rejected).
    if name in {"RDEP", "RMED"} and unit_n in {"", "IN"}:
        good = cand.values[np.isfinite(cand.values)]
        if good.size >= 20:
            med = float(np.median(good))
            if 5 <= med <= 15 and float((good < 2).mean()) < 0.02:
                cand.rejected = "resistivity_values_look_like_caliper"
                return cand

    if is_computed_source(cand.path, cand.mnemonic):
        cand.computed_file = True
        cand.flags.append("used_LFP_or_computed_curve")
    if cand.standard == "RMED" and mnem in {"RS", "RSHA"}:
        cand.flags.append("used_shallow_fallback_RS")
    if name == "NPHI":
        flag = nphi_matrix_flag(cand.mnemonic, spec)
        if flag:
            cand.flags.append(flag)

    lo_hi = (cs or {}).get("plausible_median")
    if lo_hi and np.isfinite(cand.median) and not (lo_hi[0] <= cand.median <= lo_hi[1]):
        cand.flags.append("median_outside_plausible_range")
    return cand


def pick_best(cands: list[Candidate]) -> Candidate | None:
    """Alias priority, then longest valid interval, then prefer raw over computed, then not :N copies."""
    ok = [c for c in cands if c.rejected is None and c.n_valid > 0]
    if not ok:
        return None

    def key(c: Candidate):
        return (c.alias_rank, -c.span_m, -c.n_valid, int(c.computed_file), int(c.deprioritized), c.path, c.mnemonic)

    ok.sort(key=key)
    return ok[0]


def range_flags(values: np.ndarray, spec: dict, name: str) -> dict:
    """Flag out-of-range samples; never drop them."""
    cs = curve_spec(spec, name) or {}
    good = values[np.isfinite(values)]
    meta: dict = {}
    if good.size == 0:
        return meta
    rng = cs.get("plausible_range")
    if rng:
        n_bad = int(np.sum((good < rng[0]) | (good > rng[1])))
        if n_bad:
            meta["samples_outside_plausible_range"] = n_bad
    med = cs.get("plausible_median")
    if med:
        m = float(np.median(good))
        if not (med[0] <= m <= med[1]):
            meta["median_outside_plausible_range"] = True
            meta["median"] = m
    return meta


def depth_flags(depth_m: np.ndarray) -> list[str]:
    flags = []
    d = depth_m[np.isfinite(depth_m)]
    if d.size == 0:
        return ["no_valid_depth"]
    if float(np.nanmin(d)) < 0:
        flags.append("negative_depth")
    if d.size and float(np.nanmax(d)) > 8000:
        flags.append("depth_exceeds_8000m")
    diffs = np.diff(d)
    if diffs.size and np.any(diffs < 0):
        flags.append("depth_not_increasing_in_source")
    return flags
