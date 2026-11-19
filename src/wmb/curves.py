"""Standard curve names and mnemonic mapping (from tools/fetch_open_data/fh_logs.py).

The 9 core curves are the benchmark vocabulary. Every dataset maps its own
mnemonics onto these names, ideally once, in its wells.json `curve_map`.
"""
from __future__ import annotations

import re

CORE = ["GR", "RDEP", "RMED", "CALI", "RHOB", "NPHI", "PEF", "DTC", "DTS"]
UNITS = {"GR": "gAPI", "RDEP": "ohm.m", "RMED": "ohm.m", "CALI": "in", "RHOB": "g/cm3",
         "NPHI": "v/v", "PEF": "b/e", "DTC": "us/ft", "DTS": "us/ft"}
LOG_SCALE = {"RDEP", "RMED"}          # model these in log10
LITHOLOGY = "LITH"                     # FORCE 2020 lithofacies code, when present

# Ordered: the first listed wins a coverage tie.
MNEMONICS: dict[str, list[str]] = {
    "GR": ["GR", "GRD", "GRR", "GRS", "GRSL", "SGRD", "ECGR", "GR_EDTC", "GR_TMG", "TGR", "HSGR", "GRN",
           "GAMMA", "GRGC", "GRC", "GAM", "GR_SPLICE", "GAMMA_RAY_SPLICE", "GR_PROCESS"],
    "RDEP": ["RDEP", "RD", "LLD", "HLLD", "ILD", "RILD", "DILD", "RT", "RESD", "AT90", "AHT90", "AIT90", "IDPH",
             "RLLD", "AO90", "HRLD", "THRT", "M4RX", "M4R9", "AILD", "TBIT90", "ATCO90", "AT60", "AF60", "AF90",
             "LLD_SPLICE", "LN", "64N", "64IN", "R64", "RLL", "IL"],
    "RMED": ["RMED", "RM", "LLS", "HLLS", "ILM", "RILM", "DILM", "AT30", "AHT30", "AIT30", "IMPH", "RLLS", "AO30",
             "HRLS", "M4R3", "M4R2", "TBIT30", "TBIT20", "AF30", "LLM_SPLICE", "LLS_SPLICE", "SN", "16N", "16IN",
             "R16", "AILM", "RS"],
    "CALI": ["CALI", "HCAL", "LCAL", "CAL", "CALF", "CAL1", "C1", "CAL2", "C2", "DCAL", "CALX", "CALY", "HDAR"],
    "RHOB": ["RHOB", "RHOZ", "RHOI", "DEN", "DENS", "ZDEN", "LDEN", "RHO8", "ZDL", "RHOM", "RHOC", "RHO",
             "RHOB_SPLICE", "DENB"],
    "NPHI": ["NPHI", "NPHIL", "TNPH", "NPOR", "NEU", "PHIN", "CNC", "CNL", "CNPOR", "CNCF", "NPHI_LS", "NPHS",
             "NPHL", "SNP", "CNLS", "NPLS", "NPSS", "NEUTRON_SPLICE", "NEUT"],
    "PEF": ["PEF", "PEFZ", "PEFI", "PE", "PEFS", "PEFL", "PEF8", "PEF_SL", "LPE"],
    "DTC": ["DTCO", "DTC", "DT4C", "DT4P", "DT24", "DT24QI", "DT", "AC", "BCSL", "SON", "DTCR", "DTL", "DTLN",
            "DTLF", "BHC", "DTP", "DTCO_MH"],
    "DTS": ["DTSM", "DTS", "DTSH", "DT4S", "DTSQI", "DT2", "ACS", "DTST", "DTSF", "DTSS", "DTRS", "DTSH_YD",
            "DTSH_MH", "DTSH_FAST", "DTSH_SLOW", "DT2R", "DTSH1", "DTS1"],
}
EXCLUDE = {
    "GR": {"GRMAX", "GRMIN", "THGR", "POTA", "URAN", "THOR", "SGR", "CGR", "HFK", "HTHO", "HURA"},
    "RDEP": {"RXO", "RMIC", "RSHA", "MSFL", "RXOZ", "RXOI", "RXO8", "RMUD"},
    "RMED": {"RXO", "RMIC", "RSHA", "MSFL", "RMUD"},
    "CALI": {"BS", "BIT", "BITSIZE", "BS_RM"},
    "RHOB": {"RHOFL", "RHOMA", "DRHO", "DRHOZ", "BAD_RHOB", "HDRA", "DPHI", "RHOB_PROCESSING", "RHOL"},
    "NPHI": {"DPHI"},
    "DTC": {"DTS", "DTSM", "DTSH", "DT4S", "DT2", "DTSF", "DTSS", "DTST", "DTSR", "DTSX"},
    "DTS": {"DTC", "DTCO", "DT4C", "DT4P", "DT", "DTCR", "DTCX", "DTCO_MH"},
}
DEPTH_MNEMONICS = {"DEPT", "DEPTH", "MD", "TDEP", "DEPTH_MD", "MEASURERD", "MEASUREDDEPTH"}
# FORCE 2020 CSV columns that already use (almost) standard names
FORCE_COLUMNS = {"GR": "GR", "RDEP": "RDEP", "RMED": "RMED", "CALI": "CALI", "RHOB": "RHOB", "NPHI": "NPHI",
                 "PEF": "PEF", "DTC": "DTC", "DTS": "DTS", "LITH": "FORCE_2020_LITHOFACIES_LITHOLOGY"}


def norm_mnem(m: str) -> str:
    m = str(m).strip().upper().replace("-", "_").replace(" ", "_")
    if ":" in m:
        m = m.split(":")[0]
    return re.sub(r"_NO_\d+$", "", m)


def match_property(mnemonic: str) -> str | None:
    """Map a raw LAS mnemonic to a core curve name, or None."""
    n = norm_mnem(mnemonic)
    if n in DEPTH_MNEMONICS:
        return None
    for prop, names in MNEMONICS.items():
        if n not in EXCLUDE.get(prop, set()) and n in names:
            return prop
    return None
