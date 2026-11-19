"""Standard lithology colour schemes used by the viewer."""

from __future__ import annotations

# FORCE 2020 lithofacies (Bormann / Hall colour key, widely reused).
# Labels are CC-BY-4.0 (FORCE 2020 Machine Learning Contest).
FORCE2020_LITH = {
    30000: {"label": "Sandstone", "color": "#ffff00"},
    65030: {"label": "Sandstone/Shale", "color": "#ffe119"},
    65000: {"label": "Shale", "color": "#bebebe"},
    80000: {"label": "Marl", "color": "#7cfc00"},
    74000: {"label": "Dolomite", "color": "#8080ff"},
    70000: {"label": "Limestone", "color": "#80ffff"},
    70032: {"label": "Chalk", "color": "#ffff80"},
    88000: {"label": "Halite", "color": "#7dd7d7"},
    86000: {"label": "Anhydrite", "color": "#ff80ff"},
    99000: {"label": "Tuff", "color": "#ff99cc"},
    90000: {"label": "Coal", "color": "#1a1a1a"},
    93000: {"label": "Basement", "color": "#774936"},
}

SCHEMES = {"force2020": FORCE2020_LITH}


def scheme_payload(name: str | None) -> dict:
    key = (name or "force2020").lower()
    table = SCHEMES.get(key, FORCE2020_LITH)
    return {
        "id": key if key in SCHEMES else "force2020",
        "codes": {str(k): v for k, v in table.items()},
    }
