"""Coordinate conversions used to place wells on a WGS84 map.

Assumptions
-----------
* Sodir FactPages ``wlbNsDecDeg`` / ``wlbEwDecDeg`` are ED50 (EPSG:4230)
  geographic coordinates (International 1924 ellipsoid).
* FORCE 2020 ``X_LOC`` / ``Y_LOC`` match Sodir UTM easting/northing to
  typically < 50 m and every exact-matched well sits in UTM zone 31N, so
  the fallback CRS is ED50 / UTM zone 31N (EPSG:23031).
* Geographic positions are shifted to WGS84 (EPSG:4326) with EPSG:1613
  "ED50 to WGS 84 (24)", the 7-parameter position-vector Helmert transform
  recommended by EPSG for practical oil-industry use in the Norwegian
  North Sea south of 62°N (~1 m). Almost all FORCE wells sit in that area
  (northernmost around 61.8°N).
"""

from __future__ import annotations

import math
from typing import Tuple

# International 1924 (Hayford) — ED50
INTL1924_A = 6378388.0
INTL1924_F = 1.0 / 297.0

# WGS84
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563

# EPSG:1613 — ED50 to WGS 84 (24), position-vector convention
# tx, ty, tz (m); rx, ry, rz (arc-seconds); s (ppm)
EPSG_1613 = {
    "tx": -90.365,
    "ty": -101.13,
    "tz": -123.384,
    "rx": 0.333,
    "ry": 0.077,
    "rz": 0.894,
    "s_ppm": 1.994,
}

UTM_K0 = 0.9996
UTM_E0 = 500000.0
UTM_N0_NORTH = 0.0


def _ellipsoid_e2(f: float) -> float:
    return f * (2.0 - f)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float, a: float, f: float) -> Tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e2 = _ellipsoid_e2(f)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    n = a / math.sqrt(1.0 - e2 * sl * sl)
    x = (n + h) * cl * co
    y = (n + h) * cl * so
    z = (n * (1.0 - e2) + h) * sl
    return x, y, z


def ecef_to_geodetic(x: float, y: float, z: float, a: float, f: float) -> Tuple[float, float, float]:
    """Closed-form Bowring conversion, sufficient to millimetres."""
    e2 = _ellipsoid_e2(f)
    b = a * (1.0 - f)
    ep2 = (a * a - b * b) / (b * b)
    p = math.hypot(x, y)
    if p < 1e-12:
        lat = math.copysign(math.pi / 2.0, z)
        lon = 0.0
        sl = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sl * sl)
        h = abs(z) - n * (1.0 - e2)
        return math.degrees(lat), lon, h
    theta = math.atan2(z * a, p * b)
    st, ct = math.sin(theta), math.cos(theta)
    lat = math.atan2(z + ep2 * b * st * st * st, p - e2 * a * ct * ct * ct)
    lon = math.atan2(y, x)
    sl = math.sin(lat)
    n = a / math.sqrt(1.0 - e2 * sl * sl)
    h = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), h


def helmert_position_vector(
    x: float, y: float, z: float, *, tx: float, ty: float, tz: float, rx: float, ry: float, rz: float, s_ppm: float
) -> Tuple[float, float, float]:
    """EPSG position-vector (EPSG:9606) 7-parameter Helmert.

    Rotations in arc-seconds, scale in parts-per-million. Applied as
    ``X' = T + (1+s) R X`` with the position-vector rotation matrix.
    """
    rxr = math.radians(rx / 3600.0)
    ryr = math.radians(ry / 3600.0)
    rzr = math.radians(rz / 3600.0)
    k = 1.0 + s_ppm * 1e-6
    # R ≈ [[1, -rz, ry], [rz, 1, -rx], [-ry, rx, 1]]
    xp = tx + k * (x - rzr * y + ryr * z)
    yp = ty + k * (rzr * x + y - rxr * z)
    zp = tz + k * (-ryr * x + rxr * y + z)
    return xp, yp, zp


def ed50_ll_to_wgs84(lat_ed50: float, lon_ed50: float, h: float = 0.0) -> Tuple[float, float]:
    """ED50 geographic (EPSG:4230) → WGS84 (EPSG:4326) via EPSG:1613."""
    x, y, z = geodetic_to_ecef(lat_ed50, lon_ed50, h, INTL1924_A, INTL1924_F)
    x, y, z = helmert_position_vector(x, y, z, **EPSG_1613)
    lat, lon, _ = ecef_to_geodetic(x, y, z, WGS84_A, WGS84_F)
    return lat, lon


def utm_to_ll(easting: float, northing: float, zone: int = 31, northern: bool = True, a: float = INTL1924_A, f: float = INTL1924_F) -> Tuple[float, float]:
    """Inverse UTM on the given ellipsoid (default International 1924 / ED50)."""
    e2 = _ellipsoid_e2(f)
    ep2 = e2 / (1.0 - e2)
    n0 = 0.0 if northern else 10000000.0
    x = easting - UTM_E0
    y = northing - n0
    lon0 = math.radians(zone * 6 - 183)

    e1 = (1.0 - math.sqrt(1.0 - e2)) / (1.0 + math.sqrt(1.0 - e2))
    m = y / UTM_K0
    mu = m / (a * (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2 ** 3 / 256.0))
    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1 ** 3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1 * e1 / 16.0 - 55.0 * e1 ** 4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1 ** 3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1 ** 4 / 512.0) * math.sin(8.0 * mu)
    )
    sp, cp = math.sin(phi1), math.cos(phi1)
    t = sp / cp if abs(cp) > 1e-16 else 0.0
    t2 = t * t
    n = a / math.sqrt(1.0 - e2 * sp * sp)
    r = a * (1.0 - e2) / (1.0 - e2 * sp * sp) ** 1.5
    d = x / (n * UTM_K0)
    d2, d3, d4 = d * d, d ** 3, d ** 4
    d5, d6 = d ** 5, d ** 6
    lat = phi1 - (n * t / r) * (
        d2 / 2.0
        - (5.0 + 3.0 * t2 + 10.0 * ep2 * cp * cp - 4.0 * ep2 * ep2 * cp ** 4 - 9.0 * ep2 * t2) * d4 / 24.0
        + (61.0 + 90.0 * t2 + 298.0 * ep2 * cp * cp + 45.0 * t2 * t2 - 252.0 * ep2 * t2 - 3.0 * ep2 * ep2 * cp ** 4)
        * d6
        / 720.0
    )
    lon = lon0 + (
        d
        - (1.0 + 2.0 * t2 + ep2 * cp * cp) * d3 / 6.0
        + (5.0 - 2.0 * ep2 * cp * cp + 28.0 * t2 - 3.0 * ep2 * ep2 * cp ** 4 + 8.0 * ep2 * t2 + 24.0 * t2 * t2)
        * d5
        / 120.0
    ) / cp
    return math.degrees(lat), math.degrees(lon)


def utm_ed50_to_wgs84(easting: float, northing: float, zone: int = 31, northern: bool = True) -> Tuple[float, float]:
    """ED50 UTM (default zone 31N, EPSG:23031) → WGS84 geographic."""
    lat, lon = utm_to_ll(easting, northing, zone=zone, northern=northern, a=INTL1924_A, f=INTL1924_F)
    return ed50_ll_to_wgs84(lat, lon)
