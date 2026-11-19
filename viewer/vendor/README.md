# Vendored front-end assets

These files are bundled into the generated `wellmap.html` so the viewer can
open without a network connection for libraries (map tiles still need the
internet; a Natural Earth coastline is drawn as an offline fallback).

| File | Project | Licence |
|------|---------|---------|
| `leaflet.js`, `leaflet.css` | [Leaflet 1.9.4](https://leafletjs.com) (Vladimir Agafonkin, CloudMade) | BSD-2-Clause |
| `ncs_coast.geojson` | [Natural Earth](https://www.naturalearthdata.com) 50m coastline, clipped to the North Sea / NCS | Public domain |

Do not add well log data to this folder.
