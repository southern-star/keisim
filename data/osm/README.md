# OpenStreetMap extracts of the real towns

Overpass API answers (`keisim/realworld.py` `fetch_osm`, the query of `overpass_query`) for the boxes of
`keisim.realworld.OSM_BOXES`. With these files here, `scripts/import_osm.py --name <name>`,
`scripts/evaluate.py --osm_town <name>` and `collect.py --osm_towns ...` need no network.

| file | box (south, west, north, east) | use |
|---|---|---|
| `osm_sapporo.json` | 43.088, 141.360, 43.097, 141.372 | evaluation town (held out), also `web/towns/town_osm_sapporo.json` |
| `osm_sapporo_n.json` | 43.097, 141.360, 43.106, 141.372 | training (KeiPilot v0.9) |
| `osm_sapporo_e.json` | 43.088, 141.372, 43.097, 141.384 | training |
| `osm_sapporo_s.json` | 43.079, 141.360, 43.088, 141.372 | training |
| `osm_nishioi.json` | 35.612, 139.719, 35.623, 139.732 | 品川区 西大井・二葉 (Tokyo): one-way streets, zero-shot evaluation, also `web/towns/town_osm_nishioi.json` (fetched 2026-10-25) |

The Sapporo files were fetched on 2026-10-20. Map data © OpenStreetMap contributors, available under the
[Open Database License (ODbL) 1.0](https://www.openstreetmap.org/copyright). These files, and the towns derived from
them (`web/towns/town_osm_*.json`), are derivative databases under the ODbL; the rest of this repository is MIT.
