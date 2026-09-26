# Vendored: Sakuragaoka Station (MIT)

Source: https://github.com/Kenton-GMI/sakuragaoka-station @ `4112f57208b7e29998344ca71fef74202c2b2bdd`
License: MIT, see [LICENSE](LICENSE) (Copyright (c) 2026 Sakuragaoka Station contributors).

KeiView reuses Sakuragaoka Station's cel-shading renderer, material library, canvas textures,
static batching, first-person player, the parametric Japanese-house generator and the
cherry-tree generator. Paths mirror upstream `src/` so relative imports keep working.

| here | upstream | changes |
|---|---|---|
| `core/{ctx,materials,renderer,sky,textures,geo,batch2,physics,player}.js` | `src/core/` | none |
| `world/houses/*.js` | `src/world/houses/` | `far.js`: the far-town cell skip rule asks the layout shim (`L.farSkip`) instead of using Sakuragaoka coordinates |
| `world/lib/foliage.js` | `src/world/lib/` | none |
| `world/poles/{acc,atlas,facade}.js` | `src/world/poles/` | none |
| `world/sakura/{bases,canopy,materials,textures,tree,util}.js` | `src/world/sakura/` | none (`placements.js`, the hand-placed tree list, is not vendored) |
| `world/layout.js` | `src/world/layout.js` | **replaced** by a KeiView shim: `heightAt` hook (installed by the town loader), `SUN_DIR`, walk bounds, far-town rectangles + skip hook, `configureWorld()` |

Everything town-specific (roads, lots, trees, signals) is built by KeiView's own modules in
`web/src/` from a KeiSim town export (`scripts/export_town.py`). Files there that are derived from
upstream code say so in their header: `src/main.js` (bootstrap), `src/world/houses.js` (house-kit
helpers), `src/world/plots.js` (fields / car parks / vacant lots), `src/world/trees.js` (tree build
loop), `src/world/poles.js` (utility poles: pole planning, service-drop lots and guy-wire ground test
rewritten for KeiSim towns), `tools/check.mjs` and `tools/shot.mjs`.
