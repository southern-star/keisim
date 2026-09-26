// 桜並木 — KeiSim's street trees (outer edge of the sidewalk) grown as cherry trees by Sakuragaoka
// Station's procedural generator (sakura/tree.js) in square tree pits, canopies leaning over the road.
// The build loop follows Sakuragaoka Station src/world/sakura.js (MIT, see vendor/sakuragaoka/LICENSE).
import * as THREE from 'three';
import { createSakuraTextures } from '../../vendor/sakuragaoka/world/sakura/textures.js';
import { createSakuraMaterials } from '../../vendor/sakuragaoka/world/sakura/materials.js';
import { makeTree } from '../../vendor/sakuragaoka/world/sakura/tree.js';
import { createBaseBuilder } from '../../vendor/sakuragaoka/world/sakura/bases.js';
import { createNoise } from '../../vendor/sakuragaoka/world/sakura/util.js';

const sstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

export async function build(ctx) {
  const T = ctx.town, L = ctx.L;
  const Tx = createSakuraTextures(ctx);
  const M = createSakuraMaterials(ctx, Tx);
  const env = { rng: ctx.rng, noise: createNoise(ctx.rng('sakura-noise')), heightAt: L.heightAt };
  const R = ctx.rng(`kv-trees-${T.data.seed}`);
  const jit = (a) => (R() - 0.5) * 2 * a;
  // clearance floor for the canopy (absolute y): 2.45 m over sidewalks, ~4.3 m over the carriageway
  const floorAt = (x, z) => L.heightAt(x, z) + 2.45 + 1.85 * (1 - sstep(0, 1.3, T.roadDist(x, z)));

  // unit direction from each tree to the nearest road centre line (canopies lean over the road)
  const toRoad = (x, z) => {
    let best = Infinity, bx = 0, bz = 0;
    for (const r of T.roads) for (let i = 1; i < r.pts.length; i++) {
      const a = r.pts[i - 1], b = r.pts[i], dx = b[0] - a[0], dz = b[1] - a[1];
      const t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (z - a[1]) * dz) / (dx * dx + dz * dz || 1)));
      const px = a[0] + dx * t, pz = a[1] + dz * t, d = (px - x) ** 2 + (pz - z) ** 2;
      if (d < best) { best = d; bx = px; bz = pz; }
    }
    const l = Math.hypot(bx - x, bz - z) || 1;
    return [(bx - x) / l, (bz - z) / l];
  };

  const specs = T.trees.map((t, i) => {
    const [x, z] = t.p, [dx, dz] = toRoad(x, z);
    return {
      id: 'kv' + i, kind: 'medium', x, z, height: 5.9 + jit(0.7), spread: 3.2 + jit(0.35), vr: 0.56, trunkR: 0.22 + jit(0.03),
      forkH: 2.2 + jit(0.15), lean: [dx * 0.35 + jit(0.12), dz * 0.35 + jit(0.12)], offset: [dx * 0.95, dz * 0.95], limbs: 4,
      padR: 1.05, lod: 1, rootReach: 0.5, base: { type: 'pit', size: 1.2 }, bark: 'old', floorAt, seed: `kv-sakura-${T.data.seed}-${i}`,
    };
  });

  const root = new THREE.Group(); root.name = 'sakura';
  const bases = createBaseBuilder(ctx, M);
  const published = [];
  const blobCells = new Map();
  const stats = { trees: 0, bark: 0, blob: 0, cards: 0 };
  for (const spec of specs) {
    let t;
    try { t = makeTree(spec, env); } catch (e) { console.warn('[trees] tree failed', spec.id, e); continue; }
    const tree = { spec, ...t };
    const g = new THREE.Group(); g.name = 'sakura-' + spec.id;
    const barkGeo = t.bark.build(false);
    if (barkGeo) { const m = new THREE.Mesh(barkGeo, M.barkOld); m.castShadow = true; m.receiveShadow = true; g.add(m); }
    // blossom masses: merged per 48 m cell (kept out of the core batcher for the dappled shadow material)
    const blobGeo = t.blob.build(true);
    if (blobGeo) {
      const c = blobGeo.boundingSphere.center, key = Math.floor(c.x / 48) + ',' + Math.floor(c.z / 48);
      (blobCells.get(key) || blobCells.set(key, []).get(key)).push(blobGeo);
    }
    const cardGeo = t.cards.build(true);
    if (cardGeo) { const m = new THREE.Mesh(cardGeo, M.cards); m.castShadow = true; m.receiveShadow = false; ctx.noOutline(m); g.add(m); }
    root.add(g);
    for (const c of t.colliders) ctx.physics.addCylinder(c.x, c.z, c.r, c.y0, c.y1);
    bases.build(tree);
    stats.trees++; stats.bark += t.bark.tris; stats.blob += t.blob.tris; stats.cards += t.cards.tris;
    const I = t.info;
    published.push({ id: spec.id, x: +I.x.toFixed(2), z: +I.z.toFixed(2), y: +I.y.toFixed(2), r: +I.r.toFixed(2), h: +I.h.toFixed(2), trunk: { x: spec.x, z: spec.z, r: spec.trunkR } });
  }
  for (const [key, list] of blobCells) {
    const geo = list.length > 1 ? ctx.geo.mergeGeometries(list, false) : list[0];
    if (!geo) continue;
    if (list.length > 1) for (const gg of list) gg.dispose();
    geo.computeBoundingSphere(); geo.computeBoundingBox();
    const m = new THREE.Mesh(geo, M.blob);
    m.name = 'sakura-mass-' + key; m.castShadow = true; m.receiveShadow = false;
    m.customDepthMaterial = M.blobDepth;
    ctx.noBatch(m);
    root.add(m);
  }
  const B = bases.finish();
  for (const m of B.meshes) root.add(m);
  root.add(B.group);
  ctx.addStatic(root);
  ctx.services.sakura = { trees: published };
  ctx.sakuraStats = stats;
}
