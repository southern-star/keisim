// 道路と歩道 — KeiSim's road network: asphalt, concrete gutters, raised sidewalks with kerbs,
// lane paint, stop lines and zebra crossings. Surfaces are the traced unions from the export
// (road / sidewalk-only regions), so kerbs follow the exact KeiSim geometry.
import * as THREE from 'three';
import { SIDEWALK_Y } from '../town.js';
import { flatShapes, wallGeometry, bandGeometry } from '../lib/rings.js';

export function build(ctx) {
  const T = ctx.town;
  const R = ctx.rng('kv-roads');
  const merge = (geos) => { const g = ctx.geo.mergeGeometries(geos, false); geos.forEach(x => x.dispose()); return g; };
  const add = (geo, mat, name, { shadow = false, receive = true } = {}) => {
    const m = new THREE.Mesh(geo, mat); m.name = name; m.castShadow = shadow; m.receiveShadow = receive;
    ctx.addStatic(m); return m;
  };
  const toXZ = (shape) => ({ outer: shape.outer.map(T.P), holes: shape.holes.map(h => h.map(T.P)) });
  const roadShapes = T.data.surfaces.road.map(toXZ);
  const walkShapes = T.data.surfaces.sidewalk.map(toXZ);
  const ringsOf = (shapes) => shapes.flatMap(s => [{ ring: s.outer, hole: false }, ...s.holes.map(h => ({ ring: h, hole: true }))]);

  // ------------------------------------------------------------------ textures
  const asphaltTex = ctx.tex.draw(512, 512, (g, w, h) => {
    g.fillStyle = '#ffffff'; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 40; i++) {                       // soft wash
      g.fillStyle = `rgba(${R() < 0.5 ? '90,92,100' : '255,255,255'},${0.04 + R() * 0.05})`;
      g.beginPath(); g.ellipse(R() * w, R() * h, 30 + R() * 80, 20 + R() * 50, R() * 3, 0, 7); g.fill();
    }
    for (let i = 0; i < 5000; i++) {                     // aggregate speckle
      const v = R() < 0.5 ? 70 + R() * 40 : 225 + R() * 30;
      g.fillStyle = `rgba(${v},${v},${v + 4},${0.18 + R() * 0.2})`;
      g.fillRect(R() * w, R() * h, 1 + R() * 1.6, 1 + R() * 1.6);
    }
    g.strokeStyle = 'rgba(60,60,72,0.28)'; g.lineWidth = 1.2;  // a few hairline cracks
    for (let i = 0; i < 6; i++) {
      let x = R() * w, y = R() * h; g.beginPath(); g.moveTo(x, y);
      for (let k = 0; k < 6; k++) { x += (R() - 0.5) * 40; y += (R() - 0.5) * 40; g.lineTo(x, y); }
      g.stroke();
    }
  }, { key: 'kv_asphalt', repeat: [1, 1] });
  const paveTex = ctx.tex.draw(512, 512, (g, w, h) => {  // 30 cm square pavers, 10 x 10 per 3 m tile
    const n = 10, s = w / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = 236 + R() * 19; g.fillStyle = `rgb(${v},${v - 2},${v - 6})`; g.fillRect(i * s, j * s, s, s);
    }
    g.strokeStyle = 'rgba(120,112,104,0.55)'; g.lineWidth = 2;
    for (let i = 0; i <= n; i++) { g.beginPath(); g.moveTo(i * s, 0); g.lineTo(i * s, h); g.stroke(); g.beginPath(); g.moveTo(0, i * s); g.lineTo(w, i * s); g.stroke(); }
    for (let i = 0; i < 1500; i++) { g.fillStyle = `rgba(120,110,100,${0.05 + R() * 0.08})`; g.fillRect(R() * w, R() * h, 1.5, 1.5); }
  }, { key: 'kv_pavers', repeat: [1, 1] });

  const mAsphalt = ctx.mat.toon('#6f7176', { map: asphaltTex, paint: 0.06 });
  const mGutter = ctx.mat.toon('#b9b8b1', { paint: 0.05, polygonOffset: -1 });
  const mPave = ctx.mat.toon('#d5d0c5', { map: paveTex, paint: 0.05 });
  const mCurb = ctx.mat.toon('#c7c5bd', { paint: 0.04 });
  const mPaint = ctx.mat.decal('#f2f0ea', { paint: 0.03 });

  // ------------------------------------------------------------------ road surface + gutters
  add(merge(flatShapes(roadShapes, 0, 8)), mAsphalt, 'road');
  add(bandGeometry(ringsOf(roadShapes), 0.42, 0.004, 'in', 1), mGutter, 'gutter');

  // ------------------------------------------------------------------ sidewalks (raised slab + kerb)
  const walkRings = ringsOf(walkShapes);
  add(merge(flatShapes(walkShapes, SIDEWALK_Y, 3)), mPave, 'sidewalk');
  add(wallGeometry(walkRings, -0.02, SIDEWALK_Y, 1), mCurb, 'kerb-face');
  add(bandGeometry(walkRings, 0.2, SIDEWALK_Y + 0.004, 'in', 1), mCurb, 'kerb-top');

  // ------------------------------------------------------------------ paint: lane lines, stop lines, zebras
  const polys = [...T.data.markings, ...T.data.crosswalks].map(p => ({ outer: p.map(T.P), holes: [] }));
  const paint = add(merge(flatShapes(polys, 0.008, 1)), mPaint, 'paint', { receive: true });
  ctx.noOutline(paint);

  ctx.services.roads = { roadShapes, walkShapes };
}
