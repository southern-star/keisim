// ?gallery=cars — the detailed cars side by side: today's (src/world/cars.js, ?cardetail=1) on the left of each pair,
// the lofted cars of the redesign (src/world/cars2.js, ?cardetail=2) on its right, same size and colour.
//   ?only=compact,sedan   only these body types      ?level=2   only the redesign (1: only today's)
//   ?n=3                  cars per type (sizes and colours as KeiSim draws them, by a fixed seed)
//   ?brake=1              brake lamps on            ?color=236,236,232   every car in this colour (RGB 0..255)
//   ?glass=solid,tint,clear   the redesign with each of these glasses side by side (src/world/carglass.js); ?lines=0: the
//                         see-through glass stays in the outline pass (no lines of the interior)
//   ?shine=0,1            the redesign without and with the highlight trial, side by side
//   ?wheels=1 / ?paint=1 / ?driver=0   one car per type with each wheel / each roof and pillar scheme; nobody at the wheel
//   ?types=1              every body type of each size class side by side (src/world/cars2.js SUBTYPES); ?work=taxi|patrol|school|post,
//                         ?mark=beginner|elder, ?carrier=rack|ladder, ?panel=1, ?load=0|1|2 force those on every car
//   ?faces=1              one car per type with each of its faces in turn (front and rear lists of src/world/carface.js)
import * as THREE from 'three';
import { carType } from './world/cars.js';
import { faceCount } from './world/carface.js';
import { SUBTYPES } from './world/cars2.js';

function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
// keisim/traffic.py: CAR_COLORS and _sample_dims (the classes and their shares)
const CAR_COLORS = [[236, 236, 232], [200, 204, 208], [40, 42, 48], [120, 124, 130], [176, 36, 40], [36, 70, 140], [228, 200, 60], [60, 120, 80],
  [232, 232, 232], [90, 60, 44], [150, 190, 214]];
function sample(r) {
  const u = r(), f = (lo, hi) => lo + (hi - lo) * r();
  const dims = u < 0.62 ? [f(4.0, 4.8), f(1.68, 1.85), f(1.4, 1.6), 0] : u < 0.78 ? [f(3.35, 3.45), 1.48, f(1.6, 1.8), 0]
    : u < 0.92 ? [f(4.6, 5.0), f(1.8, 1.95), f(1.75, 2.0), 0] : [f(6.0, 7.6), f(2.0, 2.25), f(2.6, 3.0), 1];
  const c = CAR_COLORS[Math.floor(r() * CAR_COLORS.length)], k = 0.85 + 0.3 * r();
  return { dims, rgb: c.map((v) => Math.max(0, Math.min(255, Math.round(v * k + (r() * 16 - 8))))) };
}
const TYPES = ['compact', 'keihatch', 'keiwagon', 'keitruck', 'sedan', 'suv', 'minivan', 'truck'];

export async function setup({ ctx, scene, params }) {
  ctx.carDetail = 2; ctx.mergeActors = true;
  const mod = await import('./world/actors.js');
  mod.build(ctx);
  const actors = ctx.services.actors;
  const only = params.get('only') ? params.get('only').split(',') : TYPES;
  const faces = params.get('faces') === '1', wheels = params.get('wheels') === '1', paint = params.get('paint') === '1';
  const sets = wheels ? ['cap', 'spoke5', 'spoke10', 'dish', 'black'].map((wheel) => ({ wheel }))
    : paint ? [{ roof: 'body', pillar: 'body' }, { roof: 'plateW' }, { roof: 'trim' }, { roof: 'body', pillar: 'trim' }] : null;
  const types = params.get('types') === '1';
  const per = sets ? sets.length : faces || types ? 4 : Number(params.get('n') || 3), level = faces || sets || types ? '2' : params.get('level');
  const brake = params.get('brake') === '1';
  const color = params.get('color') ? params.get('color').split(',').map(Number) : null;
  // the first `per` cars of every type from one stream of KeiSim-like cars
  const r = rng(Number(params.get('seed') || 7)), byType = new Map(TYPES.map((t) => [t, []]));
  for (let id = 0; id < 4000 && [...byType.values()].some((a) => a.length < per); id++) {
    const { dims: [L, W, H, kind], rgb } = sample(r), type = carType(kind, L, W, H, rgb);
    if (byType.get(type).length < per) byType.get(type).push({ id, kind, L, W, H, rgb, type });
  }
  if (sets) for (const [type, a] of byType) byType.set(type, a[0] ? sets.map((o) => ({ ...a[0], opt: o })) : []);
  if (types) for (const [type, a] of byType) byType.set(type, a[0] ? SUBTYPES[type].map(([t]) => ({ ...a[0], opt: { type: t } })) : []);
  if (faces) for (const [type, a] of byType) {                       // the first car of the type, once per face
    const n = faceCount(type), c = a[0];
    byType.set(type, c ? Array.from({ length: Math.max(n.front, n.rear) }, (_, i) => ({ ...c, opt: { face: { front: i, rear: i } } })) : []);
  }
  const root = new THREE.Group(); root.name = 'gallery cars'; root.userData.sem = 11;
  scene.add(root);
  const cars = [];
  let row = 0;
  for (const type of TYPES.filter((t) => only.includes(t))) {
    byType.get(type).forEach((c, i) => {
      const glasses = params.get('glass') ? params.get('glass').split(',') : null, shines = params.get('shine') ? params.get('shine').split(',') : null;
      const cols = glasses || shines;
      const levels = cols ? cols.map(() => 2) : level ? [Number(level)] : [1, 2];
      levels.forEach((lv, j) => {
        const opt = { ...(c.opt || {}) };
        if (glasses) { opt.glass = glasses[j]; opt.glassLines = params.get('lines') !== '0'; }
        if (shines) opt.shine = shines[j] === '1';
        if (params.get('driver') === '0') opt.driver = false;
        for (const k of ['work', 'mark', 'carrier']) if (params.get(k)) opt[k] = params.get(k);
        if (params.get('panel')) opt.panel = params.get('panel') === '1';
        if (params.get('load')) opt.load = Number(params.get('load'));
        const m = actors.detailModel(lv, c.kind, c.L, c.W, c.H, color || c.rgb, c.id, opt);
        const x = (i - (per - 1) / 2) * (levels.length > 1 ? 3.7 * levels.length : 3.6) + (levels.length > 1 ? (j - (levels.length - 1) / 2) * 3.4 : 0);
        m.g.position.set(x, 0, -row * 9);
        if (brake) actors.setBrake(m, true);
        root.add(m.g);
        cars.push({ ...c, level: lv, g: m.g, meshes: m.g.children.length });
      });
    });
    row++;
  }
  return {
    title: '車', cars,
    caption: `各組の左が今の詳細モデル（cardetail=1）、右が作り直し（cardetail=2）。列は ${TYPES.filter((t) => only.includes(t)).join(' / ')}。`,
    orbit: { az: 35, el: 14, dist: 6 + 5 * per + 4 * row, y: 0.7, z: -(row - 1) * 4.5 },
    update() {},
  };
}
