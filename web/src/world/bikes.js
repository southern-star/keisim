// 自転車 — KeiSim's bicycles and their riders (keisim/bicycles.py), posed every frame from the simulator state:
// lean, steering, turning wheels and cranks, legs that follow the pedals (the left foot goes to the ground at a stop),
// hands on the bars, the look back over the shoulder. Three kinds: 0 the city bike (ママチャリ: step-through frame,
// fenders, chain case, carrier, front basket, upright rider), 1 the sport bike (diamond frame, drop bars, helmet,
// low rider), 2 the slow rider (the city bike, a hat). keisim/render/meshes.py (bike_faces) draws the same build
// with far fewer parts; the numbers that place the rider (saddle, grips, pedals, lean) are the same in both.
// Geometry is written in the bike's frame (x forward, y left, z up, origin on the ground under its middle) and
// mapped to the model's (+Z forward, +X left, +Y up) by V(). One vertex-coloured mesh per rigid part.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export const BIKE_FINE_DIST = 45;        // m: spokes and basket wires are drawn closer than this to the camera

const V = (x, y, z) => new THREE.Vector3(y, z, x);
const UP = new THREE.Vector3(0, 1, 0);
const BB = [-0.06, 0.29], CRANK = 0.17, AXLE = 0.54, PEDAL_Y = 0.12;
const PIVOT = [0.44, 0.64], AXIS = [-0.08, 0.36];       // the steering axis: a point of it and its direction (x, z)
const HAIR = ['#2b2630', '#3d2f2a', '#5a4130', '#1f1d24', '#6b5845'];
const HELMET = ['#eceef0', '#2a2c33', '#c8342f', '#e8c62a'];
const BAG = ['#3f5f8a', '#7a4a3a', '#4f6b4a', '#8a8f98'];
const C = { tyre: '#26262b', dark: '#2c2d33', metal: '#b9bec6', steel: '#8d9199', red: '#c8282d', amber: '#f0a020',
  lens: '#fff2c4', hat: '#d9c9a3', grey: '#b9b9bd', white: '#eceae4', sole: '#e8e6e0' };

// ---- geometry helpers (bike frame in, model frame out)
function tube(a, b, r0, r1 = r0, seg = 8, open = false) {
  const d = b.clone().sub(a), len = d.length();
  const g = new THREE.CylinderGeometry(r1, r0, len, seg, 1, open);
  g.translate(0, len / 2, 0);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(UP, d.normalize()));
  return g.translate(a.x, a.y, a.z);
}
const bend = (pts, r, seg = 8, n = pts.length * 5) => new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), n, r, seg, false);
const ball = (c, r, sx = 1, sy = 1, sz = 1, w = 12, h = 9) => new THREE.SphereGeometry(r, w, h).scale(sy, sz, sx).translate(c.x, c.y, c.z);
const slab = (c, lx, wy, hz, r = 0) => (r > 0 ? new RoundedBoxGeometry(wy, hz, lx, 2, Math.min(r, lx / 2, wy / 2, hz / 2))
  : new THREE.BoxGeometry(wy, hz, lx)).translate(c.x, c.y, c.z);
const ring = (c, R, r, radial = 8, n = 28) => new THREE.TorusGeometry(R, r, radial, n).rotateY(Math.PI / 2).translate(c.x, c.y, c.z);
const disc = (c, r, w, seg = 16) => new THREE.CylinderGeometry(r, r, w, seg).rotateZ(Math.PI / 2).translate(c.x, c.y, c.z);
/** A strip swept around the wheel axis through c (bike frame): profile [[y, radius], ...], angles a0..a1 with
 *  0 = forward, pi/2 = top, pi = back. Both sides are drawn. */
function sweep(c, profile, a0, a1, n = 14) {
  const pos = [], idx = [], m = profile.length;
  for (let i = 0; i <= n; i++) {
    const a = a0 + (a1 - a0) * i / n, ca = Math.cos(a), sa = Math.sin(a);
    for (const [y, r] of profile) { const p = V(c[0] + r * ca, c[1] + y, c[2] + r * sa); pos.push(p.x, p.y, p.z); }
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < m - 1; j++) {
    const a = i * m + j, b = a + m;
    idx.push(a, b, a + 1, a + 1, b, b + 1, a, a + 1, b, a + 1, b + 1, b);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  const out = g.toNonIndexed(); out.computeVertexNormals();
  return out;
}
/** A limb along +Y from the origin: a tapered tube with rounded joints at both ends. */
function limb(len, r0, r1, seg = 8) {
  const g = [new THREE.CylinderGeometry(r1, r0, len, seg, 1, true).translate(0, len / 2, 0),
    new THREE.SphereGeometry(r0, seg, 6), new THREE.SphereGeometry(r1, seg, 6).translate(0, len, 0)];
  return g;
}

/** Collects coloured geometries per part; build(name) merges a part into one mesh. */
function collector(mat) {
  const parts = new Map(), col = new THREE.Color();
  function add(name, color, ...geos) {
    if (!parts.has(name)) parts.set(name, []);
    for (const g0 of geos.flat()) {
      const g = g0.index ? g0.toNonIndexed() : g0;
      for (const k of Object.keys(g.attributes)) if (k !== 'position' && k !== 'normal') g.deleteAttribute(k);
      col.set(color);
      const n = g.attributes.position.count, c = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) { c[3 * i] = col.r; c[3 * i + 1] = col.g; c[3 * i + 2] = col.b; }
      g.setAttribute('color', new THREE.BufferAttribute(c, 3));
      g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(n * 2), 2));
      parts.get(name).push(g);
    }
  }
  function build(name, { shadow = true } = {}) {
    const geos = parts.get(name);
    if (!geos || !geos.length) return null;
    const mesh = new THREE.Mesh(mergeGeometries(geos, false), mat);
    for (const g of geos) g.dispose();
    mesh.castShadow = shadow; mesh.receiveShadow = true; mesh.name = name;
    return mesh;
  }
  return { add, build };
}

/** Two-bone reach from a to the target: the middle joint (bending towards pole) and the end the limb can reach. */
function twoBone(a, tgt, l1, l2, pole, mid, end) {
  end.copy(tgt).sub(a);
  const dist = Math.max(end.length(), 1e-6);
  end.divideScalar(dist);                                  // e
  const dc = Math.min(Math.max(dist, 0.05), 0.995 * (l1 + l2));
  const along = (l1 * l1 - l2 * l2 + dc * dc) / (2 * dc);
  const out = Math.sqrt(Math.max(l1 * l1 - along * along, 0));
  mid.copy(pole).addScaledVector(end, -pole.dot(end)).normalize().multiplyScalar(out).addScaledVector(end, along).add(a);
  end.multiplyScalar(dc).add(a);
}
const _d = new THREE.Vector3();
function setBone(o, a, b) { o.position.copy(a); o.quaternion.setFromUnitVectors(UP, _d.copy(b).sub(a).normalize()); }

/** The rider's measures (the same numbers as keisim/render/meshes.py bike_faces). */
export function riderFit(kind, h) {
  const sport = kind === 1;
  const leg = 0.245 * h, ext = sport ? 0.93 : kind === 0 ? 0.87 : 0.85;
  const hipD = ext * 2 * leg - 0.08;
  const hx = BB[0] - 0.309 * hipD, hz = BB[1] + 0.951 * hipD;
  return { sport, leg, hipD, hx, hz, sz: hz - 0.10, rw: sport ? 0.34 : 0.33, lt: 0.30 * h, upper: 0.172 * h, fore: 0.165 * h,
    lean0: sport ? 0.62 : kind === 0 ? 0.24 : 0.12, grip: sport ? [0.565, 0.21, 0.945] : [0.24, 0.28, 1.10] };
}

function addWheel(P, name, fit, sport) {
  const rw = fit.rw, tr = sport ? 0.014 : 0.022, o = V(0, 0, 0);
  P.add(name, C.tyre, ring(o, rw - tr, tr, 8, 32));
  P.add(name, sport ? C.dark : C.metal, ring(o, rw - 2 * tr - 0.007, 0.009, 6, 32));
  P.add(name, C.metal, tube(V(0, -0.05, 0), V(0, 0.05, 0), 0.02, 0.02, 10));
  const rr = rw - 2 * tr - 0.01, ns = sport ? 12 : 14;
  for (let i = 0; i < ns; i++) {
    const a = (i + 0.5) * 2 * Math.PI / ns, s = i % 2 ? 1 : -1, b = a + s * 0.35;
    P.add(name + 'Fine', C.steel, tube(V(0.02 * Math.cos(b), s * 0.032, 0.02 * Math.sin(b)), V(rr * Math.cos(a), 0, rr * Math.sin(a)), 0.0035, 0.0035, 4, true));
  }
  if (!sport) for (const a of [0.4, 0.4 + Math.PI]) {
    const p = V(0.62 * rw * Math.cos(a), 0, 0.62 * rw * Math.sin(a));
    P.add(name, C.amber, slab(V(0, 0, 0), 0.085, 0.014, 0.03).rotateX(-a).translate(p.x, p.y, p.z));
  }
}


const ax = (t) => [PIVOT[0] + AXIS[0] * t, PIVOT[1] + AXIS[1] * t];          // a point of the steering axis
const F = (x, y, z) => V(x - PIVOT[0], y, z - PIVOT[1]);                     // front assembly: relative to its pivot
const seatAt = (d) => [BB[0] - 0.309 * d, BB[1] + 0.951 * d];                // a point of the seat tube's line

/** The parts that do not move against the frame: 'frame'. */
function addFrame(P, fit, kind, id, col) {
  const { sport, rw, hx, sz } = fit, [bx, bz] = BB;
  const dc = (sz - 0.05 - bz) / 0.951;                                       // the saddle clamp, along the seat tube
  const st = Math.max(0.3, Math.min(sport ? 0.50 : 0.44, dc - (sport ? 0.08 : 0.06)));
  const sp = (d, y = 0) => { const [x, z] = seatAt(d); return V(x, y, z); };
  const f = 'frame';
  P.add(f, col, tube(sp(0), sp(st), sport ? 0.016 : 0.018));
  P.add(f, sport ? C.dark : C.metal, tube(sp(st - 0.02), sp(dc), 0.012));
  P.add(f, C.dark, tube(V(bx, -0.045, bz), V(bx, 0.045, bz), 0.026, 0.026, 10));          // bottom bracket shell
  for (const s of [-1, 1]) {
    P.add(f, col, tube(V(bx, s * 0.03, bz), V(-AXLE, s * 0.062, rw), sport ? 0.009 : 0.011));        // chain stays
    P.add(f, col, tube(sp(st - 0.04, s * 0.02), V(-AXLE, s * 0.062, rw), sport ? 0.007 : 0.009));     // seat stays
    P.add(f, C.steel, ball(V(-AXLE, s * 0.062, rw), 0.016));
  }
  if (sport) {
    const [h0x, h0z] = ax(0.22), [h1x, h1z] = ax(0.58);
    P.add(f, col, tube(V(h0x, 0, h0z), V(h1x, 0, h1z), 0.022, 0.022, 10));
    P.add(f, col, tube(V(0.398, 0, 0.83), sp(st - 0.02), 0.016));                         // top tube
    P.add(f, col, tube(V(0.418, 0, 0.735), V(bx, 0, bz), 0.021));                         // down tube
    P.add(f, C.dark, ball(V(hx - 0.04, 0, sz - 0.025), 1, 0.14, 0.065, 0.028), ball(V(hx + 0.09, 0, sz - 0.028), 1, 0.07, 0.03, 0.024));
    P.add(f, C.red, slab(V(hx - 0.16, 0, sz - 0.07), 0.025, 0.045, 0.05, 0.008));          // rear light
    P.add(f, C.steel, disc(V(-AXLE, -0.04, rw), 0.05, 0.028));                            // sprockets
    P.add(f, C.dark, tube(V(bx, -0.05, bz + 0.09), V(-AXLE, -0.045, rw + 0.045), 0.004, 0.004, 4),
      tube(V(bx, -0.05, bz - 0.09), V(-AXLE + 0.01, -0.045, 0.235), 0.004, 0.004, 4),
      slab(V(-AXLE + 0.005, -0.052, 0.27), 0.03, 0.015, 0.11, 0.005));                    // chain, derailleur
    P.add(f, id % 2 ? C.white : col, tube(V(0.186, 0, 0.589), V(0.054, 0, 0.467), 0.034, 0.034, 10),
      tube(V(0.186, 0, 0.589), V(0.208, 0, 0.609), 0.034, 0.016, 10));                    // bottle
    return;
  }
  const [h0x, h0z] = ax(0.2), [h1x, h1z] = ax(0.62);
  P.add(f, col, tube(V(h0x, 0, h0z), V(h1x, 0, h1z), 0.024, 0.024, 10));
  P.add(f, col, bend([V(0.42, 0, 0.715), V(0.20, 0, 0.505), V(0.05, 0, 0.36), V(bx, 0, bz)], 0.023));     // main tube
  const [jx, jz] = seatAt(0.21);
  P.add(f, col, bend([V(0.395, 0, 0.84), V(0.25, 0, 0.66), V(0.05, 0, 0.51), V(jx, 0, jz)], 0.016));       // upper tube
  P.add(f, C.dark, ball(V(hx - 0.06, 0, sz - 0.03), 1, 0.13, 0.11, 0.04), ball(V(hx + 0.075, 0, sz - 0.03), 1, 0.09, 0.05, 0.034));
  const R = rw + 0.028;
  P.add(f, C.metal, sweep([-AXLE, 0, rw], [[-0.032, R - 0.012], [-0.02, R], [0.02, R], [0.032, R - 0.012]], 0.25, 3.3, 16));
  for (const s of [-1, 1]) {
    P.add(f, C.steel, tube(V(-AXLE, s * 0.05, rw), V(-AXLE + R * Math.cos(2.75), s * 0.03, rw + R * Math.sin(2.75)), 0.004, 0.004, 4));
    // the carrier: side rails, struts to the axle and to the seat stays
    P.add(f, C.steel, tube(V(-0.40, s * 0.07, 0.72), V(-0.86, s * 0.07, 0.72), 0.007, 0.007, 6),
      tube(V(-AXLE, s * 0.075, rw), V(-0.50, s * 0.07, 0.72), 0.006, 0.006, 5), tube(V(-AXLE, s * 0.075, rw), V(-0.78, s * 0.07, 0.72), 0.006, 0.006, 5),
      tube(V(-0.40, s * 0.07, 0.72), sp(st - 0.05, s * 0.02), 0.006, 0.006, 5));
    P.add(f, C.steel, tube(V(-AXLE, s * 0.08, rw), V(-0.80, s * 0.10, rw - 0.05), 0.007, 0.007, 5));     // stand, folded up
  }
  for (const x of [-0.40, -0.55, -0.70, -0.86]) P.add(f, C.steel, tube(V(x, -0.07, 0.72), V(x, 0.07, 0.72), 0.006, 0.006, 5));
  P.add(f, C.steel, tube(V(-0.80, -0.10, rw - 0.05), V(-0.80, 0.10, rw - 0.05), 0.007, 0.007, 5));
  P.add(f, C.red, slab(V(-0.878, 0, 0.69), 0.022, 0.07, 0.045, 0.006));                    // reflector
  // chain case on the right
  P.add(f, col, slab(V(-0.30, -0.066, 0.335), 0.60, 0.02, 0.115, 0.009), disc(V(bx, -0.066, bz), 0.105, 0.02, 18));
}

/** The steered front end, relative to the pivot: 'front' (and 'frontFine': the basket's wires). */
function addFront(P, fit, kind, id, col, basket) {
  const { sport, rw } = fit, f = 'front';
  const A = (t, y = 0) => { const [x, z] = ax(t); return F(x, y, z); };
  if (sport) {
    for (const s of [-1, 1]) P.add(f, col, bend([F(0.425, s * 0.045, 0.705), F(0.46, s * 0.05, 0.52), F(AXLE, s * 0.05, rw)], 0.011));
    P.add(f, col, tube(F(0.425, -0.052, 0.705), F(0.425, 0.052, 0.705), 0.014));
    P.add(f, C.dark, tube(A(0.58), A(0.67), 0.018, 0.018, 10), tube(A(0.63), F(0.47, 0, 0.90), 0.014));
    const side = (s) => [F(0.45, s * 0.21, 0.755), F(0.53, s * 0.21, 0.757), F(0.59, s * 0.21, 0.80), F(0.60, s * 0.21, 0.86),
      F(0.575, s * 0.21, 0.90), F(0.53, s * 0.205, 0.905), F(0.475, s * 0.17, 0.90), F(0.47, s * 0.08, 0.90)];
    P.add(f, C.dark, bend([...side(1), ...side(-1).reverse()], 0.012, 8, 64));
    for (const s of [-1, 1]) {
      P.add(f, C.dark, ball(F(0.592, s * 0.21, 0.925), 1, 0.045, 0.018, 0.03), ball(F(0.45, s * 0.21, 0.755), 0.012));
      P.add(f, C.metal, tube(F(0.625, s * 0.21, 0.93), F(0.615, s * 0.21, 0.81), 0.006, 0.005, 5));
    }
    return;
  }
  for (const s of [-1, 1]) P.add(f, col, bend([F(0.428, s * 0.05, 0.694), F(0.455, s * 0.055, 0.52), F(AXLE, s * 0.055, rw)], 0.012));
  P.add(f, col, tube(F(0.428, -0.058, 0.694), F(0.428, 0.058, 0.694), 0.015));
  P.add(f, C.metal, tube(A(0.62), A(1.14), 0.013), tube(A(1.14), F(0.37, 0, 1.06), 0.013), ball(A(1.14), 0.014));
  const side = (s) => [F(0.18, s * 0.285, 1.078), F(0.27, s * 0.28, 1.08), F(0.33, s * 0.265, 1.08), F(0.375, s * 0.20, 1.078), F(0.385, s * 0.09, 1.068)];
  P.add(f, C.metal, bend([...side(1), F(0.385, 0, 1.064), ...side(-1).reverse()], 0.011, 8, 56));
  for (const s of [-1, 1]) {
    P.add(f, C.dark, tube(F(0.295, s * 0.276, 1.08), F(0.18, s * 0.285, 1.078), 0.017), ball(F(0.18, s * 0.285, 1.078), 0.018));
    P.add(f, C.metal, tube(F(0.31, s * 0.262, 1.066), F(0.215, s * 0.245, 1.056), 0.005, 0.004, 5));
  }
  P.add(f, C.metal, ball(F(0.36, 0.17, 1.10), 0.022));                                    // bell
  const R = rw + 0.028, c = [AXLE - PIVOT[0], 0, rw - PIVOT[1]];
  P.add(f, C.metal, sweep(c, [[-0.032, R - 0.012], [-0.02, R], [0.02, R], [0.032, R - 0.012]], 0.6, 3.5, 16));
  for (const s of [-1, 1]) P.add(f, C.steel, tube(F(AXLE, s * 0.05, rw), F(AXLE + R * Math.cos(2.9), s * 0.03, rw + R * Math.sin(2.9)), 0.004, 0.004, 4));
  P.add(f, C.metal, tube(F(0.585, 0, 0.722), F(0.635, 0, 0.722), 0.028, 0.03, 10));       // lamp
  P.add(f, C.lens, tube(F(0.635, 0, 0.722), F(0.642, 0, 0.722), 0.025, 0.022, 10));
  P.add(f, C.steel, tube(A(0.3), F(0.59, 0, 0.722), 0.006, 0.006, 5));
  if (!basket) return;
  const wire = id % 4 === 2 ? C.dark : C.white;
  const loop = (x0, x1, w, z) => [F(x0, -w, z), F(x1, -w, z), F(x1, w, z), F(x0, w, z)];
  const top = loop(0.50, 0.78, 0.17, 0.98), bot = loop(0.52, 0.75, 0.14, 0.76);
  const mid = top.map((p, i) => p.clone().lerp(bot[i], 0.5));
  for (const [L, r] of [[top, 0.007], [mid, 0.004], [bot, 0.005]]) for (let i = 0; i < 4; i++) {
    P.add(L === mid ? 'frontFine' : f, wire, tube(L[i], L[(i + 1) % 4], r, r, 5));
    if (L !== mid) P.add(f, wire, ball(L[i], r, 1, 1, 1, 6, 4));
  }
  for (let i = 0; i < 4; i++) {
    const n = i % 2 ? 7 : 6;                                // ribs: the long sides are the front and the back
    for (let k = 0; k < n; k++) {
      const t = (k + (i % 2 ? 0.5 : 0)) / n + (i % 2 ? 0 : 0);
      P.add('frontFine', wire, tube(bot[i].clone().lerp(bot[(i + 1) % 4], t), top[i].clone().lerp(top[(i + 1) % 4], t), 0.0035, 0.0035, 4, true));
    }
  }
  for (let k = 1; k < 6; k++) P.add('frontFine', wire, tube(bot[0].clone().lerp(bot[1], k / 6), bot[3].clone().lerp(bot[2], k / 6), 0.0035, 0.0035, 4, true));
  P.add(f, C.steel, tube(A(1.0), F(0.50, 0, 0.93), 0.006, 0.006, 5));
  for (const s of [-1, 1]) P.add(f, C.steel, tube(F(AXLE, s * 0.062, rw), F(0.60, s * 0.135, 0.76), 0.005, 0.005, 5));
  if (id % 3 === 1) P.add(f, BAG[id % BAG.length], slab(F(0.64, 0, 0.865), 0.19, 0.23, 0.2, 0.04));
}

/** The cranks ('crank', about the crank axle) and a pedal ('pedalR' / 'pedalL', about its own axle). */
function addCranks(P, fit) {
  P.add('crank', C.metal, tube(V(0, -0.072, 0), V(0, 0.072, 0), 0.013), tube(V(0, -0.072, 0), V(0, -0.078, CRANK), 0.012, 0.009),
    tube(V(0, 0.072, 0), V(0, 0.078, -CRANK), 0.012, 0.009));
  P.add('crank', fit.sport ? C.dark : C.steel, disc(V(0, -0.05, 0), fit.sport ? 0.095 : 0.08, 0.006, 20));
  for (const n of ['pedalR', 'pedalL']) {
    P.add(n, C.dark, slab(V(0, 0, -0.011), 0.095, 0.085, 0.022, 0.006));
    P.add(n, C.amber, slab(V(0.047, 0, -0.011), 0.005, 0.05, 0.014), slab(V(-0.047, 0, -0.011), 0.005, 0.05, 0.014));
  }
}

/** The rider's parts: 'pelvis', 'torso', 'head', the arms and legs as bones along +Y, the shoes about the ankle. */
function addRider(P, fit, kind, id, shirt, pants, skin) {
  const { sport, lt, leg, upper, fore, hx, hz } = fit, s = fit.lt / 0.51;
  const sleeves = !sport && (kind === 2 || id % 3 !== 0), longPants = !sport && (kind === 2 || id % 5 !== 0);
  const ell = (rx, ry, rz, x, y, z, w = 12, hh = 9) => new THREE.SphereGeometry(1, w, hh).scale(rx, ry, rz).translate(x, y, z);
  P.add('pelvis', pants, ball(V(hx, 0, hz + 0.01), 1, 0.105 * s, 0.155 * s, 0.10 * s));
  P.add('torso', shirt, new THREE.CylinderGeometry(0.158 * s, 0.135 * s, lt - 0.07, 12, 1, true).scale(1, 1, 0.62).translate(0, (lt - 0.07) / 2, 0),
    ell(0.19 * s, 0.078 * s, 0.105 * s, 0, lt - 0.062, 0), ell(0.137 * s, 0.07 * s, 0.087 * s, 0, 0.005, 0));
  P.add('torso', skin, new THREE.CylinderGeometry(0.042 * s, 0.045 * s, 0.1, 8).translate(0, lt + 0.01, 0.005));
  if (kind === 0 && id % 4 === 1) P.add('torso', BAG[(id >> 2) % BAG.length], new RoundedBoxGeometry(0.26 * s, 0.34 * s, 0.13 * s, 2, 0.04).translate(0, lt * 0.52, -0.155 * s));
  P.add('head', skin, ell(0.103 * s, 0.114 * s, 0.108 * s, 0, 0.10 * s, 0.015 * s, 14, 10));
  if (sport) {
    P.add('head', HAIR[id % HAIR.length], ell(0.104 * s, 0.1 * s, 0.1 * s, 0, 0.105 * s, -0.02 * s));
    P.add('head', HELMET[id % HELMET.length], ell(0.118 * s, 0.085 * s, 0.152 * s, 0, 0.168 * s, -0.008 * s, 14, 10));
    P.add('head', C.dark, new RoundedBoxGeometry(0.168 * s, 0.036 * s, 0.05 * s, 2, 0.012).translate(0, 0.108 * s, 0.088 * s));
  } else if (kind === 2) {
    P.add('head', C.grey, ell(0.11 * s, 0.11 * s, 0.108 * s, 0, 0.112 * s, -0.014 * s));
    P.add('head', C.hat, new THREE.CylinderGeometry(0.168 * s, 0.172 * s, 0.012, 18).translate(0, 0.168 * s, 0.01 * s),
      new THREE.CylinderGeometry(0.098 * s, 0.108 * s, 0.075 * s, 14).translate(0, 0.205 * s, 0));
  } else {
    P.add('head', HAIR[id % HAIR.length], ell(0.112 * s, 0.112 * s, 0.112 * s, 0, 0.118 * s, -0.012 * s));
    if (id % 3 === 0) P.add('head', HAIR[id % HAIR.length], ell(0.095 * s, 0.12 * s, 0.062 * s, 0, 0.03 * s, -0.082 * s));
  }
  for (const n of ['L', 'R']) {
    P.add('armU' + n, shirt, limb(upper, 0.05 * s, 0.04 * s));
    P.add('armF' + n, sleeves ? shirt : skin, limb(fore, 0.04 * s, 0.032 * s));
    P.add('armF' + n, skin, new THREE.SphereGeometry(0.042 * s, 8, 6).translate(0, fore + 0.02, 0));
    P.add('thigh' + n, pants, limb(leg, 0.075 * s, 0.056 * s));
    P.add('shin' + n, longPants ? pants : skin, limb(leg, 0.054 * s, 0.038 * s));
    P.add('shoe' + n, id % 2 ? C.sole : C.dark, new RoundedBoxGeometry(0.092 * s, 0.065, 0.25 * s, 2, 0.028).translate(0, -0.052, 0.055));
  }
}

const _q = new THREE.Quaternion(), _e = new THREE.Euler();
/**
 * One bicycle with its rider. cols: [frame, shirt, pants, skin] as THREE.Color; mat: a white vertex-coloured material.
 * Returns { g, fine, pose }: g faces local +Z with its origin on the ground under the middle of the bike; fine: the
 * meshes of thin wires (the caller hides them far from the camera); pose([roll, steer, crank, wheel, foot, look,
 * effort]) as Bicycles.scene gives them.
 */
export function makeBike(ctx, mat, kind, h, cols, id) {
  const fit = riderFit(kind, h), { sport, rw, hx, hz, leg, upper, fore, lt } = fit;
  const P = collector(mat), fine = [];
  const basket = !sport && (kind === 2 || id % 5 !== 2);
  addWheel(P, 'wheel', fit, sport);
  addFrame(P, fit, kind, id, cols[0]);
  addFront(P, fit, kind, id, cols[0], basket);
  addCranks(P, fit);
  addRider(P, fit, kind, id, cols[1], cols[2], cols[3]);

  const grp = (parent, pos, sem) => { const o = new THREE.Group(); if (pos) o.position.copy(pos); if (sem !== undefined) o.userData.sem = sem; parent.add(o); return o; };
  const put = (parent, name, opts) => { const m = P.build(name, opts); if (m) parent.add(m); return m; };
  const thin = (parent, name) => { const m = put(parent, name, { shadow: false }); if (m) { ctx.noOutline(m); fine.push(m); } };
  const g = new THREE.Group();
  const lean = grp(g);
  const bike = grp(lean, null, 11), rider = grp(lean, null, 12);
  put(bike, 'frame');
  const wheelR = grp(bike, V(-AXLE, 0, rw));
  const front = grp(bike, V(PIVOT[0], 0, PIVOT[1]));
  put(front, 'front'); thin(front, 'frontFine');
  const wheelF = grp(front, F(AXLE, 0, rw));
  const tyre = P.build('wheel'), spokes = P.build('wheelFine', { shadow: false });
  for (const w of [wheelR, wheelF]) {                       // both wheels share the geometry
    const t = w === wheelR ? tyre : tyre.clone(), sp = w === wheelR ? spokes : spokes.clone();
    w.add(t, sp); ctx.noOutline(sp); fine.push(sp);
  }
  const crankG = grp(bike, V(BB[0], 0, BB[1]));
  put(crankG, 'crank');
  const pedalR = grp(crankG, V(0, -PEDAL_Y, CRANK)), pedalL = grp(crankG, V(0, PEDAL_Y, -CRANK));
  put(pedalR, 'pedalR'); put(pedalL, 'pedalL');
  put(rider, 'pelvis');
  const torso = grp(rider, V(hx, 0, hz));
  put(torso, 'torso');
  const head = grp(torso, new THREE.Vector3(0, lt + 0.05, 0.01));
  put(head, 'head');
  const bone = (name) => { const o = grp(rider); put(o, name); return o; };
  const side = (n, sgn) => ({ sgn, armU: bone('armU' + n), armF: bone('armF' + n), thigh: bone('thigh' + n), shin: bone('shin' + n), shoe: bone('shoe' + n),
    hip: V(hx, sgn * 0.055 * h, hz), grip: F(fit.grip[0], sgn * fit.grip[1], fit.grip[2]),
    shoulder0: new THREE.Vector3(sgn * 0.115 * h, lt - 0.045, -0.015), pedal: sgn < 0 ? pedalR : pedalL });
  const sides = [side('L', 1), side('R', -1)];
  for (const sd of sides) sd.shoe.rotation.order = 'ZYX';

  const axis = V(AXIS[0], 0, AXIS[1]).normalize(), pivot = front.position, torso0 = torso.position;
  const sh = [new THREE.Vector3(), new THREE.Vector3()], hand = [new THREE.Vector3(), new THREE.Vector3()];
  const mid = new THREE.Vector3(), end = new THREE.Vector3(), tgt = new THREE.Vector3(), pole = new THREE.Vector3(), gnd = new THREE.Vector3();
  const reach = 0.975 * (upper + fore);

  function pose(p) {
    const [roll, steer, crank, wheel, foot, look, effort] = p;
    lean.rotation.z = -roll;
    front.quaternion.setFromAxisAngle(axis, steer);
    wheelR.rotation.x = wheel; wheelF.rotation.x = wheel;
    crankG.rotation.x = crank;
    // the trunk leans until the hands reach the bars
    const twist = 0.2 * look;
    let ln = fit.lean0 + 0.10 * effort;
    for (let i = 0; i < 2; i++) hand[i].copy(sides[i].grip).applyQuaternion(front.quaternion).add(pivot);
    for (let it = 0; it < 7; it++) {
      _q.setFromEuler(_e.set(ln, twist, 0));
      let over = 0;
      for (let i = 0; i < 2; i++) { sh[i].copy(sides[i].shoulder0).applyQuaternion(_q).add(torso0); over = Math.max(over, sh[i].distanceTo(hand[i]) - reach); }
      if (over <= 0 || it === 6) break;
      ln += Math.min(0.25, 1.6 * over);
    }
    torso.rotation.set(ln, twist, 0);
    head.rotation.set(-0.75 * ln, 1.1 * look - twist, 0);
    for (let i = 0; i < 2; i++) {
      const sd = sides[i], sgn = sd.sgn;
      twoBone(sh[i], hand[i], upper, fore, pole.set(sgn * 0.6, -1, -0.3), mid, end);
      setBone(sd.armU, sh[i], mid); setBone(sd.armF, mid, end);
      // the foot on its pedal: more toe-down through the bottom of the stroke
      const ph = crank + (sgn > 0 ? Math.PI : 0);
      let phi = 0.30 - 0.22 * Math.cos(ph - 0.5);
      sd.pedal.rotation.x = phi - crank;
      const px = BB[0] + CRANK * Math.sin(ph), pz = BB[1] + CRANK * Math.cos(ph);
      tgt.copy(V(px - 0.105 * Math.cos(phi) + 0.085 * Math.sin(phi), sgn * 0.115, pz + 0.105 * Math.sin(phi) + 0.085 * Math.cos(phi)));
      let down = 0;
      if (sgn > 0 && foot > 0) {                             // the left foot to the ground, on its toes if the saddle is high
        down = foot;
        const cr = Math.cos(roll), sr = Math.sin(roll);
        let pg = 0;
        for (; pg < 0.95; pg += 0.15) {
          const y = 0.27, z = 0.005 + 0.105 * Math.sin(pg) + 0.085 * Math.cos(pg);
          gnd.copy(V(hx + 0.205 - 0.105 * Math.cos(pg) + 0.085 * Math.sin(pg), y * cr - z * sr, y * sr + z * cr));
          if (gnd.distanceTo(sd.hip) <= 0.99 * 2 * leg) break;
        }
        tgt.lerp(gnd, foot); phi += (pg - phi) * foot;
      }
      twoBone(sd.hip, tgt, leg, leg, pole.copy(V(1.0, sgn * (0.10 + 0.25 * down), 0.2)), mid, end);
      setBone(sd.thigh, sd.hip, mid); setBone(sd.shin, mid, end);
      sd.shoe.position.copy(end); sd.shoe.rotation.set(phi, 0, roll * down);
    }
  }
  pose([0, 0, 0, 0, 0, 0, 0]);
  return { g, fine, pose };
}
