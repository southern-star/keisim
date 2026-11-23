// 車の顔 — the faces of the redesigned detailed cars (src/world/cars2.js): headlamps, grilles, bumper openings, fog
// and daytime running lamps, tail lamps, plates. Everything is a patch lying on the lofted skin (src/world/carbody.js
// patch(): a few millimetres off the surface, no rim: a rim that thin breaks the outline pass's line into dashes).
// A car's face is picked from its body type's list by a hash of what KeiSim gives the car (id, size, colour): the
// same car always gets the same face, and no Math.random().
import * as THREE from 'three';
import { patch } from './carbody.js';

/** 32-bit hash of a car's inputs. */
export function carHash(id, L, W, H, rgb) {
  let h = 2166136261;
  for (const v of [id, Math.round(L * 1000), Math.round(W * 1000), Math.round(H * 1000), rgb[0], rgb[1], rgb[2]]) {
    const u = Math.round(v) | 0;
    h = Math.imul(h ^ (u & 0xffff), 16777619); h = Math.imul(h ^ (u >>> 16), 16777619);
  }
  return h >>> 0;
}
/** The k-th number in [0, 1) of a hash (independent draws for independent choices). */
export function roll(h, k) {
  let a = (h + Math.imul(k + 1, 0x9E3779B9)) | 0;
  a = Math.imul(a ^ (a >>> 16), 0x85ebca6b); a = Math.imul(a ^ (a >>> 13), 0xc2b2ae35);
  return ((a ^ (a >>> 16)) >>> 0) / 4294967296;
}

/** Faces by body type: front [headlamp, grille], rear tail lamp. */
const FACES = {
  compact: { front: [['wide', 'bars'], ['slant', 'hex'], ['slant', 'line'], ['round', 'none']], rear: ['wrap', 'bar', 'ell'] },
  keihatch: { front: [['wide', 'bars'], ['round', 'line'], ['slant', 'hex'], ['tall', 'none']], rear: ['wrap', 'tall', 'ell'] },
  keiwagon: { front: [['tall', 'bars'], ['wide', 'bars'], ['round', 'line'], ['slim', 'hex']], rear: ['tall', 'ell', 'slim'] },
  sedan: { front: [['wide', 'bars'], ['slant', 'hex'], ['slim', 'line'], ['slant', 'none']], rear: ['wrap', 'bar', 'slim'] },
  suv: { front: [['wide', 'bars'], ['slim', 'hex'], ['round', 'bars'], ['slant', 'line']], rear: ['slim', 'bar', 'ell'] },
  minivan: { front: [['tall', 'bars'], ['wide', 'hex'], ['slim', 'bars'], ['slant', 'line']], rear: ['tall', 'ell', 'bar'] },
  keitruck: { front: [['wide', 'slot'], ['round', 'slot'], ['tall', 'mask']], rear: ['box', 'stack', 'twin'] },
  truck: { front: [['wide', 'panel'], ['twin', 'panel'], ['tall', 'mask']], rear: ['box', 'stack', 'twin'] },
  // the types that share a size class with the eight above (src/world/cars2.js SUBTYPES)
  keisedan: { front: [['wide', 'bars'], ['slant', 'hex'], ['slim', 'line'], ['round', 'line']], rear: ['wrap', 'slim', 'ell'] },
  keisuv: { front: [['round', 'bars'], ['round', 'line'], ['tall', 'bars']], rear: ['tall', 'ell', 'wrap'] },
  keivan: { front: [['tall', 'none'], ['wide', 'line'], ['round', 'line']], rear: ['tall', 'slim', 'ell'] },
  tallwagon: { front: [['tall', 'bars'], ['wide', 'bars'], ['slim', 'hex'], ['slant', 'line']], rear: ['tall', 'ell', 'bar'] },
  smallsuv: { front: [['wide', 'bars'], ['slim', 'hex'], ['round', 'bars'], ['slant', 'line']], rear: ['slim', 'bar', 'ell'] },
  wagon: { front: [['wide', 'bars'], ['slant', 'hex'], ['slim', 'line'], ['slant', 'none']], rear: ['slim', 'bar', 'ell'] },
  hatch: { front: [['slim', 'line'], ['slant', 'hex'], ['slant', 'none'], ['wide', 'bars']], rear: ['bar', 'slim', 'wrap'] },
  fourwd: { front: [['round', 'bars'], ['wide', 'bars'], ['tall', 'bars']], rear: ['tall', 'ell', 'slim'] },
  talltaxi: { front: [['wide', 'bars'], ['slim', 'bars'], ['wide', 'hex']], rear: ['tall', 'ell', 'bar'] },
  onebox: { front: [['tall', 'bars'], ['wide', 'line'], ['tall', 'none'], ['slim', 'bars']], rear: ['tall', 'slim', 'ell'] },
};
for (const t of ['flatbed', 'reefer', 'delivery']) FACES[t] = FACES.truck;
export const faceCount = (type) => ({ front: FACES[type].front.length, rear: FACES[type].rear.length });

/** A car's face: { lamp, grille, tail, drl ('brow' | 'fang' | null), fog, intake, key }. force: { front, rear } picks
 *  the list entries by index (the gallery's ?faces=1). */
export function chooseFace(type, h, force = null) {
  const t = FACES[type];
  const fi = force && force.front != null ? force.front % t.front.length : Math.floor(roll(h, 1) * t.front.length);
  const ri = force && force.rear != null ? force.rear % t.rear.length : Math.floor(roll(h, 2) * t.rear.length);
  const [lamp, grille] = t.front[fi];
  const drl = roll(h, 3) < 0.4 ? (roll(h, 6) < 0.5 && lamp !== 'round' && lamp !== 'twin' ? 'brow' : 'fang') : null;
  const fog = !drl && roll(h, 4) < 0.5, intake = roll(h, 5) < 0.5 ? 'wide' : 'narrow';
  return { lamp, grille, tail: t.rear[ri], drl, fog, intake, key: `${lamp}+${grille}/${t.rear[ri]}${drl ? `+${drl}` : fog ? '+fog' : ''}${intake === 'wide' ? '' : '+n'}` };
}

const F = [0, 0, 1], R = [0, 0, -1];
/** Part of a quadrilateral [c00, c10, c11, c01]: u in [u0, u1] along its first edge, v in [v0, v1] across. */
const sub = (c, u0, u1, v0, v1) => {
  const P = (u, v) => [0, 1].map((k) => (1 - u) * (1 - v) * c[0][k] + u * (1 - v) * c[1][k] + u * v * c[2][k] + (1 - u) * v * c[3][k]);
  return [P(u0, v0), P(u1, v0), P(u1, v1), P(u0, v1)];
};
const mir = (c, s) => c.map(([a, b]) => [s * a, b]);
const rect = (a0, a1, y0, y1) => [[a0, y0], [a1, y0], [a1, y1], [a0, y1]];
const lay = (M, key, fn, q, out, lift = 0.005, na = 4, nb = 1) => patch(M, key, fn, q, { na, nb, lift, out, rim: false });
/** A band across a face and round its corners, from -half to half: few columns on the flat middle (|s| < flat), one
 *  every 7 cm or less round the corners (wider ones would cut through the rounded skin); nb rows up it (a band
 *  over the nose's tip needs several to follow the profile). */
function band(M, key, fn, half, flat, y0, y1, out, lift = 0.004, nb = 2) {
  const f = Math.min(flat, half);
  lay(M, key, fn, rect(-f, f, y0, y1), out, lift, 4, nb);
  if (half > f) for (const s of [-1, 1]) lay(M, key, fn, mir(rect(f, half, y0, y1), s), out, lift, Math.ceil((half - f) / 0.07), nb);
}
/** A disc (or ring from r0) on a surface round (sc, yc). */
const disc = (M, key, fn, sc, yc, r0, r1, out, lift, n = 12) => patch(M, key,
  (a, b) => fn(sc + b * Math.cos(a), yc + b * Math.sin(a)), [[0, Math.max(r0, 0.004)], [2 * Math.PI, Math.max(r0, 0.004)], [2 * Math.PI, r1], [0, r1]],
  { na: n, nb: 1, lift, out, rim: false });

// seven-segment digits: which of a (top), b, c (right, upper and lower), d (bottom), e, f (left, lower and upper), g (middle)
const SEG = ['abcdef', 'bc', 'abged', 'abgcd', 'fgbc', 'afgcd', 'afgedc', 'abc', 'abcdefg', 'abcdfg'];
/**
 * A number plate: a thin board at height y and depth z (its middle), leaning by tilt, facing sgn (1 forward, -1
 * back), with its four-digit number no in flat strokes and a short line above them (where the office's name stands).
 * White and yellow plates carry dark strokes, green ones (key 'plateG') white ones.
 */
export function plateAt(M, key, y, z, tilt, sgn, no, w = 0.33, h = 0.165) {
  const g = new THREE.BoxGeometry(w, h, 0.012);
  const pos = [], zf = 0.0068, t = 0.011, dw = 0.046, dh = 0.082, y0 = -0.058;
  const quad = (x0, ya, x1, yb) => pos.push(x0, ya, zf, x1, ya, zf, x1, yb, zf, x0, ya, zf, x1, yb, zf, x0, yb, zf);
  const digits = String(no % 10000).padStart(4, '0');
  [-0.118, -0.06, 0.014, 0.072].forEach((x, i) => {
    const n = Number(digits[i]);
    if (i < 3 && Number(digits.slice(0, i + 1)) === 0) { quad(x + dw / 2 - 0.006, y0 + dh / 2 - 0.006, x + dw / 2 + 0.006, y0 + dh / 2 + 0.006); return; }   // leading zeros: a dot
    const on = SEG[n], ym = y0 + dh / 2;
    if (on.includes('a')) quad(x, y0 + dh - t, x + dw, y0 + dh);
    if (on.includes('g')) quad(x, ym - t / 2, x + dw, ym + t / 2);
    if (on.includes('d')) quad(x, y0, x + dw, y0 + t);
    if (on.includes('f')) quad(x, ym, x + t, y0 + dh);
    if (on.includes('b')) quad(x + dw - t, ym, x + dw, y0 + dh);
    if (on.includes('e')) quad(x, y0, x + t, ym);
    if (on.includes('c')) quad(x + dw - t, y0, x + dw, ym);
  });
  quad(-0.01, y0 + dh / 2 - 0.004, 0.006, y0 + dh / 2 + 0.004);                            // hyphen
  quad(-0.07, 0.042, 0.07, 0.058);                                                        // the line above
  const ink = new THREE.BufferGeometry();
  ink.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  ink.computeVertexNormals();
  for (const x of [g, ink]) { if (sgn < 0) x.rotateY(Math.PI); x.rotateX(tilt); x.translate(0, y, z); }
  M.geometry(key, g);
  M.geometry(key === 'plateG' ? 'plateW' : 'trim', ink);
}

/** A learner's or an older driver's mark on a surface round (sc, yc): 'beginner' a two-coloured arrow head, 'elder' a
 *  four-leaved one. Flat colours only (the plates' yellow and green). */
function mark(M, fn, kind, sc, yc, out) {
  const q = (key, pts) => lay(M, key, fn, pts.map(([a, b]) => [sc + a, yc + b]), out, 0.005, 1, 1);
  if (kind === 'beginner') {
    q('plateY', [[-0.055, 0.075], [0, 0.04], [0, -0.075], [-0.055, -0.02]]);
    q('plateG', [[0, 0.04], [0.055, 0.075], [0.055, -0.02], [0, -0.075]]);
  } else {
    q('plateY', [[-0.07, 0], [-0.035, -0.04], [0, 0], [-0.035, 0.04]]); q('plateG', [[0, 0], [0.035, -0.04], [0.07, 0], [0.035, 0.04]]);
    q('plateG', [[-0.035, 0.04], [0, 0], [0.035, 0.04], [0, 0.08]]); q('plateY', [[-0.035, -0.04], [0, -0.08], [0.035, -0.04], [0, 0]]);
  }
}

/** A plate standing on a surface at its centre line, y up from the ground, with a dark recess behind it. */
function plate(M, surf, y, sgn, key, no, recess = true) {
  const p = surf(0, y), q = surf(0, y + 0.08);
  plateAt(M, key, y, p[2] + sgn * 0.004, Math.atan2(q[2] - p[2], 0.08), sgn, no);
  if (recess) lay(M, 'trim', surf, rect(-0.185, 0.185, y - 0.095, y + 0.095), [0, 0, sgn], 0.004, 2, 1);
}

/** Headlamp outlines on the +X half of a face of half width hw, top at y1; k narrows them on a narrow car. */
function lampShape(kind, hw, y1, k) {
  switch (kind) {
    case 'slant': return { q: [[hw - 0.5 * k, y1 - 0.1], [hw - 0.06, y1 - 0.095], [hw + 0.05, y1 + 0.004], [hw - 0.4 * k, y1 - 0.06]], inner: hw - 0.5 * k };
    case 'tall': return { q: [[hw - 0.25 * k, y1 - 0.19], [hw - 0.02, y1 - 0.2], [hw + 0.04, y1], [hw - 0.23 * k, y1 - 0.005]], inner: hw - 0.25 * k };
    case 'slim': return { q: [[hw - 0.52 * k, y1 - 0.075], [hw - 0.04, y1 - 0.065], [hw + 0.04, y1], [hw - 0.5 * k, y1 - 0.025]], inner: hw - 0.52 * k };
    case 'round': return { c: [[hw - 0.27 * k, y1 - 0.09]], r: 0.076, inner: hw - 0.27 * k - 0.1 };
    default: return { q: [[hw - 0.47 * k, y1 - 0.12], [hw - 0.03, y1 - 0.1], [hw + 0.03, y1], [hw - 0.45 * k, y1 - 0.035]], inner: hw - 0.47 * k };
  }
}
/** A lamp on side s: the lens, a dark frame inside it and the bright projector within that. */
function drawLamp(M, fn, sh, s, out) {
  if (sh.q) {
    const q = mir(sh.q, s);
    patch(M, 'head', fn, q, { na: 6, nb: 2, lift: 0.005, out, rim: false });
    lay(M, 'trim', fn, sub(q, 0.07, 0.52, 0.2, 0.8), out, 0.008, 3, 1);
    lay(M, 'head', fn, sub(q, 0.13, 0.44, 0.34, 0.66), out, 0.011, 2, 1);
    return;
  }
  for (const [sc, yc] of sh.c) {
    disc(M, 'trim', fn, s * sc, yc, 0, sh.r + 0.02, out, 0.004, 14);
    disc(M, 'head', fn, s * sc, yc, 0, sh.r, out, 0.007, 14);
    disc(M, 'trim', fn, s * sc, yc, sh.r * 0.5, sh.r * 0.68, out, 0.01, 12);
  }
}

/** Grille between the lamps: gi is the half width they leave, y0 .. y1 its band. */
function drawGrille(M, fn, kind, gi, y0, y1, out) {
  const a = Math.max(0.12, gi - 0.06), b = Math.max(0.15, gi - 0.02);
  const bar = (y, half) => lay(M, 'hub', fn, rect(-half, half, y - 0.006, y + 0.006), out, 0.008, 6, 1);
  if (kind === 'bars') {                                 // dark opening with bright bars across
    lay(M, 'trim', fn, [[-a, y0], [a, y0], [b, y1], [-b, y1]], out, 0.005, 6, 2);
    const n = y1 - y0 > 0.12 ? 3 : 2;
    for (let i = 1; i <= n; i++) { const v = i / (n + 1); bar(y0 + (y1 - y0) * v, a + (b - a) * v - 0.02); }
  } else if (kind === 'hex') {                           // a six-sided mesh opening that reaches down into the bumper
    const ym = y0 + (y1 - y0) * 0.45, yb = y0 - 0.07, c = a * 0.6, t = b * 0.86;
    lay(M, 'trim', fn, [[-c, yb], [c, yb], [b, ym], [-b, ym]], out, 0.005, 6, 2);
    lay(M, 'trim', fn, [[-b, ym], [b, ym], [t, y1], [-t, y1]], out, 0.005, 6, 1);
    bar(ym, b - 0.03);
  } else if (kind === 'line') {                          // one thin slot from lamp to lamp
    lay(M, 'trim', fn, rect(-b, b, y1 - 0.042, y1 - 0.014), out, 0.005, 6, 1);
  }                                                      // 'none': a closed nose (electric car style)
}

/**
 * The front of a car with a nose. c: { M, nose(s, y), W, gc, yHood (the bonnet's front edge), gh (grille height),
 * clad (SUV cladding), plateKey }.
 */
export function frontFace(c, f) {
  const { M, nose, W, gc, yHood } = c, hw = W / 2, k = Math.min(1, W / 1.72), y1 = yHood - 0.02;
  const sh = lampShape(f.lamp, hw, y1, k);
  for (const s of [-1, 1]) {
    drawLamp(M, nose, sh, s, F);
    if (f.drl === 'brow') {                              // daytime running lamp: a bright line under the headlamp
      const [a, b] = sh.q;
      lay(M, 'head', nose, mir([[a[0] + 0.02, a[1] - 0.034], [b[0], b[1] - 0.034], [b[0], b[1] - 0.014], [a[0] + 0.02, a[1] - 0.014]], s), F, 0.005, 5, 1);
    } else if (f.drl === 'fang') {                       // or an upright one at the bumper's corner
      const y = c.clad ? gc + 0.17 : gc + 0.075;
      lay(M, 'head', nose, mir([[hw - 0.2 * k, y], [hw - 0.17 * k, y], [hw - 0.14 * k, y + 0.13], [hw - 0.17 * k, y + 0.13]], s), F, 0.005, 1, 3);
    }
    if (f.fog) {
      const y = c.clad ? gc + 0.21 : gc + 0.125, x = hw - 0.2 * k;
      disc(M, 'trim', nose, s * x, y, 0, 0.052, F, 0.004, 10);
      disc(M, 'head', nose, s * x, y, 0, 0.034, F, 0.007, 10);
    }
  }
  drawGrille(M, nose, f.grille, sh.inner, yHood - 0.045 - (c.gh || 0.09), yHood - 0.045, F);
  if (c.clad) {                                          // unpainted lower bumper with a bright skid plate
    band(M, 'trim', nose, hw + 0.2, hw - c.r - 0.04, gc + 0.004, gc + 0.14, F);
    lay(M, 'hub', nose, [[-W * 0.17, gc + 0.012], [W * 0.17, gc + 0.012], [W * 0.21, gc + 0.07], [-W * 0.21, gc + 0.07]], F, 0.008, 6, 1);
  } else {
    const w = f.intake === 'wide' ? 0.27 : 0.17;
    lay(M, 'trim', nose, [[-W * w, gc + 0.07], [W * w, gc + 0.07], [W * (w + 0.035), gc + 0.155], [-W * (w + 0.035), gc + 0.155]], F, 0.005, 6, 2);
  }
  plate(M, nose, gc + 0.26, 1, c.plateKey, c.plateNo);
  if (c.mark) mark(M, nose, c.mark, -W * 0.3, gc + 0.3, F);
}

/**
 * The rear of a car. c: { M, T (the lamps' own mesher), tail(s, y), gtail(s, y) (the greenhouse's rear), W, gc,
 * yT (top of the rear face), yE (the roof's rear edge), belt, hatch, clad, plateKey }.
 */
export function rearFace(c, f) {
  const { M, T, tail, gtail, W, gc, yT } = c, hw = W / 2, k = Math.min(1, W / 1.72);
  const lamp = (q, na = 6, nb = 2) => patch(T, 'tail', tail, q, { na, nb, lift: 0.006, out: R, rim: false });
  const white = (q) => lay(M, 'hub', tail, q, R, 0.009, 2, 1);                       // reversing lamp cell
  for (const s of [-1, 1]) {
    if (f.tail === 'tall') {                             // upright lamps on the corners
      lamp(mir([[hw - 0.2 * k, yT - 0.4], [hw, yT - 0.4], [hw + 0.03, yT - 0.035], [hw - 0.18 * k, yT - 0.035]], s), 3, 4);
      white(mir(rect(hw - 0.17 * k, hw - 0.06, yT - 0.37, yT - 0.31), s));
    } else if (f.tail === 'slim') {                      // thin lamps reaching far in
      const q = mir([[hw - 0.52 * k, yT - 0.125], [hw + 0.08, yT - 0.125], [hw + 0.1, yT - 0.05], [hw - 0.52 * k, yT - 0.065]], s);
      lamp(q); white(sub(q, 0.06, 0.26, 0.25, 0.75));
    } else if (f.tail === 'ell') {                       // an upright on the corner with a leg along the top
      lamp(mir([[hw - 0.15 * k, yT - 0.3], [hw + 0.03, yT - 0.3], [hw + 0.05, yT - 0.05], [hw - 0.13 * k, yT - 0.05]], s), 3, 3);
      lamp(mir([[hw - 0.48 * k, yT - 0.115], [hw - 0.15 * k, yT - 0.115], [hw - 0.13 * k, yT - 0.05], [hw - 0.48 * k, yT - 0.05]], s), 4, 1);
      white(mir(rect(hw - 0.12 * k, hw - 0.03, yT - 0.29, yT - 0.24), s));
    } else if (f.tail === 'bar') {                       // one line of light across the whole rear
      lamp(mir([[hw - 0.4 * k, yT - 0.15], [hw + 0.07, yT - 0.15], [hw + 0.09, yT - 0.05], [hw - 0.4 * k, yT - 0.06]], s));
    } else {                                             // 'wrap': blocks round the corners
      const q = mir([[hw - 0.42 * k, yT - 0.2], [hw + 0.06, yT - 0.2], [hw + 0.1, yT - 0.06], [hw - 0.4 * k, yT - 0.08]], s);
      lamp(q); white(sub(q, 0.07, 0.3, 0.12, 0.42));
    }
  }
  if (f.tail === 'bar') lamp(rect(-(hw - 0.4 * k), hw - 0.4 * k, yT - 0.118, yT - 0.082), 8, 1);
  // high-mount stop lamp under the roof's edge
  patch(T, 'tail', gtail, rect(-0.16, 0.16, c.yE - 0.125, c.yE - 0.09), { na: 2, nb: 1, lift: 0.008, out: R, rim: false });
  if (c.clad) {
    band(M, 'trim', tail, hw + 0.16, hw - c.rR - 0.04, gc + 0.004, gc + 0.15, R);
    lay(M, 'hub', tail, [[-W * 0.17, gc + 0.012], [W * 0.17, gc + 0.012], [W * 0.2, gc + 0.07], [-W * 0.2, gc + 0.07]], R, 0.008, 6, 1);
  } else lay(M, 'trim', tail, rect(-W * 0.3, W * 0.3, gc + 0.015, gc + 0.075), R, 0.005, 6, 1);
  lay(M, 'trim', tail, rect(-(hw - 0.22), hw - 0.22, gc + 0.425, gc + 0.437), R, 0.004, 8, 1);       // bumper / lid seam
  plate(M, tail, gc + 0.31, -1, c.plateKey, c.plateNo);
  if (c.mark) mark(M, tail, c.mark, W * 0.3, gc + 0.33, R);
  if (c.hatch) lay(M, 'trim', gtail, [[-0.28, c.belt + 0.07], [0.14, c.belt + 0.1], [0.14, c.belt + 0.118], [-0.28, c.belt + 0.088]], R, 0.008, 3, 1);   // wiper
}

/** The front of a cab-over truck. c: { M, nose, W, kei, plateKey }. */
export function cabFace(c, f) {
  const { M, nose, W } = c, hw = W / 2;
  const bars = (half, y0, y1, n) => { for (let i = 1; i <= n; i++) { const y = y0 + (y1 - y0) * i / (n + 1); lay(M, 'hub', nose, rect(-half, half, y - 0.007, y + 0.007), F, 0.008, 6, 1); } };
  if (c.kei) {
    band(M, 'trim', nose, hw + 0.1, hw - c.r - 0.04, 0.335, 0.47, F, 0.005, 3);               // bumper
    const sh = f.lamp === 'round' ? { c: [[hw - 0.22, 0.78]], r: 0.076, inner: hw - 0.33 }
      : f.lamp === 'tall' ? { q: [[hw - 0.26, 0.64], [hw - 0.07, 0.64], [hw - 0.05, 0.87], [hw - 0.26, 0.87]], inner: hw - 0.26 }
        : { q: [[hw - 0.38, 0.71], [hw - 0.07, 0.71], [hw - 0.05, 0.85], [hw - 0.38, 0.85]], inner: hw - 0.38 };
    if (f.grille === 'mask') lay(M, 'trim', nose, rect(-(sh.inner + 0.01), sh.inner + 0.01, 0.7, 0.84), F, 0.004, 6, 1);
    else lay(M, 'trim', nose, rect(-(sh.inner - 0.04), sh.inner - 0.04, 0.705, 0.75), F, 0.005, 4, 1);
    for (const s of [-1, 1]) drawLamp(M, nose, sh, s, F);
    plate(M, nose, 0.405, 1, c.plateKey, c.plateNo, false);
    return;
  }
  band(M, 'trim', nose, hw + 0.1, hw - c.r - 0.04, 0.47, 0.69, F, 0.005, 4);                  // bumper
  const sh = f.lamp === 'twin' ? { c: [[hw - 0.2, 0.58], [hw - 0.36, 0.58]], r: 0.058, inner: hw - 0.46 }
    : f.lamp === 'tall' ? { q: [[hw - 0.27, 0.74], [hw - 0.06, 0.74], [hw - 0.04, 1.02], [hw - 0.27, 1.02]], inner: hw - 0.27 }
      : { q: [[hw - 0.45, 0.51], [hw - 0.12, 0.51], [hw - 0.12, 0.65], [hw - 0.45, 0.65]], inner: hw - 0.45 };
  if (f.grille === 'mask') {
    lay(M, 'trim', nose, rect(-(sh.inner + 0.01), sh.inner + 0.01, 0.76, 1.0), F, 0.004, 6, 2);
    bars(sh.inner - 0.03, 0.76, 1.0, 3);
  } else {
    lay(M, 'trim', nose, [[-W * 0.25, 0.86], [W * 0.25, 0.86], [W * 0.27, 1.06], [-W * 0.27, 1.06]], F, 0.005, 6, 2);
    bars(W * 0.24, 0.86, 1.06, 2);
  }
  for (const s of [-1, 1]) {
    drawLamp(M, nose, sh, s, F);
    if (f.lamp === 'tall') disc(M, 'head', nose, s * (hw - 0.3), 0.58, 0, 0.045, F, 0.008, 10);
  }
  plate(M, nose, 0.58, 1, c.plateKey, c.plateNo, false);
}

/** Tail lamps of a truck: boxes (or drums) on the frame's rear end, on both sides at x, y with their rear face at z. */
export function truckTails(tails, kind, x, y, z, big) {
  const k = big ? 1.15 : 1;
  for (const s of [-1, 1]) {
    if (kind === 'twin') {
      for (const e of [-1, 1]) {
        const g = new THREE.CylinderGeometry(0.05 * k, 0.05 * k, 0.04, 12); g.rotateX(Math.PI / 2); g.translate(s * (x + e * 0.06 * k), y, z + 0.02); tails.push(g);
      }
    } else {
      const [w, h] = kind === 'stack' ? [0.1 * k, 0.2 * k] : [0.2 * k, 0.12 * k];
      const g = new THREE.BoxGeometry(w, h, 0.04); g.translate(s * x, y, z + 0.02); tails.push(g);
    }
  }
}
