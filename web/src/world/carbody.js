// 車体 (lofted) — the bodies of the redesigned detailed cars (src/world/cars2.js). Where cars.js extrudes a side
// profile across the width, a body here is a skin wrapped round a plan outline: front face, rounded front corner,
// side, rear corner, rear face and back along the other side, from a bottom edge (cut round the wheel arches) up to a
// top edge, closed by a deck (bonnet, shelf under the glass, boot lid; for the greenhouse: the roof). The side
// profile is still the table of cars.js design(); the plan (corner radii, taper to the nose and tail, bow of the
// faces), the section (tuck at the sill, fender swell at the axles, shoulder) and the tumblehome are added on top.
// Frame as cars.js: +Z forward, +Y up, origin on the ground at the centre; +X is the car's left.
import * as THREE from 'three';

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/** Piecewise-linear f(x) through [[x, f], ...], constant beyond the ends. */
export function lin(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0]);
  return (x) => {
    if (x <= p[0][0]) return p[0][1];
    for (let i = 1; i < p.length; i++) {
      if (x <= p[i][0]) { const [x0, y0] = p[i - 1], [x1, y1] = p[i]; return x1 - x0 < 1e-9 ? y1 : y0 + (y1 - y0) * (x - x0) / (x1 - x0); }
    }
    return p[p.length - 1][1];
  };
}

/** Round the inner corners of a polyline: each is cut back by up to r along its two edges and replaced by a curve. */
export function roundPoly(pts, r, n = 3) {
  const out = [pts[0]];
  for (let i = 1; i < pts.length - 1; i++) {
    const [a, b, c] = [pts[i - 1], pts[i], pts[i + 1]];
    const la = Math.hypot(b[0] - a[0], b[1] - a[1]), lc = Math.hypot(c[0] - b[0], c[1] - b[1]);
    const ra = Math.min(r, 0.45 * la), rc = Math.min(r, 0.45 * lc);
    if (ra < 1e-4 || rc < 1e-4) { out.push(b); continue; }
    const p0 = [b[0] + (a[0] - b[0]) * ra / la, b[1] + (a[1] - b[1]) * ra / la], p1 = [b[0] + (c[0] - b[0]) * rc / lc, b[1] + (c[1] - b[1]) * rc / lc];
    for (let k = 0; k <= n; k++) {
      const t = k / n, u = 1 - t;
      out.push([u * u * p0[0] + 2 * u * t * b[0] + t * t * p1[0], u * u * p0[1] + 2 * u * t * b[1] + t * t * p1[1]]);
    }
  }
  out.push(pts[pts.length - 1]);
  return out;
}

// ---------------------------------------------------------------------------------------------- meshes
/**
 * Collects triangles per material key. grid() takes a lattice of points and gives every vertex a smooth normal
 * within its smoothing group (default: the material), so a change of group is a crease: the cel shading's step and
 * the outline pass's line appear there.
 */
export function mesher() {
  const out = new Map();
  const bucket = (key) => { if (!out.has(key)) out.set(key, { p: [], n: [] }); return out.get(key); };

  /** P[j * nu + i] = [x, y, z]; quads (i, j) – (i + 1, j + 1), normal along du x dv. mat(i, j): material key or null
   *  (no quad); group(i, j): smoothing group. wrap: the lattice closes in i. */
  function grid(P, nu, nv, { mat, group = null, wrap = false, flip = false }) {
    const acc = new Map(), tris = [];
    const nI = wrap ? nu : nu - 1;
    const e1 = [0, 0, 0], e2 = [0, 0, 0];
    for (let j = 0; j < nv - 1; j++) for (let i = 0; i < nI; i++) {
      const m = typeof mat === 'function' ? mat(i, j) : mat;
      if (!m) continue;
      const g = (group && group(i, j)) || m;
      if (!acc.has(g)) acc.set(g, new Float64Array(nu * nv * 3));
      const A = acc.get(g);
      const a = j * nu + i, b = j * nu + (i + 1) % nu, c = (j + 1) * nu + (i + 1) % nu, d = (j + 1) * nu + i;
      for (const t of flip ? [[a, c, b], [a, d, c]] : [[a, b, c], [a, c, d]]) {
        const [p, q, r] = [P[t[0]], P[t[1]], P[t[2]]];
        for (let k = 0; k < 3; k++) { e1[k] = q[k] - p[k]; e2[k] = r[k] - p[k]; }
        const nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
        if (nx * nx + ny * ny + nz * nz < 1e-16) continue;          // collapsed (rows squeezed onto a wheel arch)
        for (const v of t) { A[3 * v] += nx; A[3 * v + 1] += ny; A[3 * v + 2] += nz; }
        tris.push(m, g, t);
      }
    }
    for (let k = 0; k < tris.length; k += 3) {
      const B = bucket(tris[k]), A = acc.get(tris[k + 1]);
      for (const v of tris[k + 2]) {
        const l = Math.hypot(A[3 * v], A[3 * v + 1], A[3 * v + 2]) || 1;
        B.p.push(P[v][0], P[v][1], P[v][2]); B.n.push(A[3 * v] / l, A[3 * v + 1] / l, A[3 * v + 2] / l);
      }
    }
  }

  /** Flat triangles (each with its face normal). pts: [[x, y, z], ...] in threes. */
  function flat(key, pts) {
    const B = bucket(key);
    for (let k = 0; k + 2 < pts.length; k += 3) {
      const [p, q, r] = [pts[k], pts[k + 1], pts[k + 2]];
      const e1 = [q[0] - p[0], q[1] - p[1], q[2] - p[2]], e2 = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const l = Math.hypot(...n);
      if (l < 1e-9) continue;
      for (const v of [p, q, r]) { B.p.push(v[0], v[1], v[2]); B.n.push(n[0] / l, n[1] / l, n[2] / l); }
    }
  }

  /** A three.js geometry (any attributes), moved into the bucket of `key`. */
  function geometry(key, g) {
    const x = g.index ? g.toNonIndexed() : g;
    if (!x.attributes.normal) x.computeVertexNormals();
    const B = bucket(key), p = x.attributes.position.array, n = x.attributes.normal.array;
    for (let i = 0; i < p.length; i++) { B.p.push(p[i]); B.n.push(n[i]); }
  }

  /** Every vertex of the given keys (default: all) through fn([x, y, z]) -> [x, y, z] (positions only: for the
   *  small corrections that fit a body to its box). */
  function warp(fn, keys = null) {
    for (const [key, B] of out) {
      if (keys && !keys.includes(key)) continue;
      for (let i = 0; i < B.p.length; i += 3) { const q = fn([B.p[i], B.p[i + 1], B.p[i + 2]]); B.p[i] = q[0]; B.p[i + 1] = q[1]; B.p[i + 2] = q[2]; }
    }
  }

  function bounds(keys = null) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const [key, B] of out) {
      if (keys && !keys.includes(key)) continue;
      for (let i = 0; i < B.p.length; i++) { const k = i % 3; if (B.p[i] < lo[k]) lo[k] = B.p[i]; if (B.p[i] > hi[k]) hi[k] = B.p[i]; }
    }
    return { lo, hi };
  }

  /** [[BufferGeometry, key], ...] */
  function parts(rename = (k) => k) {
    const res = [];
    for (const [key, B] of out) {
      if (!B.p.length) continue;
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(B.p, 3));
      g.setAttribute('normal', new THREE.Float32BufferAttribute(B.n, 3));
      res.push([g, rename(key)]);
    }
    return res;
  }
  const triangles = () => { let n = 0; for (const B of out.values()) n += B.p.length / 9; return n; };
  return { grid, flat, geometry, warp, bounds, parts, triangles, out };
}

// ---------------------------------------------------------------------------------------------- the shell
/**
 * The lattice of a lofted shell. S:
 *   hw(z, y)       half width of the side skin
 *   stations       side stations from the front corner's start to the rear corner's start: [{ z(y) }, ...]
 *   rows(st)       the K + 1 row heights of the side at a station, bottom to top
 *   front, rear    { z(y): the face's profile, rows: its K + 1 heights, r: corner radius across, bow: how far the
 *                  face's middle stands proud of its ends }
 *   deckY(z)       height of the deck's centre line
 *   nF, nC         quads across half a face, round a corner
 * Returns the skin (a closed loop of columns round the plan, K + 1 rows), the deck's lattice and the half outline's
 * descriptors: { zone: 'front' | 'cornerF' | 'side' | 'cornerR' | 'rear', ... } from the front's middle to the rear's.
 */
export function shell(S) {
  const nF = S.nF ?? 3, nC = S.nC ?? 6, K = S.front.rows.length - 1;
  const half = [];
  for (let m = 0; m < nF; m++) half.push({ zone: 'front', c: m / nF });
  for (let i = 0; i < nC; i++) half.push({ zone: 'cornerF', th: (Math.PI / 2) * (1 - i / nC) });
  S.stations.forEach((st, i) => half.push({ zone: 'side', st, i }));
  for (let i = 1; i <= nC; i++) half.push({ zone: 'cornerR', th: (Math.PI / 2) * (i / nC) });
  for (let m = nF - 1; m >= 0; m--) half.push({ zone: 'rear', c: m / nF });
  const N = half.length;
  const rowsOf = (st) => st.rows || (st.rows = S.rows(st));
  const ends = { F: { ...S.front, sgn: 1, st: S.stations[0] }, R: { ...S.rear, sgn: -1, st: S.stations[S.stations.length - 1] } };
  for (const e of Object.values(ends)) {            // the face's top edge follows the deck's crown
    const yK = e.rows[K];
    e.crown = Math.max(0, S.deckY(e.z(yK)) - yK);
    e.crown = Math.max(0, S.deckY(e.z(yK + e.crown)) - yK);
    e.q = e.crown / Math.max(yK - e.rows[0], 1e-6);
  }

  function point(o, k) {
    if (o.zone === 'side') { const y = rowsOf(o.st)[k], z = o.st.z(y); return [S.hw(z, y), y, z]; }
    const e = o.zone === 'front' || o.zone === 'cornerF' ? ends.F : ends.R;
    if (o.zone === 'front' || o.zone === 'rear') {
      const y = e.rows[0] + (e.rows[k] - e.rows[0]) * (1 + e.q * (1 - o.c * o.c));
      const xc = S.hw(e.st.z(y), y) - e.r;
      return [o.c * xc, y, e.z(y) - e.sgn * e.bow * o.c * o.c];
    }
    const w = Math.sin(o.th), ys = rowsOf(e.st)[k], y = ys + (e.rows[k] - ys) * w;
    const zs = e.st.z(y), xc = S.hw(zs, y) - e.r;
    const rz = Math.max(0.012, e.sgn * (e.z(y) - zs) - e.bow);
    return [xc + e.r * Math.cos(o.th), y, zs + e.sgn * rz * w];
  }

  // the skin: the half outline, then its mirror image back to the front
  const nu = 2 * N - 2, P = new Array(nu * (K + 1));
  for (let k = 0; k <= K; k++) {
    for (let h = 0; h < N; h++) {
      const p = point(half[h], k);
      P[k * nu + h] = p;
      if (h > 0 && h < N - 1) P[k * nu + nu - h] = [-p[0], p[1], p[2]];
    }
  }
  /** What the quad between columns i and i + 1 is: { zone, i (side: station index), side (1: +X, -1: -X), h }. */
  function quad(i) {
    const side = i < N - 1 ? 1 : -1, h = side > 0 ? i : nu - 1 - i;
    const a = half[h], b = half[h + 1];
    const zone = a.zone === 'front' ? 'front' : a.zone === 'cornerF' ? 'cornerF' : a.zone === 'side' ? (b.zone === 'side' ? 'side' : 'cornerR')
      : a.zone === 'cornerR' && b.zone === 'cornerR' ? 'cornerR' : 'rear';
    return { zone, side, h, i: a.zone === 'side' ? a.i : -1, a, b };
  }

  // the deck: one line across for every outline point from the front corner's start to the rear corner's end
  const cols = 2 * nF + 1, rim = half.slice(nF, N - nF), D = [];
  rim.forEach((o, j) => {
    const first = j === 0, last = j === rim.length - 1;
    if (first || last) {                              // the faces' own top edges
      for (let m = -nF; m <= nF; m++) { const p = point({ zone: first ? 'front' : 'rear', c: Math.abs(m) / nF }, K); D.push([Math.sign(m) * p[0], p[1], p[2]]); }
      return;
    }
    const [xe, ye, ze] = point(o, K);
    // how far a line's middle stands ahead of its ends: as the face's top edge does (its bow, less what the crown
    // takes back on a leaning face), fading out round the corner
    const lead = (e, zone) => point({ zone, c: 0 }, K)[2] - point({ zone, c: 1 }, K)[2];
    const bow = o.zone === 'cornerF' ? lead(ends.F, 'front') * Math.sin(o.th) : o.zone === 'cornerR' ? lead(ends.R, 'rear') * Math.sin(o.th) : 0;
    const crown = Math.max(0, S.deckY(ze + bow) - ye);
    for (let m = -nF; m <= nF; m++) { const c = m / nF, f = 1 - c * c; D.push([c * xe, ye + crown * f, ze + bow * f]); }
  });
  return { P, nu, nv: K + 1, quad, half, N, K, nF, nC, point, deck: { P: D, nu: cols, nv: rim.length, rim }, ends };
}

/** The top edge of a shell's skin in plan: x(z) of the +X side, round the corners too. */
export function planEdge(sh) {
  const pts = [];
  for (let h = 0; h < sh.N; h++) { const p = sh.P[sh.K * sh.nu + h]; pts.push([p[2], p[0]]); }
  return lin(pts);
}

// ---------------------------------------------------------------------------------------------- patches
/**
 * A panel lying on a surface: fn(a, b) -> [x, y, z] sampled over the quadrilateral with corners c00, c10, c11, c01
 * (in the surface's own coordinates), lifted `lift` off it along the normal, with a rim back down to the surface.
 * out: a vector the normal should point along (to orient it). Returns the lattices for mesher.grid.
 */
export function patch(M, key, fn, corners, { na = 4, nb = 2, lift = 0.006, out = [0, 0, 1], rim = true, group = null } = {}) {
  const [c00, c10, c11, c01] = corners;
  const S = (u, v) => {
    const a = (1 - u) * (1 - v) * c00[0] + u * (1 - v) * c10[0] + u * v * c11[0] + (1 - u) * v * c01[0];
    const b = (1 - u) * (1 - v) * c00[1] + u * (1 - v) * c10[1] + u * v * c11[1] + (1 - u) * v * c01[1];
    return fn(a, b);
  };
  const base = [], top = [], e = 1e-3;
  for (let j = 0; j <= nb; j++) for (let i = 0; i <= na; i++) {
    const u = i / na, v = j / nb, p = S(u, v);
    const pu = S(Math.min(1, u + e), v), pm = S(Math.max(0, u - e), v), pv = S(u, Math.min(1, v + e)), pw = S(u, Math.max(0, v - e));
    const du = [pu[0] - pm[0], pu[1] - pm[1], pu[2] - pm[2]], dv = [pv[0] - pw[0], pv[1] - pw[1], pv[2] - pw[2]];
    let n = [du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]];
    const l = Math.hypot(...n) || 1;
    n = n.map((x) => x / l);
    if (n[0] * out[0] + n[1] * out[1] + n[2] * out[2] < 0) n = n.map((x) => -x);
    base.push(p); top.push([p[0] + n[0] * lift, p[1] + n[1] * lift, p[2] + n[2] * lift]);
  }
  // orientation of the lattice against the wanted normal
  const p0 = top[0], pa = top[1], pb = top[na + 1];
  const du = [pa[0] - p0[0], pa[1] - p0[1], pa[2] - p0[2]], dv = [pb[0] - p0[0], pb[1] - p0[1], pb[2] - p0[2]];
  const nn = [du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]];
  const flip = nn[0] * out[0] + nn[1] * out[1] + nn[2] * out[2] < 0;
  M.grid(top, na + 1, nb + 1, { mat: key, flip, group: group ? () => group : null });
  if (rim && lift > 0.0015) {
    const ring = [];
    for (let i = 0; i <= na; i++) ring.push(i);
    for (let j = 1; j <= nb; j++) ring.push(j * (na + 1) + na);
    for (let i = na - 1; i >= 0; i--) ring.push(nb * (na + 1) + i);
    for (let j = nb - 1; j >= 1; j--) ring.push(j * (na + 1));
    const n = ring.length, R = [];
    for (const i of ring) R.push(base[i]);
    for (const i of ring) R.push(top[i]);
    M.grid(R, n, 2, { mat: key, wrap: true, flip: !flip, group: () => `${group || key}:rim` });
  }
}

/** The side of an ellipse quarter measured along it: angle (from the face's end, 0, to the side's start, pi / 2)
 *  for an arc length s, and the arc's whole length. Crude (mean radius), enough to place lamps round a corner. */
const arcOf = (r, rz) => ({ len: Math.PI / 2 * (r + rz) / 2, ang: (s) => s / ((r + rz) / 2) });

/**
 * Surface coordinates on a shell S (the spec given to shell()): returns
 *   side(z, y)        the +X side skin
 *   nose(s, y)        the front, s metres from the centre line towards +X: across the face, round the corner and on
 *                     along the side
 *   tail(s, y)        the same at the rear
 * Mirror x for the other side.
 */
export function surface(S) {
  const end = (e, sgn, st) => (s, y) => {
    const neg = s < 0; s = Math.abs(s);
    const zs = st.z(y), ws = S.hw(zs, y), xc = ws - e.r, rz = Math.max(0.012, sgn * (e.z(y) - zs) - e.bow);
    let p;
    if (s <= xc) p = [s, y, e.z(y) - sgn * e.bow * (s / xc) ** 2];
    else {
      const A = arcOf(e.r, rz), t = s - xc;
      if (t < A.len) { const th = Math.PI / 2 - A.ang(t); p = [xc + e.r * Math.cos(th), y, zs + sgn * rz * Math.sin(th)]; }
      else { const z = zs - sgn * (t - A.len); p = [S.hw(z, y), y, z]; }
    }
    return neg ? [-p[0], p[1], p[2]] : p;
  };
  return {
    side: (z, y) => [S.hw(z, y), y, z],
    nose: end(S.front, 1, S.stations[0]),
    tail: end(S.rear, -1, S.stations[S.stations.length - 1]),
  };
}

/** Station z values: the given ones (kept), filled so that no gap is wider than `step`; descending. */
export function fillStations(keep, step) {
  const ks = [...new Set(keep.map((z) => Math.round(z * 1e4) / 1e4))].sort((a, b) => b - a);
  const out = [];
  for (let i = 0; i < ks.length; i++) {
    out.push(ks[i]);
    if (i + 1 < ks.length) {
      const gap = ks[i] - ks[i + 1], n = Math.ceil(gap / step - 1e-6);
      for (let k = 1; k < n; k++) out.push(ks[i] - gap * k / n);
    }
  }
  return out;
}
