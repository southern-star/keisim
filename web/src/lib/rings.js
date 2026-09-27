// Geometry helpers for closed outlines ("rings") in the XZ plane: walls along a ring (curb faces)
// and constant-width bands just inside / outside a ring (gutters, curb stones).
import * as THREE from 'three';

/** Signed area in the (x, z) plane (positive = counter-clockwise when +x right, +z up). */
export function signedArea(ring) {
  let a = 0;
  for (let i = 0; i < ring.length; i++) { const p = ring[i], q = ring[(i + 1) % ring.length]; a += p[0] * q[1] - q[0] * p[1]; }
  return a / 2;
}

/** Unit normal of every edge pointing away from the region the ring bounds.
 *  hole = true: the region lies outside the ring (it is a hole of a shape). */
function edgeNormals(ring, hole) {
  const ccw = signedArea(ring) > 0;
  const out = [];
  for (let i = 0; i < ring.length; i++) {
    const p = ring[i], q = ring[(i + 1) % ring.length];
    let dx = q[0] - p[0], dz = q[1] - p[1]; const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
    // right normal (dz, -dx) points outside a CCW ring
    let nx = dz, nz = -dx;
    if (!ccw) { nx = -nx; nz = -nz; }
    if (hole) { nx = -nx; nz = -nz; }
    out.push([nx, nz]);
  }
  return out;
}

/** Mitred per-vertex offset directions (length >= 1, clamped) from edge normals. */
function vertexOffsets(en) {
  const n = en.length, out = [];
  for (let i = 0; i < n; i++) {
    const a = en[(i - 1 + n) % n], b = en[i];
    let x = a[0] + b[0], z = a[1] + b[1]; const l = Math.hypot(x, z);
    if (l < 1e-6) { out.push([b[0], b[1]]); continue; }
    x /= l; z /= l;
    const c = Math.max(0.5, x * b[0] + z * b[1]);   // cos of half the turn; clamp the mitre at 2x
    out.push([x / c, z / c]);
  }
  return out;
}

/** Vertical wall along a ring from y0 to y1, facing away from the region (hole: region is outside). */
export function wallGeometry(rings, y0, y1, uvScale = 1) {
  const P = [], N = [], U = [], I = [];
  for (const { ring, hole } of rings) {
    const en = edgeNormals(ring, hole);
    let u = 0;
    for (let i = 0; i < ring.length; i++) {
      const p = ring[i], q = ring[(i + 1) % ring.length], n = en[i];
      const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
      if (len < 1e-4) continue;
      const k = P.length / 3;
      P.push(p[0], y0, p[1], q[0], y0, q[1], q[0], y1, q[1], p[0], y1, p[1]);
      for (let j = 0; j < 4; j++) N.push(n[0], 0, n[1]);
      U.push(u / uvScale, y0 / uvScale, (u + len) / uvScale, y0 / uvScale, (u + len) / uvScale, y1 / uvScale, u / uvScale, y1 / uvScale);
      u += len;
      // winding: front face must look along n
      const ex = q[0] - p[0], ez = q[1] - p[1];            // bottom edge p->q, up = +y
      const cx = -ez * (y1 - y0), cz = ex * (y1 - y0);     // (q-p) x up  (horizontal part)
      if (cx * n[0] + cz * n[1] >= 0) I.push(k, k + 1, k + 2, k, k + 2, k + 3); else I.push(k, k + 2, k + 1, k, k + 3, k + 2);
    }
  }
  return indexed(P, N, U, I);
}

/** Flat band of width w along rings at height y: side 'in' = inside the region, 'out' = outside. */
export function bandGeometry(rings, w, y, side = 'in', uvScale = 1) {
  const P = [], N = [], U = [], I = [];
  for (const { ring, hole } of rings) {
    const vo = vertexOffsets(edgeNormals(ring, hole));
    const s = side === 'in' ? -w : w;
    const n = ring.length, k0 = P.length / 3;
    let u = 0;
    for (let i = 0; i <= n; i++) {
      const p = ring[i % n], o = vo[i % n];
      if (i > 0) { const a = ring[i - 1]; u += Math.hypot(p[0] - a[0], p[1] - a[1]); }
      P.push(p[0], y, p[1], p[0] + o[0] * s, y, p[1] + o[1] * s);
      N.push(0, 1, 0, 0, 1, 0);
      U.push(u / uvScale, 0, u / uvScale, w / uvScale);
    }
    for (let i = 0; i < n; i++) {
      const a = k0 + i * 2;
      I.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const g = indexed(P, N, U, I);
  faceUp(g);
  return g;
}

/** THREE.ShapeGeometry for shapes given as {outer, holes} rings in (x, z), laid flat at height y,
 *  facing up, with world-space UVs (metres / uvScale). */
export function flatShapes(shapes, y, uvScale = 1) {
  const geos = [];
  for (const s of shapes) {
    const sh = new THREE.Shape(s.outer.map(([x, z]) => new THREE.Vector2(x, -z)));
    for (const h of s.holes) sh.holes.push(new THREE.Path(h.map(([x, z]) => new THREE.Vector2(x, -z))));
    const g = new THREE.ShapeGeometry(sh, 1);
    g.rotateX(-Math.PI / 2);          // shape (x, -z) -> world (x, 0, z), normal +Y
    g.translate(0, y, 0);
    const pos = g.attributes.position, uv = g.attributes.uv;
    for (let i = 0; i < pos.count; i++) uv.setXY(i, pos.getX(i) / uvScale, -pos.getZ(i) / uvScale);
    geos.push(g);
  }
  return geos;
}

function indexed(P, N, U, I) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(U, 2));
  g.setIndex(I);
  return g;
}

/** Flip triangles whose winding faces down (for flat, horizontal geometry). */
export function faceUp(g) {
  const p = g.attributes.position, I = g.index.array;
  for (let i = 0; i < I.length; i += 3) {
    const a = I[i], b = I[i + 1], c = I[i + 2];
    const e1x = p.getX(b) - p.getX(a), e1z = p.getZ(b) - p.getZ(a), e2x = p.getX(c) - p.getX(a), e2z = p.getZ(c) - p.getZ(a);
    if (e1z * e2x - e1x * e2z < 0) { I[i + 1] = c; I[i + 2] = b; }
  }
  g.index.needsUpdate = true;
}
