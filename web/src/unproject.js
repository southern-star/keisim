// Depth images of the model's cameras back to the world (ReconView, docs/recon_view.md): keisim/render/rays.py in
// JavaScript, pure functions that run without a browser (tools/unproject_check.mjs compares the two).
// Depth is the straight distance r [m] from the camera centre to the surface, 16 bit: q = round(65535 ln(r / 0.25) /
// ln(1600)), q = 0: no surface. A line of sight is a unit vector in the camera's level frame (x right, y down,
// z forward); cameras are src/rig.js configs (degrees). Pixel (v, u) of a pinhole camera looks through
// (u + 0.5, v + 0.5), of the stitched cylinder through (u, v) (rig.js shifts its maps by half a pixel).

const DEG = Math.PI / 180;
export const DEPTH_NEAR = 0.25, DEPTH_FAR = 400;
const LOG_RANGE = Math.log(DEPTH_FAR / DEPTH_NEAR);
export const PLANAR_TOL = 1e-3;

// numpy's rint and OpenCV's cvRound: halves to the even neighbour
const rint = (x) => { const r = Math.round(x); return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r; };

/** The 16 bit code of a distance r [m]; 0: no surface (r <= 0, not finite, or beyond DEPTH_FAR). */
export function encodeDepth(r) {
  if (!(r > 0) || !(r <= DEPTH_FAR * (1 + 1e-9))) return 0;
  return Math.min(65535, Math.max(1, rint(65535 * Math.log(Math.max(r, 1e-9) / DEPTH_NEAR) / LOG_RANGE)));
}

/** The distance [m] of a 16 bit code; 0: no surface. */
export const decodeDepth = (q) => (q > 0 ? DEPTH_NEAR * Math.exp(q * (LOG_RANGE / 65535)) : 0);

/** Every code's distance: decodeDepth as a table (Float64Array of 65536). */
export function depthTable() {
  const t = new Float64Array(65536);
  for (let q = 1; q < 65536; q++) t[q] = decodeDepth(q);
  return t;
}

const pinhole = (cam) => ({ f: cam.width / 2 / Math.tan(cam.fov * DEG / 2), p: cam.pitch * DEG });
const cylinder = (cam) => { const f = cam.width / (cam.fov * DEG); return { f, cy: cam.height / 2 - f * Math.tan(cam.pitch * DEG) }; };

/** Unit lines of sight of a camera's pixels (step * i, step * j): Float64Array of 3 per pixel, rows of
 *  floor((width + step - 1) / step) pixels (step 2: the pixels cv2.resize INTER_NEAREST keeps at half resolution). */
export function cameraRays(cam, step = 1) {
  const w = Math.ceil(cam.width / step), h = Math.ceil(cam.height / step), out = new Float64Array(3 * w * h);
  const cyl = cam.projection === 'cylindrical';
  const { f, p, cy } = cyl ? cylinder(cam) : pinhole(cam);
  const cp = Math.cos(p || 0), sp = Math.sin(p || 0);
  for (let i = 0, o = 0; i < h; i++) {
    for (let j = 0; j < w; j++, o += 3) {
      const u = j * step, v = i * step;
      let x, y, z;
      if (cyl) { const phi = (u - cam.width / 2) / f; x = Math.sin(phi); y = -(cy - v) / f; z = Math.cos(phi); }
      else { const a = (u + 0.5 - cam.width / 2) / f, b = (v + 0.5 - cam.height / 2) / f; x = a; y = b * cp + sp; z = cp - b * sp; }
      const n = Math.hypot(x, y, z);
      out[o] = x / n; out[o + 1] = y / n; out[o + 2] = z / n;
    }
  }
  return out;
}

/** The camera centre C and its level frame (right, down, fwd: KeiSim world vectors) for the car at (x, y, yaw). */
export function levelFrame(cam, x, y, yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw), a = yaw + (cam.yaw || 0) * DEG;
  return { C: [x + c * cam.x - s * cam.y, y + s * cam.x + c * cam.y, cam.z],
    right: [Math.sin(a), -Math.cos(a), 0], down: [0, 0, -1], fwd: [Math.cos(a), Math.sin(a), 0] };
}

/** KeiSim world points (3 per pixel, into out) of distances r [m] (0: no surface, the point is C) along rays. */
export function unproject(r, rays, frame, out = new Float64Array(3 * r.length)) {
  const { C, right: R, down: D, fwd: F } = frame;
  for (let i = 0, o = 0; i < r.length; i++, o += 3) {
    const dx = r[i] * rays[o], dy = r[i] * rays[o + 1], dz = r[i] * rays[o + 2];
    out[o] = C[0] + dx * R[0] + dy * D[0] + dz * F[0];
    out[o + 1] = C[1] + dx * R[1] + dy * D[1] + dz * F[1];
    out[o + 2] = C[2] + dx * R[2] + dy * D[2] + dz * F[2];
  }
  return out;
}

/** Where a KeiSim world point is in a camera's image: {u, v} in pixel indices (as cameraRays counts them), its
 *  distance r from the camera centre, and ok: in front of the camera and inside the image. */
export function project(cam, frame, P) {
  const { C, right: R, down: D, fwd: F } = frame;
  const px = P[0] - C[0], py = P[1] - C[1], pz = P[2] - C[2];
  const x = px * R[0] + py * R[1] + pz * R[2], y = px * D[0] + py * D[1] + pz * D[2], z = px * F[0] + py * F[1] + pz * F[2];
  const r = Math.hypot(x, y, z);
  let u, v, ok;
  if (cam.projection === 'cylindrical') {
    const { f, cy } = cylinder(cam), h = Math.hypot(x, z);
    u = cam.width / 2 + f * Math.atan2(x, z); v = cy + f * y / Math.max(h, 1e-9); ok = h > 1e-6;
  } else {
    const { f, p } = pinhole(cam), cp = Math.cos(p), sp = Math.sin(p);
    const yc = y * cp - z * sp, zc = y * sp + z * cp;
    ok = zc > 1e-6;
    u = cam.width / 2 + f * x / Math.max(zc, 1e-9) - 0.5; v = cam.height / 2 + f * yc / Math.max(zc, 1e-9) - 0.5;
  }
  ok = ok && u > -0.5 && u < cam.width - 0.5 && v > -0.5 && v < cam.height - 0.5;
  return { u, v, r, ok };
}

/** KeiSim world (x east, y north, z up) to three.js (src/town.js T.P): [x - ox, z, -(y - oy)]. */
export const toThree = (P, origin) => [P[0] - origin[0], P[2], -(P[1] - origin[1])];

/** The depth of a cylindrical camera from the depth images of its faces (rays.py DepthStitcher). cyl: rig.js
 *  cylinderFaces(cam). Returns stitch(atlas, out): atlas Uint16Array of the faces side by side (top row first),
 *  out Uint16Array of cam.width * cam.height.
 *  mode 'nearest': the face pixel cv2.remap INTER_NEAREST picks; 'planar': where the surface around that pixel is
 *  one plane, the distance where the cylinder pixel's own line of sight meets it (exact for planes). */
export function makeDepthStitch(cyl, mode = 'planar', tol = PLANAR_TOL) {
  const { faces, mapX, mapY, Wf, Hf } = cyl, n = mapX.length, aw = faces.length * Wf;
  const ff = Wf / 2 / Math.tan(faces[0].fov * DEG / 2);
  const idx = new Int32Array(n), ax = new Float64Array(n), ay = new Float64Array(n), sPx = new Float64Array(n);
  const inner = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const col = rint(mapX[i]), row = rint(mapY[i]);
    if (col < 0 || col >= aw || row < 0 || row >= Hf) { idx[i] = -1; continue; }
    idx[i] = row * aw + col;
    ax[i] = mapX[i] - col; ay[i] = mapY[i] - row;
    const face = Math.floor(col / Wf), c = col % Wf;
    sPx[i] = Math.hypot((mapX[i] - face * Wf + 0.5 - Wf / 2) / ff, (mapY[i] + 0.5 - Hf / 2) / ff, 1);
    inner[i] = c > 0 && c < Wf - 1 && row > 0 && row < Hf - 1 ? 1 : 0;
  }
  const sAtlas = new Float64Array(aw * Hf);
  for (let y = 0; y < Hf; y++) for (let x = 0; x < aw; x++) sAtlas[y * aw + x] = Math.hypot(((x % Wf) + 0.5 - Wf / 2) / ff, (y + 0.5 - Hf / 2) / ff, 1);
  const table = depthTable();
  const w = (atlas, k) => (atlas[k] > 0 ? sAtlas[k] / table[atlas[k]] : 0);       // 1/z; 0: no surface
  return (atlas, out) => {
    for (let i = 0; i < n; i++) {
      const k = idx[i];
      if (k < 0) { out[i] = 0; continue; }
      out[i] = atlas[k];
      if (mode !== 'planar' || !inner[i] || !atlas[k]) continue;
      const w0 = w(atlas, k), xm = w(atlas, k - 1), xp = w(atlas, k + 1), ym = w(atlas, k - aw), yp = w(atlas, k + aw);
      if (!(xm > 0 && xp > 0 && ym > 0 && yp > 0)) continue;
      if (Math.abs(xm + xp - 2 * w0) > tol * w0 || Math.abs(ym + yp - 2 * w0) > tol * w0) continue;
      const wp = w0 + (ax[i] >= 0 ? xp - w0 : w0 - xm) * ax[i] + (ay[i] >= 0 ? yp - w0 : w0 - ym) * ay[i];
      if (wp > 0) out[i] = encodeDepth(sPx[i] / wp);
    }
    return out;
  };
}
