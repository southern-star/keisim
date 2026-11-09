// Camera rigs for KeiPilot in the browser (index.html?pilot=mc): keisim CAMERA_RIGS (keisim/config.py) and the
// cylindrical camera of keisim/render/rig.py. A cylindrical camera is drawn as FACES pinhole images side by side
// around its vertical axis and resampled into the cylinder with the maps of rig.py's cylinder_faces, as KeiView draws
// it for training (keisim/render/keiview.py: the faces from src/ego.js, stitched by cv2.remap); remapRGB is
// cv2.remap's 8-bit INTER_LINEAR with BORDER_REPLICATE as OpenCV 5 computes it (float32 weights, rounded to the
// nearest), so the stitched image is the one the model was trained on, to the pixel.

const DEG = Math.PI / 180;

// keisim CameraConfig defaults: 320x160, 0.6 m ahead of the car centre, 1.55 m up, pitched 8 deg down
const camera = (o) => ({ width: 320, height: 160, fov: 100, x: 0.6, y: 0, z: 1.55, pitch: 8, yaw: 0, projection: 'pinhole', ...o });

export const RIGS = {
  front: [camera({ name: 'front' })],
  // a cylindrical 180 deg wide camera and a 50 deg narrow one (pitched 2 deg down), both facing forward
  wide_narrow: [camera({ name: 'wide', projection: 'cylindrical', fov: 180 }), camera({ name: 'narrow', fov: 50, pitch: 2 })],
};
export const FACES = 3, OVERSAMPLE = 1.25;              // keisim CylinderRenderer

/** The pinhole faces of a cylindrical camera and the maps from every cylinder pixel into the atlas of the faces side
 *  by side (rig.py cylinder_faces, float32 as numpy keeps them), shifted half a pixel for GL's pixel centres. */
export function cylinderFaces(cam, n = FACES, oversample = OVERSAMPLE) {
  const W = cam.width, H = cam.height, fov = cam.fov * DEG, f = W / fov;
  const cy = H / 2 - f * Math.tan(cam.pitch * DEG), span = fov / n;
  const tanTop = cy / f, tanBot = (cy - (H - 1)) / f;
  const p = -(Math.atan(tanTop) + Math.atan(tanBot)) / 2;   // faces look down at the middle of the elevations
  const cp = Math.cos(p), sp = Math.sin(p), ff = oversample * f;
  const ks = new Int32Array(W), dxs = new Float64Array(W), dzs = new Float64Array(W);
  for (let u = 0; u < W; u++) {
    const phi = (u - W / 2) / f;
    const k = Math.min(n - 1, Math.max(0, Math.floor((phi + fov / 2) / span)));
    const rel = phi - (k - (n - 1) / 2) * span;
    ks[u] = k; dxs[u] = Math.sin(rel); dzs[u] = Math.cos(rel);
  }
  const yc = new Float64Array(W * H), zc = new Float64Array(W * H);
  let maxT = 0;
  for (let v = 0; v < H; v++) {
    const dy = -(cy - v) / f;
    for (let u = 0; u < W; u++) {
      const i = v * W + u;
      yc[i] = dy * cp - dzs[u] * sp; zc[i] = dy * sp + dzs[u] * cp;
      maxT = Math.max(maxT, Math.abs(yc[i] / zc[i]));
    }
  }
  const Wf = Math.ceil(2 * ff * Math.tan(span / 2) + 2), Hf = Math.ceil(2 * ff * maxT + 2);
  const faces = [];
  for (let i = 0; i < n; i++) {
    const centre = (i - (n - 1) / 2) * span;              // + to the right
    faces.push({ ...cam, projection: 'pinhole', width: Wf, height: Hf, fov: 2 * Math.atan(Wf / 2 / ff) / DEG,
      pitch: p / DEG, yaw: cam.yaw - centre / DEG, name: `${cam.name}_face${i}` });
  }
  const mapX = new Float32Array(W * H), mapY = new Float32Array(W * H), f32 = Math.fround;
  for (let v = 0; v < H; v++) {
    for (let u = 0; u < W; u++) {
      const i = v * W + u;
      mapX[i] = f32(f32(ks[u] * Wf + Wf / 2 + ff * dxs[u] / zc[i]) - 0.5);
      mapY[i] = f32(f32(Hf / 2 + ff * yc[i] / zc[i]) - 0.5);
    }
  }
  return { faces, mapX, mapY, W, H, Wf, Hf };
}

/** cv2.remap(src, mapX, mapY, INTER_LINEAR, borderMode=BORDER_REPLICATE) for 8-bit RGB images of sw x sh (3 bytes per
 *  pixel, top row first): returns remap(src, dst), with the neighbours and weights of every pixel worked out once. */
export function makeRemap(sw, sh, mapX, mapY) {
  const f = Math.fround, n = mapX.length;
  const idx = new Int32Array(4 * n), w = new Float32Array(4 * n);
  for (let i = 0; i < n; i++) {
    const x0 = Math.floor(mapX[i]), y0 = Math.floor(mapY[i]);
    const ax = f(mapX[i] - x0), ay = f(mapY[i] - y0), bx = f(1 - ax), by = f(1 - ay);
    const xa = Math.min(sw - 1, Math.max(0, x0)), xb = Math.min(sw - 1, Math.max(0, x0 + 1));
    const ya = Math.min(sh - 1, Math.max(0, y0)), yb = Math.min(sh - 1, Math.max(0, y0 + 1));
    idx.set([3 * (ya * sw + xa), 3 * (ya * sw + xb), 3 * (yb * sw + xa), 3 * (yb * sw + xb)], 4 * i);
    w.set([bx * by, ax * by, bx * ay, ax * ay], 4 * i);
  }
  const round = (x) => { const r = Math.round(x); return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r; };   // ties to even
  return (src, dst) => {
    for (let i = 0, k = 0, o = 0; i < n; i++, k += 4, o += 3) {
      const a = idx[k], b = idx[k + 1], c = idx[k + 2], d = idx[k + 3];
      const w00 = w[k], w01 = w[k + 1], w10 = w[k + 2], w11 = w[k + 3];
      for (let ch = 0; ch < 3; ch++) {     // float32 sums in OpenCV's order
        const v = f(f(f(f(src[a + ch] * w00) + f(src[b + ch] * w01)) + f(src[c + ch] * w10)) + f(src[d + ch] * w11));
        dst[o + ch] = Math.min(255, Math.max(0, round(v)));
      }
    }
    return dst;
  };
}

/** What a rig draws for one model frame: per camera its shots (pinhole configs) and, for a cylindrical camera, the
 *  maps that stitch them. */
export function rigShots(rig) {
  return RIGS[rig].map((cam) => {
    if (cam.projection === 'cylindrical') {
      const cyl = cylinderFaces(cam);
      return { cam, shots: cyl.faces, cyl };
    }
    return { cam, shots: [cam], cyl: null };
  });
}

/** keisim camera_pose of a pinhole config for the car at (x, y, yaw): the camera centre in KeiSim's world (z up) and
 *  its forward direction, with the vertical field of view of a w x h image (three.js cameras take it). */
export function shotPose(s, x, y, yaw) {
  const c = Math.cos(yaw), sn = Math.sin(yaw);
  const pos = [x + c * s.x - sn * s.y, y + sn * s.x + c * s.y, s.z];
  const a = yaw + s.yaw * DEG, pitch = s.pitch * DEG;
  const fwd = [Math.cos(a) * Math.cos(pitch), Math.sin(a) * Math.cos(pitch), -Math.sin(pitch)];
  const vfov = 2 * Math.atan(Math.tan(s.fov * DEG / 2) * s.height / s.width) / DEG;
  return { pos, fwd, vfov };
}
