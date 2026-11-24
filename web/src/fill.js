// ReconFill (index.html?pilot=mc&recon=model&fill=oracle; docs/recon_fill.md): what ReconView (src/recon.js) leaves
// dark is filled in. The fill is a third camera image: colour (320x160) and depth (the pixels (2i, 2j), metres, 0: no
// surface) from the chase camera at the pose of a model frame. It becomes a mesh the same way the cameras' images do
// (a vertex per depth pixel along its line of sight, the same triangle rule), so the view and the measurement take it
// as they take the cameras' meshes, and what is seen and what is filled in stay apart, pixel by pixel.
// fill=oracle: the image is the town's own, drawn with the model frame (src/pilot.js: the colour, by the model
// frames' pipeline; src/recon.js capture: the depth). The upper bound of what a model can fill in, the tool to measure
// with (tools/recon_check.mjs --fill oracle) and to write training data with (tools/fill_dump.mjs).
// fill=model: the image is a small model's (keipilot/fillnet.py, models/reconfill.onnx), from the view's own picture
// without a fill (colour and depth from the chase camera) and, for models with the input, the current frame alone by
// the loose triangle rule. It runs after the driving model has answered, in the worker src/pilot.js gives it, and
// the fill shows when it answers; a turn that comes while it is still running is skipped. A measurement also makes
// the town's own fill of that frame, to compare (under "oracle").
// A fill is made every FILL.every model frames and replaces the one before: fills are not accumulated.
// &fillshow=holes (the default): of the fill's mesh only the triangles that cover a pixel where the view, from the
// chase camera at that frame, drew nothing; behind what the cameras saw (a larger polygon offset than the keyframes').
// &fillshow=all: all of the fill's mesh, and the cameras' meshes and the keyframes hidden.
// &filltint=1: the fill's surfaces tinted, to tell them from the seen ones.
// Nothing here is loaded without ?fill.
import * as THREE from 'three';
import { CHASE, GROUND_H, MAP, MAX_R, STEP } from './recon.js';
import { cameraRays, levelFrame, unproject } from './unproject.js';

// every: a fill with every n-th model frame; offset: the fill's polygon offset (factor, units; the keyframes' are 2
// and 50..440); tint: &filltint=1; loose: the triangle rule of a training sample's second input (the current frame
// alone with a wide edge ratio: surfaces the cameras see but the view's rule cuts)
export const FILL = { every: 2, offset: [4, 600], tint: 0xff8ad8, loose: { ground: 'height', ratio: 2.0, scaled: false } };

// depths within 10 % of each other, as a difference of 16 bit codes (65535 over ln(400 / 0.25))
const FLICKER_CODES = Math.round(65535 * Math.log(1.1) / Math.log(1600));

export function createFill({ world, origin, table, rule, params, label, rr, rscene, measCam, carFrame, meshTriangles, triangles, reconDepth,
  keyframes, observed, bare, session = null }) {
  const mode = params.get('fill');
  const loose = !!session && session.inputNames.includes('loose_rgb');     // the model takes the loose rule's picture
  const show = params.get('fillshow') === 'all' ? 'all' : 'holes';
  const tint = params.get('filltint') === '1';
  const W = CHASE.width, H = CHASE.height, w = Math.ceil(W / STEP), h = Math.ceil(H / STEP), n = w * h;
  const rays = cameraRays(CHASE, STEP);
  label.textContent += mode === 'oracle' ? ' · 見えていない所は正解で補完（開発用）' : ' · 見えていない所はモデルが補完';

  // two meshes: the fill that is shown and the one before it (for the measurement and the training samples)
  function slot() {
    const geo = new THREE.BufferGeometry();
    const position = new THREE.BufferAttribute(new Float32Array(3 * n), 3), uv = new Float32Array(2 * n);
    position.setUsage(THREE.DynamicDrawUsage);
    for (let i = 0, o = 0; i < h; i++) for (let j = 0; j < w; j++, o += 2) { uv[o] = (j * STEP + 0.5) / W; uv[o + 1] = (i * STEP + 0.5) / H; }
    const index = new THREE.BufferAttribute(new Uint16Array(6 * (w - 1) * (h - 1)), 1);
    index.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', position);
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setIndex(index);
    geo.setDrawRange(0, 0);
    const pixels = new Uint8Array(4 * W * H).fill(255);
    const tex = new THREE.DataTexture(pixels, W, H, THREE.RGBAFormat);          // row 0 (the top row) at v = 0
    tex.colorSpace = THREE.SRGBColorSpace; tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter;
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, toneMapped: false,
      color: tint ? FILL.tint : 0xffffff, polygonOffset: true, polygonOffsetFactor: FILL.offset[0], polygonOffsetUnits: FILL.offset[1] }));
    mesh.frustumCulled = false; mesh.renderOrder = -2; mesh.visible = false;
    world.add(mesh);
    // the fields src/recon.js meshTriangles reads; the fill has no classes: ground is what is at the ground's height
    const ok = new Uint8Array(n), gHeight = new Uint8Array(n);
    return { cam: CHASE, w, h, n, geo, position, index, tex, pixels, mesh, ok, gClass: gHeight, gHeight, inOther: new Uint8Array(n),
      r: new Float64Array(n), P: new Float64Array(3 * n), full: new Uint16Array(index.array.length), count: 0, hole: new Uint8Array(n),
      shown: null, seq: 0, frame: 0, pose: null, stats: null };
  }
  let cur = slot(), prev = slot();
  const orc = mode === 'model' ? slot() : null;             // the town's own fill of a measured frame, to compare
  let since = Infinity, seq = 0, frameNo = 0, made = 0, ms = 0, maps = {}, sampleArmed = false, sample = null;
  let gen = 0, running = null, skipped = 0, prepMs = 0, inferMs = 0;

  /** Of a fill's triangles (by the rule) the ones that are drawn: all, or those of the cells that cover a hole. */
  function apply(s, how) {
    const idx = s.index.array, full = s.full;
    let c = 0;
    for (let i = 0; i < s.count; i += 3) {
      // the two triangles of the cell at vertex a are (a, a + w, a + 1) and (a + 1, a + w, a + w + 1)
      if (how === 'all' || s.hole[full[i + 1] - w]) { idx[c++] = full[i]; idx[c++] = full[i + 1]; idx[c++] = full[i + 2]; }
    }
    s.index.needsUpdate = true;
    s.geo.setDrawRange(0, c);
    s.shown = how;
  }

  /** What is drawn: 'none' (the view without a fill), 'view' (as the page shows it), or {slot, how, obs}: this fill,
   *  its triangles ('holes' or 'all'), with or without what the cameras saw. */
  function layers(sel) {
    if (sel === 'none') sel = { slot: null, obs: true };
    else if (sel === 'view') sel = { slot: cur, how: show, obs: show !== 'all' || !cur.seq };
    observed(sel.obs);
    for (const s of orc ? [cur, prev, orc] : [cur, prev]) s.mesh.visible = s === sel.slot && s.seq > 0;
    if (sel.slot && sel.slot.seq > 0 && sel.slot.shown !== sel.how) apply(sel.slot, sel.how);
  }

  // the view's colours from the chase camera, as the measurement's depth is drawn (src/recon.js reconDepth): no
  // antialiasing, so a pixel has a colour exactly where it has a depth
  const cRT = new THREE.WebGLRenderTarget(W, H, { type: THREE.UnsignedByteType, samples: 0, minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter, generateMipmaps: false, colorSpace: THREE.SRGBColorSpace });
  const cRGBA = new Uint8Array(4 * W * H);
  /** What is shown now, from the chase camera at a pose: RGB, 3 bytes per pixel, top row first. */
  function colour(pose) {
    carFrame(pose.x, pose.y, pose.yaw);
    bare(() => {
      rr.setRenderTarget(cRT);
      rr.render(rscene, measCam);
      rr.readRenderTargetPixels(cRT, 0, 0, W, H, cRGBA);
      rr.setRenderTarget(null);
    });
    const out = new Uint8Array(3 * W * H);
    for (let y = 0, o = 0; y < H; y++) for (let x = 0, i = 4 * (H - 1 - y) * W; x < W; x++, i += 4, o += 3) { out[o] = cRGBA[i]; out[o + 1] = cRGBA[i + 1]; out[o + 2] = cRGBA[i + 2]; }
    return out;
  }

  /** Is this model frame a fill's turn? Returns {make, truth}: a fill is made from it; it needs the town itself
   *  (src/pilot.js then draws the chase camera's image with the frame). force: a measurement is armed. */
  function turn(force) {
    frameNo++;
    const due = force || sampleArmed || since >= FILL.every;
    since = due ? 1 : since + 1;
    return { make: due, truth: due && (mode === 'oracle' || force || sampleArmed) };
  }

  /** A slot's mesh from a fill's image at pose: metres(v), the distance of depth pixel v [m] (0: no surface), and
   *  pixel(o), the colour byte o of the RGB image (320x160); input: the view's depth codes without a fill. */
  function build(s, pose, metres, pixel, input) {
    const pos = s.position.array;
    for (let v = 0; v < n; v++) {
      const r = metres(v);
      s.ok[v] = r > 0 && r <= MAX_R ? 1 : 0;
      s.r[v] = s.ok[v] ? r : 0;
    }
    unproject(s.r, rays, levelFrame(CHASE, pose.x, pose.y, pose.yaw), s.P);
    for (let v = 0, o = 0; v < n; v++, o += 3) {
      pos[o] = s.P[o] - origin[0]; pos[o + 1] = s.P[o + 2]; pos[o + 2] = -(s.P[o + 1] - origin[1]);
      s.gHeight[v] = s.P[o + 2] >= GROUND_H[0] && s.P[o + 2] <= GROUND_H[1] ? 1 : 0;
    }
    s.position.needsUpdate = true;
    for (let i = 0, o = 0; o < s.pixels.length; i += 3, o += 4) { s.pixels[o] = pixel(i); s.pixels[o + 1] = pixel(i + 1); s.pixels[o + 2] = pixel(i + 2); }
    s.tex.needsUpdate = true;
    s.stats = meshTriangles(s, rule);
    s.count = s.geo.drawRange.count;
    s.full.set(s.index.array.subarray(0, s.count));
    // a cell covers a hole when the view drew nothing at one of the pixels between its four vertices
    let holes = 0;
    for (let i = 0; i + 1 < h; i++) {
      for (let j = 0; j + 1 < w; j++) {
        let hol = 0;
        for (let y = STEP * i; y <= STEP * (i + 1) && y < H && !hol; y++) for (let x = STEP * j; x <= STEP * (j + 1) && x < W; x++) if (input[y * W + x] === 0) { hol = 1; break; }
        s.hole[i * w + j] = hol;
        holes += hol;
      }
    }
    s.stats.holeCells = holes;
    s.shown = null; s.seq = ++seq; s.pose = { ...pose };
  }
  /** The town's own image as a slot's mesh. */
  const fromTown = (s, pose, shot, input) => build(s, pose, (v) => table[shot.town[Math.floor(v / w) * STEP * W + (v % w) * STEP]], (o) => shot.fillRgb[o], input);

  /** A fill from a model frame at pose; the one before it is kept as `prev`. fill=oracle: from shot.town (the town's
   *  depth codes from the chase camera, 320x160) and shot.fillRgb (its colours, RGB), at once. fill=model: from the
   *  model's answer; returns a promise that ends when the fill is there (or was dropped). */
  function make(pose, shot) {
    const t0 = performance.now(), frame = frameNo;
    if (mode === 'model' && running) { skipped++; return null; }   // the model is still at the fill before
    if (mode !== 'model' && !(shot.town && shot.fillRgb)) return null;
    layers('none');
    const input = reconDepth(pose);                           // what the view draws without a fill
    if (mode !== 'model') {
      [cur, prev] = [prev, cur];
      fromTown(cur, pose, shot, input);
      cur.frame = frame;
      made++;
      if (sampleArmed) take(pose, shot, input);
      layers('view');
      ms = performance.now() - t0;
      return null;
    }
    if (orc && shot.town && shot.fillRgb) { fromTown(orc, pose, shot, input); orc.frame = frame; }
    else if (orc) orc.seq = 0;
    // the model's input: the view's colours and depth and, for models with it, the loose rule's picture
    const chw = (rgb) => { const out = new Float32Array(3 * W * H), px = W * H; for (let i = 0, o = 0; i < px; i++, o += 3) { out[i] = rgb[o]; out[px + i] = rgb[o + 1]; out[2 * px + i] = rgb[o + 2]; } return out; };
    const feeds = { in_rgb: { data: chw(colour(pose)), dims: [1, 3, H, W] }, in_dep: { data: Float32Array.from(input), dims: [1, 1, H, W] } };
    if (loose) {
      observed(true, false);                                  // the current frame alone
      triangles(FILL.loose);
      feeds.loose_dep = { data: Float32Array.from(reconDepth(pose)), dims: [1, 1, H, W] };
      feeds.loose_rgb = { data: chw(colour(pose)), dims: [1, 3, H, W] };
      triangles(rule);
    }
    layers('view');
    prepMs = performance.now() - t0;
    const g = gen, t1 = performance.now();
    running = session.run(feeds).then((out) => {
      inferMs = performance.now() - t1;
      if (g !== gen) return;                                  // not after a reset
      const t2 = performance.now(), depth = out.depth.data, rgb = out.rgb.data, px = W * H;
      [cur, prev] = [prev, cur];
      build(cur, pose, (v) => depth[v], (o) => rgb[(o % 3) * px + Math.floor(o / 3)], input);
      cur.frame = frame;
      made++;
      layers('view');
      ms = performance.now() - t2;
    }).catch((e) => { console.error('[fill] the fill model failed', e); }).finally(() => { running = null; });
    return running;
  }

  /** A training sample of this frame (tools/fill_dump.mjs): the view without a fill (colour and depth), the current
   *  frame alone by the loose rule, the town itself (the fill's image), and the fill before this one drawn from this
   *  frame's chase camera. Depths are codes, images 320x160, top row first. */
  function take(pose, shot, input) {
    sampleArmed = false;
    layers('none');
    const inRgb = colour(pose);
    observed(true, false);                                    // the current frame alone
    triangles(FILL.loose);
    const looseDep = reconDepth(pose), looseRgb = colour(pose);
    triangles(rule);
    let prevDep = null, prevRgb = null;
    if (prev.seq) {
      layers({ slot: prev, how: 'all', obs: false });
      prevDep = reconDepth(pose); prevRgb = colour(pose);
    }
    sample = { pose: { ...pose }, keyframes: keyframes(), prevPose: prev.seq ? { ...prev.pose } : null, prevFrames: prev.seq ? cur.frame - prev.frame : 0,
      in_rgb: inRgb, in_dep: input, loose_rgb: looseRgb, loose_dep: looseDep, gt_rgb: Uint8Array.from(shot.fillRgb), gt_dep: Uint16Array.from(shot.town),
      prev_rgb: prevRgb, prev_dep: prevDep };
  }
  const b64 = (a) => {
    const u8 = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s);
  };

  /** The same frame with the fill (src/recon.js measure): what the view draws with this frame's fill and with the one
   *  before it (what a visitor sees until the next fill), the holes only and all of the fill alone. RP: the view's
   *  depth without a fill; score: the measurement of a depth image. `added`: the pixels the fill draws where the view
   *  drew nothing. */
  function measure(pose, RP, score) {
    maps = {};
    const one = (s, how, obs, keep) => {
      if (!s.seq) return null;
      layers({ slot: s, how, obs });
      const R = reconDepth(pose), map = new Uint8Array(R.length), m = score(R, map);
      const a = { added: 0, addedGood: 0, addedThrough: 0, addedPhantom: 0, triangles: s.geo.drawRange.count / 3 };
      for (let i = 0; i < R.length; i++) {
        if (RP[i] !== 0 || R[i] === 0) continue;
        a.added++;
        if (map[i] === MAP.good) a.addedGood++; else if (map[i] === MAP.through) a.addedThrough++; else a.addedPhantom++;
      }
      if (keep) maps[keep] = map;
      return { ...m, ...a };
    };
    if (!cur.seq) { layers('view'); return null; }            // no fill yet (the model's first answer is still out)
    const out = { mode, show, every: FILL.every, stats: cur.stats, ms,
      age: prev.seq ? { frames: cur.frame - prev.frame, metres: Math.hypot(cur.pose.x - prev.pose.x, cur.pose.y - prev.pose.y) } : null,
      holes: one(cur, 'holes', true, 'holes'), holesPrev: one(prev, 'holes', true, 'holesPrev'),
      all: one(cur, 'all', false, 'all'), allPrev: one(prev, 'all', false, 'allPrev') };
    // flicker: this fill and the one before, each alone, from this frame's chase camera; where the view has a hole
    // and the two fills have the same surface (depths within 10 %), how far apart their colours are
    if (prev.seq) {
      const alone = (s) => { layers({ slot: s, how: 'all', obs: false }); return { R: reconDepth(pose), C: colour(pose) }; };
      const a = alone(cur), b = alone(prev);
      let nf = 0, sum = 0, both = 0;
      for (let i = 0, o = 0; i < RP.length; i++, o += 3) {
        if (RP[i] !== 0 || a.R[i] === 0 || b.R[i] === 0) continue;
        both++;
        if (Math.abs(a.R[i] - b.R[i]) > FLICKER_CODES) continue;
        nf++;
        sum += (Math.abs(a.C[o] - b.C[o]) + Math.abs(a.C[o + 1] - b.C[o + 1]) + Math.abs(a.C[o + 2] - b.C[o + 2])) / 3;
      }
      out.flicker = { pixels: nf, both, sum };
    }
    if (mode === 'model') {
      Object.assign(out, { prepMs, inferMs, skipped, fresh: cur.frame === frameNo });
      // the town's own fill of the same frame, from the same input
      if (orc.seq && orc.frame === frameNo) out.oracle = { stats: orc.stats, holes: one(orc, 'holes', true, 'oracleHoles'), all: one(orc, 'all', false, 'oracleAll') };
    }
    layers('view');
    return out;
  }

  /** Tests (src/recon.js snapshot): pictures of the fill from the chase camera. render(): the view's canvas as a PNG;
   *  mapPng(map): a measurement's pixels by what they are. */
  function snapshot(render, mapPng) {
    const out = {};
    const pic = (s, how, obs, tinted = tint) => {
      if (!s.seq) return null;
      layers({ slot: s, how, obs });
      s.mesh.material.color.set(tinted ? FILL.tint : 0xffffff);
      const png = render();
      s.mesh.material.color.set(tint ? FILL.tint : 0xffffff);
      return png;
    };
    out.holes = pic(cur, 'holes', true); out.holesTint = pic(cur, 'holes', true, true); out.all = pic(cur, 'all', false);
    out.holesPrev = pic(prev, 'holes', true); out.allPrev = pic(prev, 'all', false);
    if (orc && orc.seq) { out.oracleHoles = pic(orc, 'holes', true); out.oracleAll = pic(orc, 'all', false); }
    for (const [k, map] of Object.entries(maps)) out[`${k}Map`] = mapPng(map);
    const image = (s) => {                                    // a fill's image itself
      const c = document.createElement('canvas');
      c.width = W; c.height = H;
      c.getContext('2d').putImageData(new ImageData(Uint8ClampedArray.from(s.pixels), W, H), 0, 0);
      return c.toDataURL('image/png');
    };
    if (cur.seq) out.image = image(cur);
    if (orc && orc.seq) out.oracleImage = image(orc);
    layers('view');
    return out;
  }

  function reset() {
    for (const s of orc ? [cur, prev, orc] : [cur, prev]) { s.seq = 0; s.count = 0; s.shown = null; s.geo.setDrawRange(0, 0); s.mesh.visible = false; }
    since = Infinity; maps = {}; sample = null; gen++;
    observed(true);
  }

  return { mode, show, camera: CHASE, turn, make, measure, snapshot, layers, reset,
    idle: () => running || Promise.resolve(),                 // the fill model's answer in flight (tests)
    armSample: () => { sampleArmed = mode === 'oracle'; sample = null; },   // a training sample with the next model frame (fill=oracle)
    /** The armed sample, its images as base64 (uint8, uint16 little-endian), or null. */
    takeSample: () => {
      if (!sample) return null;
      const out = {};
      for (const [k, v] of Object.entries(sample)) out[k] = ArrayBuffer.isView(v) ? b64(v) : v;
      sample = null;
      return out;
    },
    state: () => ({ mode, show, tint, every: FILL.every, made, ms, triangles: cur.geo.drawRange.count / 3, cells: cur.stats ? cur.stats.holeCells : 0,
      ...(mode === 'model' ? { loose, where: session.where, prepMs, inferMs, skipped } : {}) }) };
}
