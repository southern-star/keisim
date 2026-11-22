// ReconView (index.html?pilot=mc&recon=oracle; docs/recon_view.md): what the model's two cameras show, put together
// in 3D and drawn from behind and above the car. Every model frame becomes two meshes, one per camera: a vertex per
// pixel of the half-resolution depth image (80x160, the pixels (2i, 2j) of the camera image), placed along that
// pixel's line of sight (src/unproject.js) from the pose the frame was drawn at, and textured with the model's input
// image. What the cameras do not see stays dark: nothing is filled in.
// recon=oracle: the depth is the town's own, drawn with the model frame by the depth pass (src/depth.js): the upper
// bound of what a depth head can give, and the tool to measure with (tools/recon_check.mjs).
// It has its own canvas and its own WebGLRenderer: the main renderer's GL state, which the model frames are read back
// from, is not touched. Nothing here is loaded without ?recon.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ENCODE, createDepthPass } from './depth.js';
import { glassMaterials } from './ego.js';
import { shotPose } from './rig.js';
import { cameraRays, depthTable, levelFrame, makeDepthStitch, project, toThree, unproject } from './unproject.js';

const DEG = Math.PI / 180;
// the chase camera, as a rig camera of the car (src/rig.js): 8 m behind the car's centre and 4 m up, looking at the
// road 12 m ahead of it; 2:1 like the main view, 50 deg high. The fidelity is measured from it at 320x160.
export const CHASE = { name: 'chase', projection: 'pinhole', x: -8, y: 0, z: 4, yaw: 0, pitch: Math.atan2(4, 20) / DEG,
  fov: 2 * Math.atan(2 * Math.tan(25 * DEG)) / DEG, width: 320, height: 160 };
export const MAX_R = 80;            // vertices further than this are not drawn [m]
// a triangle whose farthest vertex is `ratio` times its nearest is an object's edge and is not drawn, unless all its
// vertices are ground (see createRecon)
export const RULE = { ground: 'height', ratio: 1.15, scaled: true };
// measure(): the design's first rule (the model's class, one ratio for both cameras) and what was tried against it
export const VARIANTS = { design: { ground: 'class', ratio: 1.15, scaled: false }, class: { ground: 'class', ratio: 1.15, scaled: true },
  both: { ground: 'both', ratio: 1.15, scaled: true }, height: { ground: 'height', ratio: 1.15, scaled: true },
  heightflat: { ground: 'height', ratio: 1.15, scaled: false },
  height110: { ground: 'height', ratio: 1.10, scaled: true }, height125: { ground: 'height', ratio: 1.25, scaled: true } };
const GROUND = new Set([1, 2, 3, 4]);   // road, marking, sidewalk, terrain (keisim SEM_NAMES): triangles of these stay
const NARROW_MARGIN = 4;            // wide triangles inside the narrow camera's image, this many pixels from its edge, go
const STEP = 2;                     // the depth image's resolution: every other pixel, as the shards keep it
const EGO = { L: 4.5, W: 1.85, H: 1.5 };
// measure(): what a pixel of the chase camera's image is, and its colour in snapshot().map
const MAP = { none: 0, good: 1, through: 2, phantom: 3, missed: 4, unseen: 5 };
const MAP_RGB = [[18, 19, 28], [90, 200, 120], [255, 170, 40], [255, 50, 60], [60, 110, 255], [70, 72, 86]];
const GROUND_H = [-0.3, 0.35];      // heights [m] a ground vertex may have (KeiSim is flat; the sidewalk is 0.15 m up)
/** The angle between neighbouring pixels at the image centre [rad]. */
const pixelAngle = (cam) => (cam.projection === 'cylindrical' ? cam.fov * DEG / cam.width : 2 * Math.tan(cam.fov * DEG / 2) / cam.width);

export function createRecon({ ctx, scene, renderer, rig, views, T, mode, renderMain, params }) {
  // The triangle rules. ground: what counts as ground for the triangles that are kept whatever their depths ('class':
  // the model's class, 'height': within GROUND_H of the flat ground, 'both'); ratio: the wide camera's edge ratio;
  // scaled: a camera with finer pixels gets a smaller ratio (the same steepest surface that is still drawn).
  // ?reconground=, ?reconratio=, ?reconscaled=0 change the view's rule; measure() tries VARIANTS on the same frame.
  const rule = { ground: ['class', 'height', 'both'].includes(params.get('reconground')) ? params.get('reconground') : RULE.ground,
    ratio: Number(params.get('reconratio')) || RULE.ratio, scaled: params.has('reconscaled') ? params.get('reconscaled') !== '0' : RULE.scaled };
  const wideAngle = Math.max(...rig.map(({ cam }) => pixelAngle(cam)));
  const table = depthTable();
  const depthPass = createDepthPass({ renderer, scene, glassMats: glassMaterials(ctx) });
  const origin = T.origin;

  // ---------------------------------------------------------------- the model frame's depth (oracle)
  const shotCam = new THREE.PerspectiveCamera();
  function aim(cam, s, x, y, yaw, w, h) {
    const p = shotPose(s, x, y, yaw);
    const [X, Z] = T.P([p.pos[0], p.pos[1]]);
    cam.fov = p.vfov; cam.aspect = w / h; cam.near = 0.1; cam.far = 2500;
    cam.updateProjectionMatrix();
    cam.position.set(X, p.pos[2], Z);
    cam.up.set(0, 1, 0);
    cam.lookAt(X + p.fwd[0], p.pos[2] + p.fwd[2], Z - p.fwd[1]);
    cam.updateMatrixWorld();
  }
  const caps = rig.map(({ cam, shots, cyl }, k) => {
    const w = views[k].w, h = views[k].h, n = shots.length;
    return { cam, shots, w, h, rt: depthPass.target(w, h), rgba: new Uint8Array(4 * w * h), face: new Uint16Array(w * h),
      atlas: new Uint16Array(n * w * h), stitch: cyl ? makeDepthStitch(cyl, 'planar') : null };
  });
  const gRT = depthPass.target(CHASE.width, CHASE.height), gRGBA = new Uint8Array(4 * CHASE.width * CHASE.height);
  const gFace = new Uint16Array(CHASE.width * CHASE.height);
  const flip = (src, w, h, dst, aw = w, x0 = 0) => {            // GL rows (bottom first) to top first, into an atlas
    for (let y = 0; y < h; y++) dst.set(src.subarray((h - 1 - y) * w, (h - y) * w), y * aw + x0);
    return dst;
  };
  let armed = false;

  /** With the model frame, in its scene state (src/pilot.js capture): each camera's depth image (uint16 codes,
   *  320x160, top row first) for the car at (x, y, yaw); armed (measure): also the town from the chase camera. */
  function capture(x, y, yaw) {
    const t0 = performance.now();
    depthPass.begin();
    const dep = caps.map((c) => {
      const aw = c.shots.length * c.w;
      c.shots.forEach((s, j) => {
        aim(shotCam, s, x, y, yaw, c.w, c.h);
        depthPass.draw(shotCam, c.rt, c.w, c.h, c.rgba, c.face);
        flip(c.face, c.w, c.h, c.atlas, aw, j * c.w);
      });
      return c.stitch ? c.stitch(c.atlas, new Uint16Array(c.cam.width * c.cam.height)) : c.atlas.slice();
    });
    let town = null;
    if (armed) {
      aim(shotCam, CHASE, x, y, yaw, CHASE.width, CHASE.height);
      depthPass.draw(shotCam, gRT, CHASE.width, CHASE.height, gRGBA, gFace);
      town = flip(gFace, CHASE.width, CHASE.height, new Uint16Array(gFace.length));
      armed = false;
    }
    depthPass.end();
    return { dep, town, ms: performance.now() - t0 };
  }

  // ---------------------------------------------------------------- the view
  const canvas = document.createElement('canvas');
  canvas.id = 'recon-view';
  canvas.style.cssText = 'position: fixed; z-index: 4; border-radius: 8px; border: 1px solid rgba(255, 255, 255, .35); touch-action: none; cursor: grab;';
  const label = document.createElement('div');
  label.id = 'recon-label';
  label.style.cssText = 'position: fixed; z-index: 5; padding: 3px 8px; border-radius: 6px 0 6px 0; background: rgba(20, 22, 34, .72); color: #c9cbe0; font: 500 11px/1.5 sans-serif; pointer-events: none;';
  label.textContent = 'ReconView · モデルのカメラ 2 台の画像から組み立てた街（深度は正解）';
  document.body.append(canvas, label);
  const rr = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  rr.outputColorSpace = THREE.SRGBColorSpace;
  rr.toneMapping = THREE.NoToneMapping;
  const BG = new THREE.Color('#12131c');
  rr.setClearColor(BG, 1);
  const rscene = new THREE.Scene();
  // the car's frame: +X ahead, +Y up, +Z to its right, the car's centre on the ground at the origin. `world` holds the
  // meshes in the town's three.js coordinates and is moved under the car.
  const world = new THREE.Group();
  rscene.add(world);
  function carFrame(x, y, yaw) {
    const [X, Z] = T.P([x, y]), c = Math.cos(yaw), s = Math.sin(yaw);
    world.rotation.set(0, -yaw, 0);
    world.position.set(-(c * X - s * Z), 0, -(s * X + c * Z));
    world.updateMatrixWorld(true);
  }
  const vfov = 2 * Math.atan(Math.tan(CHASE.fov * DEG / 2) * CHASE.height / CHASE.width) / DEG;
  const chaseCam = (cam) => {
    cam.fov = vfov; cam.aspect = CHASE.width / CHASE.height; cam.near = 0.1; cam.far = 2500;
    cam.updateProjectionMatrix();
    cam.position.set(CHASE.x, CHASE.z, 0);
    cam.lookAt(CHASE.x + Math.cos(CHASE.pitch * DEG), CHASE.z - Math.sin(CHASE.pitch * DEG), 0);
    cam.updateMatrixWorld();
    return cam;
  };
  const viewCam = chaseCam(new THREE.PerspectiveCamera()), measCam = chaseCam(new THREE.PerspectiveCamera());
  const controls = new OrbitControls(viewCam, canvas);
  controls.target.set(CHASE.x + 20, 0, 0);                    // where the chase camera looks at the road
  controls.enablePan = false; controls.minDistance = 4; controls.maxDistance = 120; controls.maxPolarAngle = 89 * DEG;
  controls.update();
  canvas.addEventListener('dblclick', () => { chaseCam(viewCam); controls.update(); });   // back to the chase camera

  // the car: a box of its length and width, and the plan's band (src/pilot.js drawPlan)
  const ego = new THREE.Group();
  const box = new THREE.BoxGeometry(EGO.L, EGO.H, EGO.W);
  ego.add(new THREE.Mesh(box, new THREE.MeshBasicMaterial({ color: 0xdfe6ff, transparent: true, opacity: 0.3, depthWrite: false })));
  ego.add(new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: 0xffffff })));
  ego.position.set(0, EGO.H / 2 + 0.05, 0);
  rscene.add(ego);
  const bandGeo = new THREE.BufferGeometry();
  const bandMat = new THREE.MeshBasicMaterial({ color: 0x28dcff, transparent: true, opacity: 0.45, side: THREE.DoubleSide, depthWrite: false });
  const bandMesh = new THREE.Mesh(bandGeo, bandMat);
  bandMesh.frustumCulled = false; bandMesh.visible = false;
  world.add(bandMesh);
  /** The plan's band: its left and right edges (KeiSim points [[x, y], ...]) and whether a safety layer holds the
   *  target (red); null: no band. */
  function band(edges, held) {
    if (!edges || edges[0].length < 2) { bandMesh.visible = false; return; }
    const [L, R] = edges, n = L.length, pos = new Float32Array(6 * n), idx = [];
    for (let i = 0; i < n; i++) {
      const [lx, lz] = T.P(L[i]), [rx, rz] = T.P(R[i]);
      pos.set([lx, 0.08, lz, rx, 0.08, rz], 6 * i);
      if (i) idx.push(2 * i - 2, 2 * i - 1, 2 * i, 2 * i - 1, 2 * i + 1, 2 * i);
    }
    bandGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    bandGeo.setIndex(idx);
    bandMat.color.set(held ? 0xff3c3c : 0x28dcff);
    bandMesh.visible = true;
  }

  // ---------------------------------------------------------------- the meshes
  const meshes = rig.map(({ cam }, k) => {
    const w = Math.ceil(cam.width / STEP), h = Math.ceil(cam.height / STEP), n = w * h;
    const geo = new THREE.BufferGeometry();
    const position = new THREE.BufferAttribute(new Float32Array(3 * n), 3), uv = new Float32Array(2 * n);
    position.setUsage(THREE.DynamicDrawUsage);
    for (let i = 0, o = 0; i < h; i++) for (let j = 0; j < w; j++, o += 2) { uv[o] = (j * STEP + 0.5) / cam.width; uv[o + 1] = (i * STEP + 0.5) / cam.height; }
    const index = new THREE.BufferAttribute(new Uint16Array(6 * (w - 1) * (h - 1)), 1);
    index.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', position);
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setIndex(index);
    geo.setDrawRange(0, 0);
    const pixels = new Uint8Array(4 * cam.width * cam.height).fill(255);
    const tex = new THREE.DataTexture(pixels, cam.width, cam.height, THREE.RGBAFormat);      // row 0 (the top row) at v = 0
    tex.colorSpace = THREE.SRGBColorSpace; tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter;
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, toneMapped: false }));
    mesh.frustumCulled = false;
    mesh.renderOrder = k;                                     // the narrow camera's mesh over the wide one's where they overlap
    world.add(mesh);
    // wide camera: the vertices the narrow camera sees too (its image without a margin)
    const rays = cameraRays(cam, STEP), inOther = new Uint8Array(n);
    const other = rig.find((r) => r.cam !== cam && r.cam.projection === 'pinhole' && r.cam.fov < cam.fov);
    if (other) {
      const f0 = levelFrame(cam, 0, 0, 0), f1 = levelFrame(other.cam, 0, 0, 0), far = unproject(new Float64Array(n).fill(1000), rays, f0);
      for (let i = 0; i < n; i++) {
        const p = project(other.cam, f1, [far[3 * i], far[3 * i + 1], far[3 * i + 2]]);
        inOther[i] = p.ok && p.u >= NARROW_MARGIN && p.u <= other.cam.width - 1 - NARROW_MARGIN && p.v >= NARROW_MARGIN && p.v <= other.cam.height - 1 - NARROW_MARGIN ? 1 : 0;
      }
    }
    return { cam, w, h, n, geo, position, index, tex, pixels, mesh, rays, inOther, r: new Float64Array(n), P: new Float64Array(3 * n),
      ok: new Uint8Array(n), gClass: new Uint8Array(n), gHeight: new Uint8Array(n) };
  });

  let frames = 0, last = null, stats = null, buildMs = 0;
  /** The triangles of every mesh by a rule; returns the counts per camera. */
  function triangles(ru) {
    return meshes.map((m) => {
      const { cam, w, h, n } = m, idx = m.index.array;
      const st = { camera: cam.name, vertices: n, valid: 0, triangles: 0, noSurface: 0, edge: 0, overlap: 0 };
      for (let v = 0; v < n; v++) st.valid += m.ok[v];
      const ratio = ru.scaled ? 1 + (ru.ratio - 1) * pixelAngle(cam) / wideAngle : ru.ratio;
      const gc = ru.ground !== 'height', gh = ru.ground !== 'class';
      const ground = (v) => (!gc || m.gClass[v]) && (!gh || m.gHeight[v]);
      let c = 0;
      const tri = (a, b, d) => {
        if (!(m.ok[a] && m.ok[b] && m.ok[d])) { st.noSurface++; return; }
        if (m.inOther[a] && m.inOther[b] && m.inOther[d]) { st.overlap++; return; }
        if (!(ground(a) && ground(b) && ground(d))) {
          const lo = Math.min(m.r[a], m.r[b], m.r[d]), hi = Math.max(m.r[a], m.r[b], m.r[d]);
          if (hi > ratio * lo) { st.edge++; return; }
        }
        idx[c++] = a; idx[c++] = b; idx[c++] = d;
      };
      for (let i = 0; i + 1 < h; i++) {
        for (let j = 0; j + 1 < w; j++) {
          const a = i * w + j;
          tri(a, a + w, a + 1); tri(a + 1, a + w, a + w + 1);
        }
      }
      st.triangles = c / 3;
      m.geo.setDrawRange(0, c);
      m.index.needsUpdate = true;
      return st;
    });
  }

  /** A model frame: the pose it was drawn at ({x, y, yaw}), each camera's depth codes (320x160) and input image (RGB,
   *  3 bytes per pixel, top row first), and the model's segmentation of it (K x 40 x 80 class ids). */
  function frame(pose, shot, rgbs, seg) {
    const t0 = performance.now();
    meshes.forEach((m, k) => {
      const { cam, w, h, n } = m, dep = shot.dep[k], sw = seg.length / rig.length / 40, pos = m.position.array;
      for (let i = 0, v = 0; i < h; i++) {
        for (let j = 0; j < w; j++, v++) {
          const r = table[dep[i * STEP * cam.width + j * STEP]];
          m.ok[v] = r > 0 && r <= MAX_R ? 1 : 0;
          m.r[v] = m.ok[v] ? r : 0;
          m.gClass[v] = GROUND.has(Number(seg[k * 40 * sw + (i >> 1) * sw + (j >> 1)])) ? 1 : 0;
        }
      }
      unproject(m.r, m.rays, levelFrame(cam, pose.x, pose.y, pose.yaw), m.P);
      for (let v = 0, o = 0; v < n; v++, o += 3) {
        pos[o] = m.P[o] - origin[0]; pos[o + 1] = m.P[o + 2]; pos[o + 2] = -(m.P[o + 1] - origin[1]);
        m.gHeight[v] = m.P[o + 2] >= GROUND_H[0] && m.P[o + 2] <= GROUND_H[1] ? 1 : 0;
      }
      m.position.needsUpdate = true;
      const rgb = rgbs[k], px = m.pixels;
      for (let i = 0, o = 0; o < px.length; i += 3, o += 4) { px[o] = rgb[i]; px[o + 1] = rgb[i + 1]; px[o + 2] = rgb[i + 2]; }
      m.tex.needsUpdate = true;
    });
    stats = triangles(rule);
    frames++;
    last = { pose: { ...pose }, dep: shot.dep };
    buildMs = performance.now() - t0;
    if (shot.town) measured = measure(pose, shot);
  }

  function reset() {
    for (const m of meshes) m.geo.setDrawRange(0, 0);
    bandMesh.visible = false;
    last = null; stats = null;
  }

  // ---------------------------------------------------------------- fidelity (docs/recon_view.md 6.4)
  const rRT = new THREE.WebGLRenderTarget(CHASE.width, CHASE.height, { type: THREE.UnsignedByteType, samples: 0,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false });
  const rRGBA = new Uint8Array(4 * CHASE.width * CHASE.height);
  const rDepthMat = new THREE.ShaderMaterial({ side: THREE.DoubleSide,
    vertexShader: 'varying vec3 vDepthPos; void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); vDepthPos = mv.xyz; gl_Position = projectionMatrix * mv; }',
    fragmentShader: `varying vec3 vDepthPos; void main(){ ${ENCODE} }` });
  const chaseRays = cameraRays(CHASE);
  let measured = null;

  /** The reconstruction's depth from the chase camera at a pose (codes, top row first), without the car and the band. */
  function reconDepth(pose) {
    carFrame(pose.x, pose.y, pose.yaw);
    const eg = ego.visible, bd = bandMesh.visible;
    ego.visible = false; bandMesh.visible = false;
    rscene.overrideMaterial = rDepthMat;
    rr.setRenderTarget(rRT);
    rr.setClearColor(0x000000, 1);
    rr.clear();
    rr.render(rscene, measCam);
    rr.readRenderTargetPixels(rRT, 0, 0, CHASE.width, CHASE.height, rRGBA);
    rr.setRenderTarget(null);
    rr.setClearColor(BG, 1);
    rscene.overrideMaterial = null;
    ego.visible = eg; bandMesh.visible = bd;
    const W = CHASE.width, H = CHASE.height, out = new Uint16Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0, i = 4 * (H - 1 - y) * W, o = y * W; x < W; x++, i += 4, o++) out[o] = (rRGBA[i] << 8) | rRGBA[i + 1];
    return out;
  }

  /** G (the town) against R (the reconstruction), both from the chase camera at the frame's pose and in the frame's
   *  scene state, for the view's rule and for every rule of VARIANTS. Omega: G within MAX_R. A point of the town is
   *  "seen now" by a camera of the rig when its distance from that camera is, within 3 %, between the nearest and the
   *  farthest of the four depth pixels around where it falls in the camera's image. */
  const BANDS = [10, 30, MAX_R];
  function measure(pose, shot) {
    const G = shot.town, n = G.length;
    const g = new Float64Array(n), seen = new Uint8Array(n);
    for (let i = 0; i < n; i++) g[i] = table[G[i]];
    const P = unproject(g, chaseRays, levelFrame(CHASE, pose.x, pose.y, pose.yaw));
    const cams = rig.map(({ cam }, k) => ({ cam, f: levelFrame(cam, pose.x, pose.y, pose.yaw), dep: shot.dep[k] }));
    for (let i = 0; i < n; i++) {
      if (!(g[i] > 0 && g[i] <= MAX_R)) continue;
      for (const c of cams) {
        const p = project(c.cam, c.f, [P[3 * i], P[3 * i + 1], P[3 * i + 2]]);
        if (!p.ok) continue;
        const u0 = Math.min(c.cam.width - 2, Math.max(0, Math.floor(p.u))), v0 = Math.min(c.cam.height - 2, Math.max(0, Math.floor(p.v)));
        let lo = Infinity, hi = 0;
        for (const o of [v0 * c.cam.width + u0, v0 * c.cam.width + u0 + 1, (v0 + 1) * c.cam.width + u0, (v0 + 1) * c.cam.width + u0 + 1]) {
          const d = table[c.dep[o]];
          if (d > 0) { lo = Math.min(lo, d); hi = Math.max(hi, d); }
        }
        if (hi > 0 && p.r >= 0.97 * lo && p.r <= 1.03 * hi) { seen[i] = 1; break; }
      }
    }
    /** One rule's numbers; map (optional, Uint8Array of n): what every pixel is (MAP). */
    const one = (ru, map) => {
      const st = triangles(ru), R = reconDepth(pose);
      const m = { omega: 0, covered: 0, good: 0, seen: 0, seenCovered: 0, seenGood: 0, drawn: 0, wrong: 0, through: 0, phantom: 0,
        bands: BANDS.map((far) => ({ far, omega: 0, covered: 0, seen: 0, seenCovered: 0 })), triangles: st.map((c) => c.triangles) };
      const errs = [];
      for (let i = 0; i < n; i++) {
        const r = table[R[i]], inOmega = g[i] > 0 && g[i] <= MAX_R;
        // wrong: no surface there in the town or more than 25 % off; `through`: the town's surface is nearer (something
        // the cameras did not see hides this place from the chase camera); `phantom`: the town's is farther or none
        // (a surface that is not there: a triangle stretched between two objects)
        const off = r > 0 && (!(g[i] > 0) || Math.abs(r - g[i]) > 0.25 * g[i]);
        const thru = off && g[i] > 0 && g[i] < r;
        if (r > 0) { m.drawn++; if (off) { m.wrong++; if (thru) m.through++; else m.phantom++; } }
        if (map) map[i] = r > 0 ? (off ? (thru ? MAP.through : MAP.phantom) : MAP.good) : inOmega ? (seen[i] ? MAP.missed : MAP.unseen) : MAP.none;
        if (!inOmega) continue;
        const b = m.bands[g[i] <= BANDS[0] ? 0 : g[i] <= BANDS[1] ? 1 : 2];
        m.omega++; b.omega++;
        if (seen[i]) { m.seen++; b.seen++; }
        if (r > 0) {
          m.covered++; b.covered++;
          if (!off) m.good++;
          if (seen[i]) { m.seenCovered++; b.seenCovered++; if (!off) m.seenGood++; }
          errs.push(Math.abs(r - g[i]) / g[i]);
        }
      }
      errs.sort((a, b) => a - b);
      m.error = errs.length ? errs[errs.length >> 1] : null;
      m.error90 = errs.length ? errs[Math.floor(0.9 * (errs.length - 1))] : null;
      return m;
    };
    const variants = {};
    for (const [name, ru] of Object.entries(VARIANTS)) variants[name] = one(ru);
    lastMap = new Uint8Array(n);
    const m = one(rule, lastMap);                              // last: the meshes are the view's again
    return { ...m, pose: { ...pose }, stats, rule: { ...rule }, variants };
  }
  let lastMap = null;

  // ---------------------------------------------------------------- every frame of the main view
  const size = new THREE.Vector2();
  function layout() {
    const r = renderer.domElement.getBoundingClientRect();
    const w = Math.max(160, Math.round(r.width * 0.4)), h = Math.round(w / 2);
    if (size.x !== w || size.y !== h) { size.set(w, h); rr.setPixelRatio(Math.min(devicePixelRatio || 1, 2)); rr.setSize(w, h, false); }
    canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
    canvas.style.left = `${Math.round(r.left + 12)}px`; canvas.style.top = `${Math.round(r.bottom - h - 12)}px`;
    label.style.left = canvas.style.left; label.style.top = canvas.style.top;
  }
  /** After the main view: draw the reconstruction around the car where it is now. */
  function render(car) {
    layout();
    carFrame(car.x, car.y, car.yaw);
    controls.update();
    rr.render(rscene, viewCam);
  }

  /** Tests (tools/recon_check.mjs): PNG data URLs of the reconstruction and of the town itself, both from the chase
   *  camera at the car (w x h; the town at the main view's size). */
  function snapshot(car, w = 960, h = 480) {
    carFrame(car.x, car.y, car.yaw);
    rr.setPixelRatio(1); rr.setSize(w, h, false);
    rr.render(rscene, measCam);
    const out = { recon: canvas.toDataURL('image/png'), town: null, map: null };
    if (lastMap) {                                            // the last measurement's pixels, by what they are
      const c = document.createElement('canvas');
      c.width = CHASE.width; c.height = CHASE.height;
      const g2 = c.getContext('2d'), im = g2.createImageData(c.width, c.height);
      for (let i = 0; i < lastMap.length; i++) { const col = MAP_RGB[lastMap[i]]; im.data.set([col[0], col[1], col[2], 255], 4 * i); }
      g2.putImageData(im, 0, 0);
      out.map = c.toDataURL('image/png');
    }
    size.set(0, 0);                                           // layout() sets the view's own size again
    if (renderMain) {
      aim(shotCam, CHASE, car.x, car.y, car.yaw, CHASE.width, CHASE.height);
      renderMain(shotCam);
      out.town = renderer.domElement.toDataURL('image/png');
    }
    return out;
  }

  return { mode, capture, frame, band, render, reset, snapshot, rule,
    arm: () => { armed = true; measured = null; },            // measure at the next model frame
    measured: () => measured,
    viewCamera: () => viewCam.position.toArray(),              // in the car's frame (tests)
    state: () => ({ mode, frames, buildMs, stats, pose: last ? last.pose : null }) };
}
