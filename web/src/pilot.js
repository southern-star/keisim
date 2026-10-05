// KeiPilot in the browser (index.html?pilot=1): KeiSim's closed loop, with the town drawn by KeiView. Ten times a
// second the car camera is rendered at the model's 320x160 exactly as in ego mode (src/ego.js), the driving model
// (scripts/export_onnx.py -> models/keipilot.onnx, run by onnxruntime-web on WebGPU or WASM) plans a path and a
// target speed from that frame, and the car follows the plan with KeiSim's controllers and vehicle model
// (keisim/control.py, keisim/vehicle.py) behind the two safety layers of keipilot/agent.py. The route and the
// model's route inputs (next turn, target point) come from src/nav.js. Signals switch with KeiSim's timing; other
// cars (src/traffic.js) drive around the town unless switched off (?traffic=0, the HUD button or T, remembered).
// Keys: ← ↑ → turn at the next junction, Space pause, R back to the start, T other cars (or the buttons).
import * as THREE from 'three';
import { buildLanes, createNavigator } from './nav.js';
import { createTraffic } from './traffic.js';

const ORT_VERSION = '1.20.1';
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
// keisim CameraConfig: 0.6 m ahead of the car centre, 1.55 m up, pitched 8 deg down, 100 deg wide, 320x160
const CAM = { ahead: 0.6, height: 1.55, pitch: 8 * Math.PI / 180, hfov: 100 * Math.PI / 180, W: 320, H: 160 };
const VFOV = 2 * Math.atan(Math.tan(CAM.hfov / 2) * CAM.H / CAM.W) * 180 / Math.PI;
const DT = 0.05, CONTROL_EVERY = 2, CONTROL_DT = DT * CONTROL_EVERY;   // keisim EnvConfig: 20 Hz physics, 10 Hz control
const ROUTE_DEVIATION = 6.0, BLOCKED_TIMEOUT = 90.0;                    // keisim EnvConfig terminations
const TURN_JA = { left: '左折', straight: '直進', right: '右折' };
const TL_NAMES = ['赤', '黄', '青', 'なし'];
const TL_CSS = ['#ff5a4e', '#ffc93c', '#4fd18b', '#c8c6d0'];
// keisim/config.py SEM_COLORS (BGR there) as RGB
const SEM_RGB = [[70, 130, 180], [128, 64, 128], [255, 255, 255], [244, 35, 232], [152, 251, 152], [70, 70, 70],
  [107, 142, 35], [153, 153, 153], [255, 0, 0], [255, 220, 0], [0, 255, 0], [0, 0, 142], [220, 20, 60]];
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// ------------------------------------------------------------------ model inputs (keipilot/model.py)
function condFeatures(cmd, tp) {
  const d = Math.hypot(tp[0], tp[1]);
  return Float32Array.from([cmd === 0 ? 1 : 0, cmd === 1 ? 1 : 0, cmd === 2 ? 1 : 0, clamp(tp[0] / 40, -3, 3),
    clamp(tp[1] / 40, -3, 3), Math.log1p(d) / 4, Math.atan2(tp[1], tp[0]) / 3.1416]);
}
function speedFeatures(v) {
  v = clamp(v, 0, 20);
  return Float32Array.from([v / 10, (v / 10) ** 2, Math.log1p(v) / 2.5]);
}

// ------------------------------------------------------------------ vehicle and controllers (keisim)
const VEH = { L: 2.7, LF: 1.35, LR: 1.35, MAX_STEER: 0.6, STEER_TAU: 0.12, A_MAX: 3.5, B_MAX: 8.0 };
const maxAccel = (v) => VEH.A_MAX * Math.max(0.25, 1 - v / 25);
const resistance = (v) => (v > 0.05 ? 0.12 : 0) + 0.02 * v + 0.0015 * v * v;

/** keisim EgoVehicle.step: kinematic bicycle with steering lag and a simple powertrain. */
function stepVehicle(car, [steer, throttle, brake], dt) {
  car.steer += (steer * VEH.MAX_STEER - car.steer) * Math.min(1, dt / VEH.STEER_TAU);
  const a = throttle * maxAccel(car.v) - brake * VEH.B_MAX - resistance(car.v);
  const v = Math.max(0, car.v + a * dt), vm = 0.5 * (car.v + v);
  const beta = Math.atan(VEH.LR / (VEH.LF + VEH.LR) * Math.tan(car.steer));
  car.x += vm * Math.cos(car.yaw + beta) * dt;
  car.y += vm * Math.sin(car.yaw + beta) * dt;
  car.yaw += vm / VEH.LR * Math.sin(beta) * dt;
  car.v = v;
}

/** keisim PurePursuit; path: ego-frame points of the car centre. */
function purePursuit(path, v) {
  const ld = clamp(3.5 + 0.45 * v, 3.5, 11);
  let ax = 0, ay = 0, da = 0, tx = null, ty = 0;
  for (const [px, py] of path) {
    const bx = px + VEH.LR, by = py;                       // reference: rear axle
    if (bx <= 0) continue;                                 // behind it (the plan is a few frames old)
    const db = Math.hypot(bx, by);
    if (db >= ld) {
      const t = clamp((ld - da) / Math.max(db - da, 1e-6), 0, 1);
      tx = ax + (bx - ax) * t; ty = ay + (by - ay) * t;
      break;
    }
    ax = bx; ay = by; da = db;
  }
  if (tx === null) { tx = ax; ty = ay; }
  const curv = 2 * ty / Math.max(tx * tx + ty * ty, 1e-3);
  return clamp(Math.atan(curv * VEH.L) / VEH.MAX_STEER, -1, 1);
}

/** keisim SpeedController: inverse-dynamics feed-forward + PI feedback. Returns [throttle, brake]. */
function createSpeedController() {
  let i = 0;
  return {
    reset() { i = 0; },
    control(target, v, dt) {
      let aDes;
      if (target < 0.05) {
        i = 0;
        if (v < 0.3) return [0, 1];
        aDes = -Math.min(6.5, Math.max(4, 2 * v));
      } else {
        const e = target - v;
        i = clamp(i + e * dt, -2, 2);
        aDes = clamp((e > 0 ? 1.5 : 3.0) * e + 0.3 * i, -6.5, 2.5);
      }
      const need = aDes + resistance(v);
      return need >= 0 ? [clamp(need / maxAccel(v), 0, 1), 0] : [0, clamp(-need / VEH.B_MAX, 0, 1)];
    },
  };
}

/** keipilot/agent.py BrakeHold (no throttle burst right after firm braking) and RedHold. */
function createSafety() {
  let left = 0, cap = null;
  return {
    reset() { left = 0; cap = null; },
    apply(raw, v, dt, tl) {
      if (raw < v - 1.5) { left = 1.5; cap = cap === null ? raw : Math.min(cap, raw); }
      else if (left > 0) { left -= dt; cap += 0.5 * dt; }
      let t = raw;
      if (left <= 0) cap = null; else t = Math.min(raw, cap);
      if (v < 1.0 && raw < 1.0 && tl[0] + tl[1] >= 0.8) t = 0;     // RedHold, judged on the model's own target
      return t;
    },
  };
}

// ------------------------------------------------------------------ loading
async function fetchModel(url, progress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const total = Number(res.headers.get('Content-Length')) || 0;
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    progress(got, total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/** A hardware WebGPU adapter, or null. Software ones (SwiftShader) are skipped: WASM is far faster than those. */
async function gpuAdapter() {
  if (!navigator.gpu) return null;
  for (const opt of [{ powerPreference: 'high-performance' }, {}, { powerPreference: 'low-power' }]) {
    try {
      const a = await navigator.gpu.requestAdapter(opt);
      const info = (a && a.info) || {};
      if (a && !a.isFallbackAdapter && !info.isFallbackAdapter && !/swiftshader/i.test(`${info.vendor} ${info.architecture}`)) return a;
    } catch (e) { /* try the next preference */ }
  }
  return null;
}

export async function installPilot({ ctx, scene, renderer, camera, sunDir, modelPipeline, params, status, progress = () => {} }) {
  const T = ctx.town;
  const lanes = buildLanes(T.data);
  const nav = createNavigator(T.data, lanes);
  // surrounding traffic (src/traffic.js): on by default, off with ?traffic=0 or the HUD button / T (the choice is
  // remembered in this browser); ?cars=N cars
  const roadKm = lanes.reduce((a, l) => a + l.len, 0) / 1000;
  const traffic = createTraffic(T.data, lanes, { count: Number(params.get('cars')) || Math.min(40, Math.round(roadKm * 12)), seed: 7 });
  const TRAFFIC_KEY = 'keipilot.traffic';
  let stored = null;
  try { stored = localStorage.getItem(TRAFFIC_KEY); } catch (e) { /* storage blocked: default on */ }
  let trafficOn = (params.has('traffic') ? params.get('traffic') : stored) !== '0';
  const want = params.get('ep');                            // force 'wasm' or 'webgpu'
  const adapter = want === 'wasm' ? null : await gpuAdapter();
  const gpu = want === 'webgpu' || !!adapter;
  status('推論エンジンを読み込み中…');
  const ort = await import(`${ORT_URL}${gpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs'}`);
  ort.env.wasm.wasmPaths = ORT_URL;
  if (adapter) ort.env.webgpu.adapter = adapter;
  if (!gpu) ort.env.wasm.proxy = true;                      // CPU inference in a worker, off the render loop
  const MB = (n) => (n / 1e6).toFixed(0);
  const bytes = await fetchModel(params.get('model') || 'models/keipilot.onnx',
    (got, total) => { status(`運転モデルを読み込み中… ${MB(got)}${total >= got ? ` / ${MB(total)}` : ''} MB`); if (total) progress(0.9 * got / total); });
  status('運転モデルを準備中…');
  let session = null, backend = '';
  for (const ep of gpu ? ['webgpu', 'wasm'] : ['wasm']) {
    try { session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] }); backend = ep; break; } catch (e) { console.warn(`[pilot] ${ep} unavailable`, e); }
  }
  if (!session) throw new Error('推論エンジンを起動できませんでした');

  // ---------------------------------------------------------------- page elements
  const overlay = document.createElement('canvas');
  overlay.id = 'pilot-overlay';
  document.body.appendChild(overlay);
  const o2d = overlay.getContext('2d');
  const hud = document.createElement('div');
  hud.id = 'pilot-hud';
  hud.innerHTML = `<div><b>KeiPilot</b> <span class="ep" data-k="ep"></span></div>
    <div>速度 <b data-k="v">0</b> km/h　目標 <b data-k="t">–</b> km/h</div>
    <div>信号 <b data-k="tl">–</b> <span data-k="tlp"></span></div>
    <div>次の交差点 <b data-k="cmd">–</b> <span data-k="dist"></span></div>
    <div class="btns"><button data-turn="left">← 左折</button><button data-turn="straight">↑ 直進</button><button data-turn="right">右折 →</button><button data-act="pause">一時停止</button><button data-act="reset">最初から</button><button data-act="traffic">周りの車</button></div>
    <div class="traffic" data-k="traffic"></div>
    <div class="msg" data-k="msg"></div>
    <div class="keys">ピンク: モデルが描いた走行経路　水色: ナビの経路<br>キー ← ↑ →: 次の交差点で曲がる方向 · Space: 一時停止 · R: 最初から · T: 周りの車</div>`;
  document.body.appendChild(hud);
  const el = {};
  for (const e of hud.querySelectorAll('[data-k]')) el[e.dataset.k] = e;
  const side = document.createElement('div');
  side.id = 'pilot-side';
  side.innerHTML = '<figure><canvas width="320" height="160"></canvas><figcaption>モデルへの入力 320×160</figcaption></figure>' +
    '<figure><canvas width="80" height="40"></canvas><figcaption>モデルの領域分割</figcaption></figure>';
  document.body.appendChild(side);
  const [inCanvas, segCanvas] = side.querySelectorAll('canvas');
  const in2d = inCanvas.getContext('2d'), inImg = in2d.createImageData(CAM.W, CAM.H);
  const s2d = segCanvas.getContext('2d'), segImg = s2d.createImageData(80, 40);

  // ---------------------------------------------------------------- state
  const car = { x: 0, y: 0, yaw: 0, v: 0, steer: 0 };
  const speedCtl = createSpeedController(), safety = createSafety();
  let plan = null, action = [0, 0, 1], tick = 0, acc = 0, stillT = 0, gen = 0;
  let busy = false, sinceModel = CONTROL_DT, paused = false, inferMs = 0, msg = '', msgT = 0, deviation = 0, resets = 0;

  function reset(why) {
    const p = nav.start();
    Object.assign(car, { x: p.x, y: p.y, yaw: p.yaw, v: 0, steer: 0 });
    plan = null; action = [0, 0, 1]; tick = 0; acc = 0; stillT = 0; gen++;
    speedCtl.reset(); safety.reset();
    if (trafficOn) traffic.reset(car);
    if (why) { resets++; flash(`${why} — 最初から走り直します`); console.warn(`[pilot] ${why}`); }
  }
  function flash(text) { msg = text; msgT = 4; }
  function ask(turn) {
    if (!nav.request(turn)) flash(`次の交差点は${TURN_JA[turn]}できません`);
  }
  function togglePause() { paused = !paused; }
  function setTraffic(on) {
    trafficOn = on;
    try { localStorage.setItem(TRAFFIC_KEY, on ? '1' : '0'); } catch (e) { /* not remembered */ }
    if (on) traffic.reset(car);
    if (ctx.services.actors) ctx.services.actors.update(on ? traffic.drawList(car.x, car.y) : [], []);
  }

  addEventListener('keydown', (e) => {
    const turn = { ArrowLeft: 'left', ArrowUp: 'straight', ArrowRight: 'right' }[e.code];
    if (turn) { ask(turn); e.preventDefault(); }
    else if (e.code === 'Space') { togglePause(); e.preventDefault(); }
    else if (e.code === 'KeyR') reset();
    else if (e.code === 'KeyT') setTraffic(!trafficOn);
  });
  hud.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.turn) ask(b.dataset.turn);
    else if (b.dataset.act === 'pause') togglePause();
    else if (b.dataset.act === 'reset') reset();
    else if (b.dataset.act === 'traffic') setTraffic(!trafficOn);
    b.blur();
  });

  // ---------------------------------------------------------------- camera
  function placeCamera() {
    const c = Math.cos(car.yaw), s = Math.sin(car.yaw);
    const [X, Z] = T.P([car.x + c * CAM.ahead, car.y + s * CAM.ahead]);
    const fx = c * Math.cos(CAM.pitch), fy = s * Math.cos(CAM.pitch), fz = -Math.sin(CAM.pitch);
    camera.fov = VFOV; camera.aspect = CAM.W / CAM.H; camera.near = 0.1; camera.far = 2500;
    camera.updateProjectionMatrix();
    camera.position.set(X, CAM.height, Z);
    camera.up.set(0, 1, 0);
    camera.lookAt(X + fx, CAM.height + fz, Z - fy);
    camera.updateMatrixWorld();
    ctx.player.position.set(X, 0, Z);
  }

  // ---------------------------------------------------------------- closed loop
  /** The plan's path (ego frame of the pose its frame was taken at) in the car's current ego frame. */
  function egoPath(p) {
    const c0 = Math.cos(p.pose.yaw), s0 = Math.sin(p.pose.yaw), c = Math.cos(car.yaw), s = Math.sin(car.yaw);
    const out = [];
    for (let i = 0; i < p.path.length; i += 2) {
      const wx = p.pose.x + c0 * p.path[i] - s0 * p.path[i + 1] - car.x, wy = p.pose.y + s0 * p.path[i] + c0 * p.path[i + 1] - car.y;
      out.push([c * wx + s * wy, -s * wx + c * wy]);
    }
    return out;
  }

  function controlStep() {
    if (!plan) { action = [0, 0, 1]; return; }
    plan.target = safety.apply(plan.raw, car.v, CONTROL_DT, plan.tl);
    action = [purePursuit(egoPath(plan), car.v), ...speedCtl.control(plan.target, car.v, CONTROL_DT)];
  }

  function physicsStep() {
    if (tick++ % CONTROL_EVERY === 0) controlStep();
    stepVehicle(car, action, DT);
    if (trafficOn) traffic.step(DT, worldT, car);
    const dev = nav.track(car.x, car.y);
    deviation = dev;
    stillT = car.v < 0.1 ? stillT + DT : 0;
    if (dev > ROUTE_DEVIATION) reset('ルートから外れました');
    else if (stillT > BLOCKED_TIMEOUT) reset('90 秒動けませんでした');
  }

  /** Before the frame is rendered: advance the car (and the traffic) in fixed physics steps, pose the other cars
   *  and put the camera on the ego. t: the town's time (signals). */
  let worldT = 0;
  function update(dt, t = worldT + dt) {
    worldT = t;
    if (!paused) {
      acc += dt;
      while (acc >= DT) { acc -= DT; physicsStep(); }
    }
    if (trafficOn && ctx.services.actors) ctx.services.actors.update(traffic.drawList(car.x, car.y), []);
    msgT -= dt;
    placeCamera();
  }

  // ---------------------------------------------------------------- the model's view and inference
  const gl = renderer.getContext();
  const modelPipe = modelPipeline();
  modelPipe.setSize(CAM.W, CAM.H, 1);
  const rgba = new Uint8Array(CAM.W * CAM.H * 4);
  const img = new Float32Array(3 * CAM.W * CAM.H);
  const vp = new THREE.Vector4();

  /** Before the main view is drawn (same scene state and camera): render and read the 320x160 model frame. */
  function capture(t, dt) {
    sinceModel += dt;
    if (paused || busy || sinceModel < CONTROL_DT) return;
    sinceModel = 0;
    renderer.getViewport(vp);
    const pr = renderer.getPixelRatio();
    renderer.setViewport(0, 0, CAM.W / pr, CAM.H / pr);
    modelPipe.render(scene, camera, sunDir, t);
    gl.readPixels(0, 0, CAM.W, CAM.H, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    renderer.setViewport(vp);
    const n = CAM.W * CAM.H;
    for (let y = 0; y < CAM.H; y++) {                       // GL rows are bottom-up
      for (let x = 0, i = (CAM.H - 1 - y) * CAM.W * 4, j = y * CAM.W; x < CAM.W; x++, i += 4, j++) {
        img[j] = rgba[i]; img[n + j] = rgba[i + 1]; img[2 * n + j] = rgba[i + 2];
        inImg.data[4 * j] = rgba[i]; inImg.data[4 * j + 1] = rgba[i + 1]; inImg.data[4 * j + 2] = rgba[i + 2]; inImg.data[4 * j + 3] = 255;
      }
    }
    in2d.putImageData(inImg, 0, 0);
    runModel({ x: car.x, y: car.y, yaw: car.yaw }, car.v, nav.inputs());
  }

  let inflight = null;
  function runModel(pose, v, inp) {
    inflight = infer(pose, v, inp);
    return inflight;
  }

  async function infer(pose, v, inp) {
    busy = true;
    const g = gen;
    const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw), dx = inp.tp[0] - pose.x, dy = inp.tp[1] - pose.y;
    const tp = [c * dx + s * dy, -s * dx + c * dy];
    const t0 = performance.now();
    try {
      const feeds = {                                       // fresh buffers: the WASM worker takes them over
        img: new ort.Tensor('float32', img.slice(), [1, 3, CAM.H, CAM.W]),
        cond: new ort.Tensor('float32', condFeatures(inp.cmd, tp), [1, 7]),
        speed: new ort.Tensor('float32', speedFeatures(v), [1, 3]),
      };
      const out = await session.run(feeds);
      inferMs = performance.now() - t0;
      const sp = out.speed_probs.data;
      let raw = 0;
      for (let i = 0; i < sp.length; i++) raw += sp[i] * i;     // expectation over the 0..11 m/s bins
      if (sp[0] > 0.5) raw = 0;
      if (g === gen) {
        plan = { path: Array.from(out.path.data), raw, target: plan ? plan.target : raw, tl: Array.from(out.tl_probs.data), pose, inp };
        drawSeg(out.seg.data);
      }
    } catch (e) { console.error('[pilot] inference failed', e); }
    busy = false;
  }

  function drawSeg(seg) {
    for (let i = 0; i < 80 * 40; i++) {
      const c = SEM_RGB[Number(seg[i])] || [0, 0, 0];
      segImg.data[4 * i] = c[0]; segImg.data[4 * i + 1] = c[1]; segImg.data[4 * i + 2] = c[2]; segImg.data[4 * i + 3] = 255;
    }
    s2d.putImageData(segImg, 0, 0);
  }

  // ---------------------------------------------------------------- overlay
  const NEAR = 0.5;
  const camPt = (x, y, h) => { const [X, Z] = T.P([x, y]); return new THREE.Vector3(X, h, Z).applyMatrix4(camera.matrixWorldInverse); };
  const toPx = (p) => {
    const q = p.clone().applyMatrix4(camera.projectionMatrix);
    return [(q.x + 1) / 2 * overlay.width, (1 - q.y) / 2 * overlay.height];
  };
  /** Polyline on the ground (KeiSim points), clipped just in front of the camera. */
  function drawGround(points, h, style, width) {
    const cam = points.map(([x, y]) => camPt(x, y, h));
    o2d.beginPath();
    let open = false;
    for (let i = 1; i < cam.length; i++) {
      let p = cam[i - 1], q = cam[i];
      const pin = p.z <= -NEAR, qin = q.z <= -NEAR;
      if (!pin && !qin) { open = false; continue; }
      if (!pin) { p = p.clone().lerp(q, (p.z + NEAR) / (p.z - q.z)); open = false; }
      if (!qin) q = p.clone().lerp(q, (p.z + NEAR) / (p.z - q.z));
      if (!open) { o2d.moveTo(...toPx(p)); open = true; }
      o2d.lineTo(...toPx(q));
      if (!qin) open = false;
    }
    o2d.strokeStyle = style; o2d.lineWidth = width; o2d.lineJoin = 'round'; o2d.lineCap = 'round';
    o2d.stroke();
  }

  function drawOverlay() {
    const r = renderer.domElement.getBoundingClientRect();
    if (overlay.width !== Math.round(r.width) || overlay.height !== Math.round(r.height)) {
      overlay.width = Math.round(r.width); overlay.height = Math.round(r.height);
    }
    overlay.style.left = `${r.left}px`; overlay.style.top = `${r.top}px`;
    overlay.style.width = `${r.width}px`; overlay.style.height = `${r.height}px`;
    o2d.clearRect(0, 0, overlay.width, overlay.height);
    const w = Math.max(2, overlay.width / 320);
    drawGround(nav.slice(nav.s, nav.s + 80), 0.02, 'rgba(110, 200, 255, 0.55)', w);
    if (plan) {
      const { pose } = plan, c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
      const pts = [[pose.x, pose.y]];
      for (let i = 0; i < plan.path.length; i += 2) pts.push([pose.x + c * plan.path[i] - s * plan.path[i + 1], pose.y + s * plan.path[i] + c * plan.path[i + 1]]);
      drawGround(pts, 0.04, '#ff4fd8', 1.5 * w);
    }
  }

  function drawHud() {
    const tl = plan ? plan.tl : null;
    const k = tl ? tl.indexOf(Math.max(...tl)) : -1;
    const inp = nav.inputs(), j = nav.nextDecision();
    el.ep.textContent = `${backend} · ${inferMs.toFixed(0)} ms`;
    el.v.textContent = (car.v * 3.6).toFixed(0);
    el.t.textContent = plan ? (plan.target * 3.6).toFixed(0) : '–';
    el.tl.textContent = k < 0 ? '–' : `● ${TL_NAMES[k]}`;
    el.tl.style.color = k < 0 ? '' : TL_CSS[k];
    el.tlp.textContent = k < 0 ? '' : `${(tl[k] * 100).toFixed(0)}%`;
    el.cmd.textContent = j ? TURN_JA[j.turn] : '直進';
    el.dist.textContent = !j ? '' : inp.dist > 0 ? `あと ${inp.dist.toFixed(0)} m` : '(交差点内)';
    el.msg.textContent = paused ? '一時停止中' : msgT > 0 ? msg : '';
    for (const b of hud.querySelectorAll('[data-turn]')) b.classList.toggle('on', !!j && b.dataset.turn === j.turn);
    hud.querySelector('[data-act="pause"]').classList.toggle('on', paused);
    hud.querySelector('[data-act="traffic"]').classList.toggle('on', trafficOn);
    el.traffic.textContent = trafficOn ? `周りの車 ${traffic.n} 台${traffic.contacts ? ` · 接触 ${traffic.contacts} 回` : ''}` : '周りの車なし';
  }

  /** After the main view is drawn. */
  function after() {
    drawOverlay();
    drawHud();
  }

  reset();
  placeCamera();
  status('');
  return {
    update, capture, after, backend, reset,
    adapter: backend === 'webgpu' && adapter && adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture}`.trim() : '',
    idle: () => inflight || Promise.resolve(),             // the inference in flight (tests: window.__pilotRun)
    state: () => ({ car: { ...car }, plan, s: nav.s, inputs: nav.inputs(), inferMs, deviation, resets, t: ctx.time,
      traffic: trafficOn ? { n: traffic.n, contacts: traffic.contacts } : null }),
  };
}
