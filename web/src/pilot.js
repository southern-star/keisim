// KeiPilot in the browser (index.html?pilot=1): KeiSim's closed loop, with the town drawn by KeiView. Ten times a
// second the car camera is rendered at the model's 320x160 exactly as in ego mode (src/ego.js), the driving model
// (scripts/export_onnx.py -> models/keipilot.onnx, run by onnxruntime-web on WebGPU or WASM in a worker,
// src/infer_worker.js; ?infer=main: on the page's main thread as before) plans a path and a
// target speed from that frame, and the car follows the plan with KeiSim's controllers and vehicle model
// (keisim/control.py, keisim/vehicle.py) behind the two safety layers of keipilot/agent.py. The route and the
// model's route inputs (next turn, target point) come from src/nav.js. Signals switch with KeiSim's timing (split
// phases, or two-phase signals with right turns that wait for oncoming gaps in the two-phase towns); other cars
// (src/traffic.js) drive around the town unless switched off (?traffic=0, the HUD button or T, remembered).
// Towns with 片側2車線 roads and ramps (KeiSim's lane graph, src/lanes.js): the route changes lanes for its next turn,
// the left turn at a junction with a ramp goes by the ramp, and the drive starts before a ramp (?ramp=N: which one,
// ?ramp=-1: the usual start); the other cars change lanes and take the ramps too.
// Multi-camera models (index.html?pilot=mc, models/keipilot_mc.onnx, whose image input scripts/export_onnx.py names
// 'imgs'): the model sees KeiSim's wide_narrow rig as KeiView draws it for training (src/rig.js), a cylindrical 180 deg
// wide camera stitched from three pinhole faces and a 50 deg narrow one, both 320x160, while the main view stays the
// 100 deg car camera of ?pilot=1. Models with the yield head also get keipilot/agent.py's YieldHold, set as in the
// multi-camera evaluations.
// Keys: ← ↑ → turn at the next junction, Space pause, R back to the start, T other cars (or the buttons).
import * as THREE from 'three';
import { buildLanes, createNavigator } from './nav.js';
import { makeRemap, rigShots, shotPose } from './rig.js';
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
// memory-input models (the ONNX has a 'stand' input): metres driven since the car last stood, counted every control
// step as keipilot/data.py stand_update does (0 while standing, saturating at STAND_CAP)
const STAND_V = 0.05, STAND_CAP = 10;
function standFeatures(m) {
  m = clamp(m, 0, 10);
  return Float32Array.from([m / 10, Math.exp(-m / 1.5), Math.exp(-m / 5)]);
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

/** keipilot/agent.py YieldHold as the multi-camera models are evaluated (--yield_hold 4 1 3.5 4 0.5 0.5 10): while the
 *  car is slow and the model's yield head reads a car with priority arriving within `on` s, the target is held at 0
 *  until that arrival plus `margin` s. A hold starts only while the model itself slows or stands, where its light head
 *  sees no light, within `afterM` m of the last stand (moving off after a 止まれ or a give-way stop). */
function createYieldHold({ on = 4, margin = 1, v = 3.5, stand = 4, brake = 0.5, noneP = 0.5, afterM = 10 } = {}) {
  let t = 0, until = -1, stood = 0;
  return {
    reset() { t = 0; until = -1; stood = 0; },
    holding: () => t < until,
    apply(raw, speed, tta, dt, tl, standM) {
      t += dt;
      stood = speed < 0.1 ? stood + dt : 0;
      if (speed < v && tta < on) {
        const end = t + Math.max(tta, 0) + margin;
        if (t < until) until = Math.min(until, end);
        else if (stood <= stand && raw < Math.max(speed - brake, brake) && tl[3] >= noneP && standM < afterM) until = end;
      }
      return t < until ? 0 : raw;
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

/** The model in a worker (src/infer_worker.js), off the main thread: {backend, adapter, inputNames, outputNames,
 *  run(feeds)} with feeds {name: {data: Float32Array, dims}} (the buffers are handed over) and outputs {name: {data}}. */
function workerSession(ort, gpu, force, bytes) {
  return new Promise((resolve, reject) => {
    let w;
    try { w = new Worker(new URL('./infer_worker.js', import.meta.url), { type: 'module' }); } catch (e) { reject(e); return; }
    const pending = new Map();
    let next = 0;
    const fail = (e) => { clearTimeout(timer); w.terminate(); reject(e); };
    const timer = setTimeout(() => fail(new Error('the worker did not start the model')), 120000);
    w.onerror = (e) => fail(new Error(e.message || 'worker error'));
    w.onmessage = ({ data: m }) => {
      if (m.type === 'ready') {
        clearTimeout(timer);
        resolve({ backend: m.backend, adapter: m.adapter, inputNames: m.inputNames, outputNames: m.outputNames, where: 'worker',
          run: (feeds) => new Promise((res, rej) => {
            const id = next++;
            pending.set(id, { res, rej });
            w.postMessage({ type: 'run', id, feeds }, Object.values(feeds).map((f) => f.data.buffer));
          }) });
      } else if (m.id === undefined) fail(new Error(m.message));
      else {
        const p = pending.get(m.id);
        pending.delete(m.id);
        if (p) { if (m.type === 'out') p.res(m.out); else p.rej(new Error(m.message)); }
      }
    };
    w.postMessage({ type: 'init', ort, gpu, force, bytes });
  });
}

/** The model on the main thread (?infer=main, or browsers without the worker): the same interface. */
async function mainSession(ortDir, gpu, adapter, bytes) {
  const ort = await import(`${ortDir}${gpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs'}`);
  ort.env.wasm.wasmPaths = ortDir;
  if (adapter) ort.env.webgpu.adapter = adapter;
  if (!gpu) ort.env.wasm.proxy = true;                      // CPU inference in a worker, off the render loop
  for (const ep of gpu ? ['webgpu', 'wasm'] : ['wasm']) {
    try {
      const s = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
      return { backend: ep, adapter: ep === 'webgpu' && adapter && adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture}`.trim() : '',
        inputNames: s.inputNames, outputNames: s.outputNames, where: 'main',
        run: (feeds) => {
          const t = {};
          for (const [k, f] of Object.entries(feeds)) t[k] = new ort.Tensor('float32', f.data, f.dims);
          return s.run(t);
        } };
    } catch (e) { console.warn(`[pilot] ${ep} unavailable`, e); }
  }
  throw new Error('推論エンジンを起動できませんでした');
}

export async function installPilot({ ctx, scene, renderer, camera, sunDir, modelPipeline, mainSize, params, status, progress = () => {} }) {
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
  // ?ort=node_modules/onnxruntime-web/dist/: onnxruntime-web from this site instead of the CDN (offline tests,
  // tools/pilot_check.mjs --ort); a relative path only, never another origin
  const ortDir = params.get('ort');
  const ORT = ortDir && /^[\w-]+(\/[\w.-]+)*\/$/.test(ortDir) && !ortDir.includes('..') ? new URL(ortDir, location.href).href : ORT_URL;
  const adapter = want === 'wasm' ? null : await gpuAdapter();
  const gpu = want === 'webgpu' || !!adapter;
  const MB = (n) => (n / 1e6).toFixed(0);
  const mc = params.get('pilot') === 'mc';                  // the multi-camera demo
  const bytes = await fetchModel(params.get('model') || (mc ? 'models/keipilot_mc.onnx' : 'models/keipilot.onnx'),
    (got, total) => { status(`運転モデルを読み込み中… ${MB(got)}${total >= got ? ` / ${MB(total)}` : ''} MB`); if (total) progress(0.9 * got / total); });
  status('運転モデルを準備中…');
  // the model runs in a worker unless ?infer=main or the worker cannot start it
  const session = (params.get('infer') !== 'main' && await workerSession(ORT, gpu, want === 'webgpu', bytes)
    .catch((e) => { console.warn('[pilot] no inference worker, the model runs on the main thread', e); return null; }))
    || await mainSession(ORT, gpu, adapter, bytes);
  const backend = session.backend;
  const useStand = session.inputNames.includes('stand');
  // camera-rig models: what the rig draws for each model frame (src/rig.js), one image per camera
  const rig = session.inputNames.includes('imgs') ? rigShots(params.get('rig') || 'wide_narrow') : null;
  const K = rig ? rig.length : 1;
  const yieldHold = session.outputNames.includes('yield_tta') ? createYieldHold() : null;

  // ---------------------------------------------------------------- page elements
  const overlay = document.createElement('canvas');
  overlay.id = 'pilot-overlay';
  document.body.appendChild(overlay);
  const o2d = overlay.getContext('2d');
  const hud = document.createElement('div');
  hud.id = 'pilot-hud';
  hud.innerHTML = `<div><b>KeiPilot</b> <span class="cams">${rig ? 'Mc · 広角 180° + 望遠 50°' : '単眼 100°'}</span> <span class="ep" data-k="ep"></span></div>
    <div class="switch" data-k="switch"></div>
    <div>速度 <b data-k="v">0</b> km/h　目標 <b data-k="t">–</b> km/h <span class="hold" data-k="hold"></span></div>
    <div>信号 <b data-k="tl">–</b> <span data-k="tlp"></span></div>
    <div>次の交差点 <b data-k="cmd">–</b> <span data-k="dist"></span></div>
    <div class="lc" data-k="lc"></div>
    <div class="btns"><button data-turn="left">← 左折</button><button data-turn="straight">↑ 直進</button><button data-turn="right">右折 →</button><button data-act="pause">一時停止</button><button data-act="reset">最初から</button><button data-act="traffic">周りの車</button></div>
    <div class="traffic" data-k="traffic"></div>
    <div class="msg" data-k="msg"></div>
    <div class="keys">ピンク: モデルが描いた走行経路　水色: ナビの経路<br>キー ← ↑ →: 次の交差点で曲がる方向 · Space: 一時停止 · R: 最初から · T: 周りの車</div>`;
  document.body.appendChild(hud);
  const el = {};
  for (const e of hud.querySelectorAll('[data-k]')) el[e.dataset.k] = e;
  // the other demo (single camera <-> multi-camera) with the same town, if this site has its model
  const other = new URL(location.href);
  other.searchParams.set('pilot', rig ? '1' : 'mc');
  other.searchParams.delete('model');
  fetch(rig ? 'models/keipilot.onnx' : 'models/keipilot_mc.onnx', { method: 'HEAD' }).then((r) => {
    if (r.ok) el.switch.innerHTML = `<a href="${other.href}">${rig ? '単眼 100° 版で走る' : 'マルチカメラ（Mc）版で走る'} →</a>`;
  }).catch(() => {});
  const side = document.createElement('div');
  side.id = 'pilot-side';
  const CAMS_JA = { wide: '広角 180°（円筒）', narrow: '望遠 50°' };
  const names = rig ? rig.map((r) => CAMS_JA[r.cam.name] || r.cam.name) : null;
  side.innerHTML = !rig
    ? '<figure><canvas width="320" height="160"></canvas><figcaption>モデルへの入力 320×160</figcaption></figure>' +
      '<figure><canvas width="80" height="40"></canvas><figcaption>モデルの領域分割</figcaption></figure>'
    : names.map((n) => `<figure><canvas width="320" height="160"></canvas><figcaption>${n} 320×160</figcaption></figure>`).join('') +
      names.map((n) => `<figure><canvas width="80" height="40"></canvas><figcaption>${n}の領域分割</figcaption></figure>`).join('');
  if (rig) side.classList.add('rig');
  document.body.appendChild(side);
  const canvases = [...side.querySelectorAll('canvas')];
  const ins = canvases.slice(0, K).map((c) => { const g = c.getContext('2d'); return { g, img: g.createImageData(CAM.W, CAM.H) }; });
  const segs = canvases.slice(K).map((c) => { const g = c.getContext('2d'); return { g, img: g.createImageData(80, 40) }; });

  // ---------------------------------------------------------------- state
  const car = { x: 0, y: 0, yaw: 0, v: 0, steer: 0 };
  const speedCtl = createSpeedController(), safety = createSafety();
  let plan = null, action = [0, 0, 1], tick = 0, acc = 0, stillT = 0, gen = 0, standM = STAND_CAP;
  let busy = false, sinceModel = CONTROL_DT, paused = false, inferMs = 0, nInfer = 0, captureMs = 0, captured = null, msg = '', msgT = 0, deviation = 0, resets = 0;

  // lane graph towns with ramps: start before one (keisim route.ramp_route), unless ?ramp=-1
  const rampStart = nav.ramps && Number(params.get('ramp') ?? 0) >= 0 ? Number(params.get('ramp') ?? 0) : null;
  function reset(why) {
    const p = nav.start(rampStart);
    Object.assign(car, { x: p.x, y: p.y, yaw: p.yaw, v: 0, steer: 0 });
    plan = null; action = [0, 0, 1]; tick = 0; acc = 0; stillT = 0; gen++; standM = STAND_CAP;
    speedCtl.reset(); safety.reset();
    if (yieldHold) yieldHold.reset();
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
    standM = car.v < STAND_V ? 0 : Math.min(standM + car.v * CONTROL_DT, STAND_CAP);
    if (!plan) { action = [0, 0, 1]; return; }
    plan.target = safety.apply(plan.raw, car.v, CONTROL_DT, plan.tl);
    if (yieldHold && plan.tta !== undefined) plan.target = Math.min(plan.target, yieldHold.apply(plan.raw, car.v, plan.tta, CONTROL_DT, plan.tl, standM));
    action = [purePursuit(egoPath(plan), car.v), ...speedCtl.control(plan.target, car.v, CONTROL_DT)];
  }

  function physicsStep() {
    if (tick++ % CONTROL_EVERY === 0) controlStep();
    stepVehicle(car, action, DT);
    if (trafficOn) traffic.step(DT, worldT, car, { s: nav.s, decision: nav.nextDecision() });
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
  const img = new Float32Array(K * 3 * CAM.W * CAM.H);
  const syncRead = params.get('readback') === 'sync';      // ?readback=sync: wait for the GPU as before (tests, A/B)
  // ?hz=N: at most N model calls a second in real time (default 10, KeiSim's control rate): in towns where the GPU is the
  // limit (Sapporo), fewer model frames leave more of it to the view
  const hz = Number(params.get('hz'));
  const MODEL_DT = hz > 0 ? Math.max(CONTROL_DT, 1 / hz) : CONTROL_DT;
  const calls = [];                                         // when the last model calls finished (the HUD's rate)
  // The town waits for the model. KeiSim decides every 0.1 s without delay; in real time the decision for a frame
  // lands when the car has driven on: with WASM (0.45 s for the multi-camera model) 4.5 m on at 36 km/h, so the
  // model's braking for a red light came a stop line's distance late and it went through. The town (car, traffic,
  // signals) therefore runs at most PACE_LAG s past the model frame being decided on (?pace=S; ?pace=0: no waiting):
  // slow motion where the model takes longer.
  const PACE_LAG = params.has('pace') ? Math.max(0, Number(params.get('pace')) || 0) : 0.25;
  const pacing = PACE_LAG > 0;
  let flightT = null, slow = 1;                             // town time of the frame in flight; town s per real s
  const paceAcc = { town: 0, real: 0 };

  /** Real time (main.js, before the town advances by dt): how much of it the town may advance now. */
  function pace(dt) {
    const d = pacing && flightT !== null ? Math.min(dt, Math.max(0, flightT + PACE_LAG - worldT)) : dt;
    paceAcc.town += d; paceAcc.real += dt;
    if (paceAcc.real >= 1) { slow = paceAcc.town / paceAcc.real; paceAcc.town = 0; paceAcc.real = 0; }
    return d;
  }

  /** n read-backs of w x h RGBA from the canvas' lower left corner: at once (sync), or into pixel buffers that
   *  fetch() copies out once the GPU has drawn them (WebGL2 PIXEL_PACK_BUFFER; the caller polls a fence), so the
   *  main thread does not stop to wait for the GPU in the middle of a frame. */
  function pixelReader(w, h, n) {
    const bufs = Array.from({ length: n }, () => new Uint8Array(w * h * 4));
    const pbos = bufs.map(() => gl.createBuffer());
    return {
      bufs,
      read(j, sync) {
        if (sync) { gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, bufs[j]); return; }
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbos[j]);
        // fresh storage every time: written again into the old storage, Chrome drops the copy it makes for the
        // read-back at the fence (with this page's antialias: false / alpha: false context) and warns every frame
        gl.bufferData(gl.PIXEL_PACK_BUFFER, w * h * 4, gl.STREAM_READ);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, 0);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      },
      fetch() {
        pbos.forEach((b, j) => { gl.bindBuffer(gl.PIXEL_PACK_BUFFER, b); gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, bufs[j]); });
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      },
    };
  }
  const single = rig ? null : pixelReader(CAM.W, CAM.H, 1);
  const vp = new THREE.Vector4(), buf = new THREE.Vector2();

  /** 320x160 RGB (top row first, 3 bytes per pixel) into camera k's planes of the model input and its side image. */
  function putImage(k, rgb) {
    const n = CAM.W * CAM.H, o = k * 3 * n, d = ins[k].img.data;
    for (let j = 0; j < n; j++) {
      img[o + j] = rgb[3 * j]; img[o + n + j] = rgb[3 * j + 1]; img[o + 2 * n + j] = rgb[3 * j + 2];
      d[4 * j] = rgb[3 * j]; d[4 * j + 1] = rgb[3 * j + 1]; d[4 * j + 2] = rgb[3 * j + 2]; d[4 * j + 3] = 255;
    }
    ins[k].g.putImageData(ins[k].img, 0, 0);
  }

  // camera rigs: each camera's shots with one pipeline per image size, the faces of the cylinder without the vignette
  // and the light leak (screen-space, they would show the seams), as src/ego.js draws them for KeiView
  const views = rig ? rig.map(({ cam, shots, cyl }) => {
    const w = shots[0].width, h = shots[0].height, pipe = modelPipeline();
    pipe.setSize(w, h, 1);
    pipe.compMat.uniforms.uExposure.value = modelPipe.compMat.uniforms.uExposure.value;
    if (cyl) pipe.compMat.uniforms.uVignette.value = 0;
    return { cam, shots, w, h, face: !!cyl, pipe, reader: pixelReader(w, h, shots.length), atlas: new Uint8Array(shots.length * w * h * 3),
      rgb: new Uint8Array(cam.width * cam.height * 3), remap: cyl ? makeRemap(shots.length * w, h, cyl.mapX, cyl.mapY) : null };
  }) : null;
  const rigW = rig ? Math.max(...views.map((v) => v.w)) : CAM.W, rigH = rig ? Math.max(...views.map((v) => v.h)) : CAM.H;

  /** Render one pinhole shot of a rig (src/rig.js) at the car into the canvas' lower left corner and read it back
   *  (shot j of camera v). */
  function shoot(v, s, j, first, t, pr, sync) {
    const p = shotPose(s, car.x, car.y, car.yaw);
    const [X, Z] = T.P([p.pos[0], p.pos[1]]);
    camera.fov = p.vfov; camera.aspect = v.w / v.h; camera.near = 0.1; camera.far = 2500;
    camera.updateProjectionMatrix();
    camera.position.set(X, p.pos[2], Z);
    camera.up.set(0, 1, 0);
    camera.lookAt(X + p.fwd[0], p.pos[2] + p.fwd[2], Z - p.fwd[1]);
    camera.updateMatrixWorld();
    renderer.setViewport(0, 0, v.w / pr, v.h / pr);
    if (ctx.wires) ctx.wires.setResolution(v.w, v.h);
    v.pipe.render(scene, camera, sunDir, t, { shadows: first, leak: !v.face });
    v.reader.read(j, sync);
  }

  /** Every camera of the rig in the same scene state, each shot read back (stitchRig puts them together). */
  function drawRig(t, pr, sync) {
    let first = true;
    views.forEach((v) => v.shots.forEach((s, j) => { shoot(v, s, j, first, t, pr, sync); first = false; }));
    if (ctx.wires && mainSize) { const m = mainSize(); ctx.wires.setResolution(m.x, m.y); }
    placeCamera();                                          // back to the main view's camera
  }

  /** The rig's images from its shots: the faces side by side (top row first) stitched into the cylinder, the
   *  pinhole cameras as they are. */
  function stitchRig() {
    views.forEach((v, k) => {
      const out = v.face ? v.atlas : v.rgb, aw = v.shots.length * v.w;
      v.reader.bufs.forEach((rgba, j) => {
        for (let y = 0; y < v.h; y++) {                     // GL rows are bottom-up
          for (let x = 0, i = (v.h - 1 - y) * v.w * 4, o = 3 * (y * aw + j * v.w); x < v.w; x++, i += 4, o += 3) {
            out[o] = rgba[i]; out[o + 1] = rgba[i + 1]; out[o + 2] = rgba[i + 2];
          }
        }
      });
      if (v.face) v.remap(v.atlas, v.rgb);
      putImage(k, v.rgb);
    });
  }

  /** Before the main view is drawn (same scene state and camera): render the 320x160 model frame, or every camera of
   *  the rig, and read it back when the GPU has drawn it, a frame or two later (the model then sees the frame of the
   *  car state it was drawn at, as before); sync (window.__pilotRun, ?readback=sync): read it at once. */
  const rgb1 = new Uint8Array(CAM.W * CAM.H * 3);
  let pending = null, drawMs = 0, readMs = 0;
  function capture(t, dt, sync = false) {
    sinceModel += dt;
    if (pending) {
      if (!sync) { collect(); return; }
      gl.deleteSync(pending.fence);                         // a test takes over (window.__pilotRun): draw its frame now
      pending = null;
    }
    if (paused || busy || sinceModel < (sync ? CONTROL_DT : MODEL_DT) - 1e-6) return;
    sinceModel = 0;
    const t0 = performance.now();
    renderer.getViewport(vp);
    const pr = renderer.getPixelRatio();
    renderer.getDrawingBufferSize(buf);
    if (buf.x < rigW || buf.y < rigH) { flash('画面が小さすぎてカメラ画像を描けません（ウィンドウを広げてください）'); return; }
    const actors = ctx.services.actors;
    if (actors && actors.view) actors.view('model');             // the model sees the plain cars it was trained on
    sync = sync || syncRead;
    if (rig) drawRig(t, pr, sync);
    else {
      renderer.setViewport(0, 0, CAM.W / pr, CAM.H / pr);
      modelPipe.render(scene, camera, sunDir, t);
      single.read(0, sync);
    }
    if (actors && actors.view) actors.view('main');
    renderer.setViewport(vp);
    const at = { pose: { x: car.x, y: car.y, yaw: car.yaw }, v: car.v, inp: nav.inputs(), stand: standM, t, g: gen, t1: performance.now() };
    drawMs = at.t1 - t0;
    flightT = t;
    if (sync) { readMs = 0; finish(at); return; }
    at.fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    pending = at;
  }

  /** Every frame while a model frame is being read back: once the GPU is done, copy it out and run the model. */
  function collect() {
    if (gl.clientWaitSync(pending.fence, 0, 0) === gl.TIMEOUT_EXPIRED) return;
    const at = pending;
    pending = null;
    gl.deleteSync(at.fence);
    (rig ? views.map((v) => v.reader) : [single]).forEach((r) => r.fetch());
    readMs = performance.now() - at.t1;
    if (at.g === gen) finish(at);                           // not after a reset
    else flightT = null;
  }

  function finish(at) {
    const t0 = performance.now();
    if (rig) stitchRig();
    else {
      const rgba = single.bufs[0];
      for (let y = 0; y < CAM.H; y++) {                     // GL rows are bottom-up
        for (let x = 0, i = (CAM.H - 1 - y) * CAM.W * 4, j = 3 * y * CAM.W; x < CAM.W; x++, i += 4, j += 3) {
          rgb1[j] = rgba[i]; rgb1[j + 1] = rgba[i + 1]; rgb1[j + 2] = rgba[i + 2];
        }
      }
      putImage(0, rgb1);
    }
    captureMs = drawMs + performance.now() - t0;            // the main thread's share of the model frame
    captured = { ...at.pose, t: at.t };
    runModel(at.pose, at.v, at.inp, at.stand);
  }

  let inflight = null;
  function runModel(pose, v, inp, stand) {
    inflight = infer(pose, v, inp, stand);
    return inflight;
  }

  async function infer(pose, v, inp, stand) {
    busy = true;
    const g = gen;
    const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw), dx = inp.tp[0] - pose.x, dy = inp.tp[1] - pose.y;
    const tp = [c * dx + s * dy, -s * dx + c * dy];
    const t0 = performance.now();
    try {
      const feeds = {                                       // fresh buffers: the worker takes them over
        [rig ? 'imgs' : 'img']: { data: img.slice(), dims: rig ? [1, K, 3, CAM.H, CAM.W] : [1, 3, CAM.H, CAM.W] },
        cond: { data: condFeatures(inp.cmd, tp), dims: [1, 7] },
        speed: { data: speedFeatures(v), dims: [1, 3] },
      };
      if (useStand) feeds.stand = { data: standFeatures(stand), dims: [1, 3] };
      const out = await session.run(feeds);
      inferMs = performance.now() - t0;
      nInfer++;
      calls.push(performance.now());
      while (calls.length && calls[0] < calls[calls.length - 1] - 2000) calls.shift();
      const sp = out.speed_probs.data;
      let raw = 0;
      for (let i = 0; i < sp.length; i++) raw += sp[i] * i;     // expectation over the 0..11 m/s bins
      if (sp[0] > 0.5) raw = 0;
      if (g === gen) {
        plan = { path: Array.from(out.path.data), raw, target: plan ? plan.target : raw, tl: Array.from(out.tl_probs.data), pose, inp,
          tta: out.yield_tta ? out.yield_tta.data[0] : undefined };
        drawSeg(out.seg.data);
      }
    } catch (e) { console.error('[pilot] inference failed', e); }
    busy = false;
    flightT = null;
  }

  /** The model's segmentation of each camera (seg: K x 40 x 80 class ids). */
  function drawSeg(seg) {
    segs.forEach(({ g, img: si }, k) => {
      for (let i = 0, o = k * 80 * 40; i < 80 * 40; i++) {
        const c = SEM_RGB[Number(seg[o + i])] || [0, 0, 0];
        si.data[4 * i] = c[0]; si.data[4 * i + 1] = c[1]; si.data[4 * i + 2] = c[2]; si.data[4 * i + 3] = 255;
      }
      g.putImageData(si, 0, 0);
    });
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
    const now = performance.now(), rate = calls.filter((c) => c > now - 2000).length / 2;
    const ep = [backend, `判断 ${rate.toFixed(1)} 回/秒`, `推論 ${inferMs.toFixed(0)} ms`, `描画 ${captureMs.toFixed(0)} ms`];
    if (slow < 0.95) ep.push(`スロー ×${slow.toFixed(2)}`);
    el.ep.innerHTML = ep.map((s) => `<span>${s}</span>`).join(' · ');   // a narrow HUD wraps between the items only
    el.ep.title = slow < 0.95 ? `モデルの判断に ${PACE_LAG} 秒より長くかかるので、町の時間をゆっくり進めています（?pace=0 で待たない）` : '';
    el.v.textContent = (car.v * 3.6).toFixed(0);
    el.t.textContent = plan ? (plan.target * 3.6).toFixed(0) : '–';
    el.hold.textContent = yieldHold && yieldHold.holding() ? '（譲り待ち）' : '';
    el.tl.textContent = k < 0 ? '–' : `● ${TL_NAMES[k]}`;
    el.tl.style.color = k < 0 ? '' : TL_CSS[k];
    el.tlp.textContent = k < 0 ? '' : `${(tl[k] * 100).toFixed(0)}%`;
    el.cmd.textContent = j ? `${TURN_JA[j.turn]}${j.ramp ? '（ランプ）' : ''}` : '直進';
    // the route's next lane change (lane graph towns): how far, or under way
    const ch = nav.changes ? nav.changes().find((c) => c.s1 > nav.s && !c.ramp) : null;
    el.lc.textContent = !ch || ch.s0 - nav.s > 80 ? '' : ch.s0 > nav.s ? `車線変更 あと ${(ch.s0 - nav.s).toFixed(0)} m` : '車線変更中';
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
    update, pace, capture, after, backend, reset,
    adapter: session.adapter, inferIn: session.where,      // 'worker' or 'main' (?infer=main)
    idle: () => inflight || Promise.resolve(),             // the inference in flight (tests: window.__pilotRun)
    traffic, lanes, nav,                                    // for tests (tools/pilot_check.mjs --yields)
    rig: rig ? rig.map((r) => r.cam.name) : null,
    images: () => ins.map((i) => i.img),                   // the model's last input images (tests: tools/pilot_check.mjs)
    state: () => ({ car: { ...car }, plan, s: nav.s, inputs: nav.inputs(), inferMs, nInfer, captureMs, readMs, captured, deviation, resets, t: ctx.time, standM, useStand,
      cams: K, yieldHold: !!yieldHold, holding: !!yieldHold && yieldHold.holding(), slow, pacing,
      traffic: trafficOn ? { n: traffic.n, contacts: traffic.contacts, npcContacts: traffic.npcContacts, twoPhase: traffic.twoPhase,
        stopSignStops: traffic.stopSignStops, laneChanges: traffic.laneChanges, rampPassages: traffic.rampPassages } : null }),
  };
}
