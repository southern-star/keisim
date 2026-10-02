// KeiView bootstrap — a KeiSim town rendered with Sakuragaoka Station's cel-shading pipeline.
// Adapted from Sakuragaoka Station src/main.js (MIT, see vendor/sakuragaoka/LICENSE): renderer,
// sky, context, module build, batching, main loop, HUD and the headless-screenshot hooks.
import * as THREE from 'three';
import * as L from '../vendor/sakuragaoka/world/layout.js';
import { createContext } from '../vendor/sakuragaoka/core/ctx.js';
import { createRenderPipeline } from '../vendor/sakuragaoka/core/renderer.js';
import { createSky } from '../vendor/sakuragaoka/core/sky.js';
import { Player } from '../vendor/sakuragaoka/core/player.js';
import { batchStatic } from '../vendor/sakuragaoka/core/batch2.js';
import { loadTown } from './town.js';
import { installEgo, tagSemantics } from './ego.js';

export const MODULES = ['ground', 'roads', 'houses', 'trees', 'signals', 'poles'];

const params = new URLSearchParams(location.search);
const SHOT = params.has('shot');
const ONLY = params.get('only') ? params.get('only').split(',').map(s => s.trim()).filter(Boolean) : null;
const SEED = params.get('town') || '1000';
const EGO = params.has('ego');                    // KeiSim ego-camera mode (src/ego.js)
const PILOT = params.has('pilot');                // KeiPilot drives in the browser (src/pilot.js)
const TDIR = (params.get('tdir') || 'towns').replace(/[^\w.-]/g, '');
const $ = (id) => document.getElementById(id);

const isTouch = matchMedia('(pointer: coarse)').matches;
const QUALITY = {
  high: { name: 'high', pixelRatio: Math.min(devicePixelRatio, 1.5), msaa: 4, shadowMap: 4096, shadowSize: 75, petals: 1.0 },
  medium: { name: 'medium', pixelRatio: Math.min(devicePixelRatio, 1.0), msaa: 4, shadowMap: 2048, shadowSize: 60, petals: 0.6 },
  low: { name: 'low', pixelRatio: Math.min(devicePixelRatio, 0.75), msaa: 0, shadowMap: 2048, shadowSize: 45, petals: 0.35 },
};
let qName = params.get('q') || (PILOT ? 'medium' : null) || (() => { try { return localStorage.getItem('keiview.q'); } catch (e) { return null; } })() || (isTouch ? 'medium' : 'high');
if (!QUALITY[qName]) qName = 'high';
const quality = { ...QUALITY[qName] };
if (SHOT) quality.pixelRatio = 1;

// ------------------------------------------------------------------ renderer & scene
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance', stencil: false, preserveDrawingBuffer: SHOT });
renderer.setPixelRatio(1);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
renderer.info.autoReset = false;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.1, 2500);
const sunDir = new THREE.Vector3(...L.SUN_DIR).normalize();
const sky = createSky(scene, sunDir, quality);
const pipeline = createRenderPipeline(renderer, quality);
const audio = { start() {}, update() {}, loop: () => ({ setVolume() {}, setParam() {}, setPosition() {}, stop() {} }), play() {}, muted: true };
const ctx = createContext({ scene, camera, renderer, audio, quality, sunDir });
ctx.sky = sky;
window.__ctx = ctx; window.THREE = THREE;
// ego mode draws signal heads larger so the lamps stay visible at KeiSim's 320x160 camera resolution
ctx.signalScale = Number(params.get('sigscale') || (EGO || PILOT ? 2.2 : 1));

function resize() {
  let w = SHOT ? Number(params.get('w') || 1280) : innerWidth, h = SHOT ? Number(params.get('h') || 720) : innerHeight;
  if (PILOT) {                                    // KeiSim's 2:1 car camera, centred
    w = Math.min(innerWidth, 2 * innerHeight); h = Math.round(w / 2);
    canvas.style.left = `${Math.round((innerWidth - w) / 2)}px`; canvas.style.top = `${Math.round((innerHeight - h) / 2)}px`;
  }
  renderer.setSize(w, h, !SHOT);
  camera.aspect = w / h; camera.updateProjectionMatrix();
  pipeline.setSize(w, h, quality.pixelRatio);
  ctx.wires.setResolution(pipeline.size.x, pipeline.size.y);
}
addEventListener('resize', resize);
resize();

// ------------------------------------------------------------------ fonts (signs, name plates)
async function loadFonts() {
  if (!document.fonts || !document.fonts.load) return;
  const faces = ['700 32px "Noto Sans JP"', '400 32px "Noto Sans JP"', '900 32px "Noto Sans JP"', '700 32px "Noto Serif JP"',
    '700 32px "Zen Maru Gothic"', '400 32px "Yusei Magic"', '400 32px "Yuji Syuku"'];
  const jp = '町丁目止まれ田中佐藤鈴木高橋渡辺山本小林中村伊藤加藤';
  await Promise.race([Promise.all(faces.map(f => document.fonts.load(f, jp).catch(() => null))), new Promise(r => setTimeout(r, 6000))]);
}

// ------------------------------------------------------------------ build
const errors = []; window.__errors = errors;
const stats = { modules: {} }; window.__stats = stats;
function setProgress(frac, label) {
  const bar = $('bar'); if (bar) bar.style.transform = `scaleX(${frac})`;
  const lab = $('loadlabel'); if (lab && label) lab.textContent = label;
}
const LABELS = { ground: '地面', roads: '道路と歩道', houses: '家並み', trees: '桜並木', signals: '信号機', poles: '電柱と電線' };

let town = null;
async function build() {
  setProgress(0.02, `town ${SEED} を読み込み中…`);
  town = await loadTown(`./${TDIR}/town_${SEED}.json`);
  ctx.town = town; window.__town = town;
  L.configureWorld({ play: town.play, heightAt: town.heightAt, farTown: town.farTown, farSkip: town.farSkip });
  const name = $('townname'); if (name) name.textContent = SEED;
  const sum = $('townsum'); if (sum) sum.textContent = `${town.junctions.length} junctions · ${town.buildings.length} lots · ${town.trees.length} trees`;
  await loadFonts();
  const mods = EGO ? MODULES.concat(['actors']) : MODULES;
  const list = ONLY ? mods.filter(m => ONLY.includes(m)).concat(ONLY.filter(m => !mods.includes(m))) : mods;
  let i = 0;
  for (const name of list) {
    setProgress((i + 1) / (list.length + 2), `${LABELS[name] || name} を準備中…`);
    await new Promise(r => setTimeout(r, 0));
    const t0 = performance.now();
    try {
      const mod = await import(`./world/${name}.js`);
      const before = ctx.staticRoot.children.length + ctx.dynamicRoot.children.length;
      const bS = ctx.staticRoot.children.length, bD = ctx.dynamicRoot.children.length;
      if (typeof mod.build !== 'function') throw new Error('module has no build(ctx) export');
      await mod.build(ctx);
      for (const o of ctx.staticRoot.children.slice(bS)) o.userData.module = name;
      for (const o of ctx.dynamicRoot.children.slice(bD)) o.userData.module = name;
      stats.modules[name] = { ms: Math.round(performance.now() - t0), objects: ctx.staticRoot.children.length + ctx.dynamicRoot.children.length - before };
    } catch (e) {
      console.error(`[module ${name}]`, e);
      errors.push({ module: name, message: String(e && e.stack || e) });
    }
    i++;
  }
  setProgress((list.length + 1) / (list.length + 2), '仕上げ中…');
  await new Promise(r => setTimeout(r, 0));
  const wm = ctx.wires.build(); if (wm) { wm.userData.labelSkip = true; scene.add(wm); ctx.wires.setResolution(pipeline.size.x, pipeline.size.y); }
  if (EGO) tagSemantics(ctx);
  stats.batch = batchStatic(ctx.staticRoot, { mat: ctx.mat });
  try { renderer.compile(scene, camera); } catch (e) { console.warn(e); }
  setProgress(1, '');
}

// ------------------------------------------------------------------ player / camera
const player = new Player(camera, canvas, ctx.physics, L.WORLD.play);
ctx.playerObj = player;
function applyView(v) {
  player.fly = !!v.fly;
  player.setPose(v.x, v.z, v.yaw, v.pitch, v.y ?? null);
}
function parseCam(s) {
  const v = s.split(',').map(Number);
  if (v.length === 4) player.setPose(v[0], v[1], v[2], v[3]);
  else if (v.length >= 5) player.setPose(v[0], v[2], v[3], v[4], v[1]);
}
window.__setCam = (x, y, z, yaw, pitch) => { if (y === null || y === undefined) player.setPose(x, z, yaw, pitch); else player.setPose(x, z, yaw, pitch, y); };
window.__view = (i) => { const v = town.views[i]; if (v) applyView(v); return v; };

// ------------------------------------------------------------------ simulation
let simT = params.has('t') ? Number(params.get('t')) : 0;
function stepUpdates(dt, t) {
  ctx.time = t; ctx.shared.uTime.value = t;
  ctx.shared.uGust.value = 0.5 + 0.28 * Math.sin(t * 0.37) + 0.14 * Math.sin(t * 1.13 + 1.7) + 0.08 * Math.sin(t * 2.9 + 0.4);
  ctx.player.position.copy(player.pos);
  for (const fn of ctx._updates) { try { fn(dt, t); } catch (e) { if (!fn.__err) { fn.__err = 1; console.error('update error', e); errors.push({ module: 'update', message: String(e && e.stack || e) }); } } }
}
/** GPU benchmark: renders n frames back-to-back (forcing sync) and returns ms/frame + counts. */
window.__bench = (n = 30) => {
  const gl = renderer.getContext(); const px = new Uint8Array(4);
  pipeline.render(scene, camera, sunDir, simT); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const t0 = performance.now();
  for (let i = 0; i < n; i++) { renderer.info.reset(); sky.update(simT, camera); pipeline.render(scene, camera, sunDir, simT); }
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const ms = (performance.now() - t0) / n;
  return { ms: +ms.toFixed(2), calls: renderer.info.render.calls, triangles: renderer.info.render.triangles, geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures, programs: renderer.info.programs?.length };
};
window.__sim = (target) => { simT = target; stepUpdates(0, simT); };

// ------------------------------------------------------------------ HUD
function hudTick(t) {
  const clock = $('clock');
  if (clock) { const m = 2 + Math.floor(t / 60); clock.textContent = `16:${String(Math.min(59, m)).padStart(2, '0')}`; }
}

// ------------------------------------------------------------------ main loop
let started = false, last = performance.now(), fpsAcc = 0, fpsN = 0, fps = 0;
let pilot = null, manual = false;
function frame(now) {
  requestAnimationFrame(frame);
  let dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (manual) return;
  if (SHOT) dt = 0;
  simT += dt;
  if (pilot) pilot.update(dt);
  else if (!SHOT) player.update(dt);
  ctx.physics.refreshDynamic();
  stepUpdates(dt, simT);
  sky.update(simT, camera);
  renderer.info.reset();
  if (pilot) pilot.capture(simT, dt);             // the model's own 320x160 frame, before the main view
  pipeline.render(scene, camera, sunDir, simT);
  if (pilot) pilot.after(dt);
  if (!SHOT) hudTick(simT);
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; const s = $('stats'); if (s && !s.hidden) s.textContent = `${fps.toFixed(0)} fps · ${renderer.info.render.calls} calls · ${(renderer.info.render.triangles / 1e6).toFixed(2)}M tris`; }
  stats.fps = fps; stats.calls = renderer.info.render.calls; stats.triangles = renderer.info.render.triangles;
}

function start() {
  if (started) { player.requestLock(); return; }
  started = true;
  player.enabled = true;
  document.body.classList.add('playing');
  player.requestLock();
}

async function main() {
  try { await build(); } catch (e) {
    console.error(e); errors.push({ module: 'load', message: String(e && e.stack || e) });
    const lab = $('loadlabel'); if (lab) lab.textContent = String(e.message || e);
    window.__ready = true; return;
  }
  if (EGO) {
    const W = Number(params.get('w') || 320), H = Number(params.get('h') || 160);
    installEgo({ ctx, renderer, scene, camera, pipeline, sky, sunDir, W, H });
    try { window.__egoWarmup(); } catch (e) { console.error(e); errors.push({ module: 'ego', message: String(e && e.stack || e) }); }
    document.body.classList.add('shot');
    window.__ready = true;
    return;
  }
  if (PILOT) {
    const lab = $('loadlabel');
    const status = (msg) => { if (lab) lab.textContent = msg; };
    try {
      const { installPilot } = await import('./pilot.js');
      // the model frame gets its own pipeline at KeiSim's medium quality, like ego mode (keisim keiview_quality)
      const modelPipeline = () => createRenderPipeline(renderer, QUALITY.medium);
      pilot = await installPilot({ ctx, scene, renderer, camera, sunDir, modelPipeline, params, status });
    } catch (e) {
      console.error(e); errors.push({ module: 'pilot', message: String(e && e.stack || e) });
      status(`運転モデルを読み込めませんでした: ${e.message || e}`);
      return;
    }
    document.body.classList.add('loaded', 'playing', 'pilot');
    wireMenus();
    window.__pilot = pilot;
    // tests (tools/pilot_check.mjs): drive `sec` seconds of simulated time, waiting for every model call; the
    // real-time frame loop stays off from the first call on
    window.__pilotRun = async (sec, dt = 0.05) => {
      manual = true;
      for (let k = Math.round(sec / dt); k > 0; k--) {
        simT += dt;
        pilot.update(dt);
        ctx.physics.refreshDynamic();
        stepUpdates(dt, simT);
        sky.update(simT, camera);
        pilot.capture(simT, dt);
        pipeline.render(scene, camera, sunDir, simT);
        pilot.after(dt);
        await pilot.idle();
      }
      return pilot.state();
    };
    window.__sim(simT);
    requestAnimationFrame(frame);
    return;
  }
  if (params.get('cam')) parseCam(params.get('cam'));
  else if (params.get('view')) window.__view(Number(params.get('view')) - 1);
  else applyView(town.views[0]);
  if (params.has('fly')) player.fly = true;
  window.__sim(simT);
  sky.update(simT, camera);
  requestAnimationFrame(frame);
  if (SHOT) {
    document.body.classList.add('shot');
    let n = 0; const wait = () => { if (++n > 6) { window.__ready = true; } else requestAnimationFrame(wait); }; requestAnimationFrame(wait);
    return;
  }
  document.body.classList.add('loaded');
  const go = $('go'); if (go) { go.disabled = false; go.focus(); go.addEventListener('click', start); }
  canvas.addEventListener('click', () => { if (started) player.requestLock(); });
  document.addEventListener('pointerlockchange', () => { document.body.classList.toggle('locked', document.pointerLockElement === canvas); });
  addEventListener('keydown', (e) => {
    if (e.code === 'Enter' && !started) start();
    if (!started) return;
    if (e.code === 'KeyH') document.body.classList.toggle('noui');
    if (e.code === 'KeyR') applyView(town.views[0]);
    if (e.code === 'Backquote') { const s = $('stats'); if (s) s.hidden = !s.hidden; }
    const m = /^Digit([1-9])$/.exec(e.code); if (m) window.__view(Number(m[1]) - 1);
  });
  wireMenus();
  if (params.has('stats')) $('stats').hidden = false;
  if (errors.length) console.warn('module errors', errors);
}

function wireMenus() {
  const q = $('quality');
  if (q) {
    q.value = qName;
    q.addEventListener('change', () => {
      if (PILOT) { const u = new URL(location.href); u.searchParams.set('q', q.value); location.href = u.toString(); return; }
      try { localStorage.setItem('keiview.q', q.value); } catch (e) {}
      location.reload();
    });
  }
  const tsel = $('townsel');
  if (tsel) { tsel.value = SEED; tsel.addEventListener('change', () => { const u = new URL(location.href); u.searchParams.set('town', tsel.value); location.href = u.toString(); }); }
}
main();
