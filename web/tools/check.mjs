// Headless (node) build check for KeiView modules — no GPU, no browser. Adapted from Sakuragaoka
// Station tools/check.mjs (MIT): stubbed DOM/canvas, builds the modules for a town, runs update()
// for a while and prints errors, triangles, meshes, materials and build time per module.
//   node tools/check.mjs [--town 1000] [module ...]        (default: every module in src/main.js order)
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const ti = argv.indexOf('--town'); const seed = ti >= 0 ? argv[ti + 1] : '1000';
const names = argv.filter((a, i) => !a.startsWith('--') && argv[i - 1] !== '--town');

// ---------------------------------------------------------------- DOM stubs (as upstream)
class Ctx2D {
  constructor(c) { this.canvas = c; this.font = '10px sans-serif'; }
  measureText(t) { const m = /(\d+(?:\.\d+)?)px/.exec(this.font); const s = m ? Number(m[1]) : 10; return { width: [...String(t)].length * s * 0.92, actualBoundingBoxAscent: s * 0.8, actualBoundingBoxDescent: s * 0.2 }; }
  createLinearGradient() { return { addColorStop() {} }; }
  createRadialGradient() { return { addColorStop() {} }; }
  createConicGradient() { return { addColorStop() {} }; }
  createPattern() { return {}; }
  getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(Math.max(1, w * h * 4)), width: w, height: h }; }
  createImageData(w, h) { return { data: new Uint8ClampedArray(Math.max(1, w * h * 4)), width: w, height: h }; }
  putImageData() {} getTransform() { return { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }; }
  isPointInPath() { return false; }
  getLineDash() { return []; }
}
const ctxProxy = (c) => new Proxy(new Ctx2D(c), { get(t, k) { if (k in t) return t[k]; return () => {}; }, set(t, k, v) { t[k] = v; return true; } });
class FakeCanvas {
  constructor() { this.width = 300; this.height = 150; this.style = {}; this._ctx = null; }
  getContext(type) { if (type !== '2d') return null; return this._ctx || (this._ctx = ctxProxy(this)); }
  toDataURL() { return 'data:,'; } addEventListener() {} removeEventListener() {}
}
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.document = {
  createElement: (t) => (t === 'canvas' ? new FakeCanvas() : { style: {}, addEventListener() {}, appendChild() {}, setAttribute() {} }),
  createElementNS: (ns, t) => (t === 'canvas' ? new FakeCanvas() : { style: {} }),
  fonts: { load: async () => [] }, body: { appendChild() {} }, addEventListener() {},
};
globalThis.Image = class { constructor() { this.width = 1; this.height = 1; } addEventListener() {} };
globalThis.HTMLCanvasElement = FakeCanvas; globalThis.OffscreenCanvas = FakeCanvas;
globalThis.requestAnimationFrame = (f) => setTimeout(() => f(Date.now()), 16);
globalThis.addEventListener = () => {};
globalThis.innerWidth = 1280; globalThis.innerHeight = 720; globalThis.devicePixelRatio = 1;
globalThis.matchMedia = () => ({ matches: false, addEventListener() {} });
const quietWarn = console.warn; console.warn = (...a) => { const s = String(a[0]); if (!/ImageUtils|toNonIndexed/.test(s)) quietWarn(...a); };
const quietLog = console.log; const log = (...a) => quietLog(...a); console.log = (...a) => { const s = String(a[0]); if (!/ImageUtils|toNonIndexed/.test(s)) quietLog(...a); };

// ---------------------------------------------------------------- build
const imp = (p) => import(pathToFileURL(path.join(root, p)).href);
const THREE = await import('three');
const L = await imp('vendor/sakuragaoka/world/layout.js');
const { createContext } = await imp('vendor/sakuragaoka/core/ctx.js');
const { batchStatic } = await imp('vendor/sakuragaoka/core/batch2.js');
const { prepareTown } = await imp('src/town.js');
// module order as in src/main.js (not imported: main.js needs a real browser)
const MODULES = JSON.parse(/export const MODULES = (\[[^\]]*\])/.exec(fs.readFileSync(path.join(root, 'src/main.js'), 'utf8'))[1].replace(/'/g, '"'));

const town = prepareTown(JSON.parse(fs.readFileSync(path.join(root, `towns/town_${seed}.json`), 'utf8')));
L.configureWorld({ play: town.play, heightAt: town.heightAt, farTown: town.farTown, farSkip: town.farSkip });
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(58, 16 / 9, 0.1, 2500);
const audio = { start() {}, update() {}, loop: () => ({ setVolume() {}, setParam() {}, setPosition() {}, stop() {} }), play() {} };
const quality = { name: 'high', pixelRatio: 1, msaa: 4, shadowMap: 4096, shadowSize: 75, petals: 1 };
const ctx = createContext({ scene, camera, renderer: null, audio, quality, sunDir: new THREE.Vector3(...L.SUN_DIR).normalize() });
ctx.sky = { sun: new THREE.DirectionalLight(), hemi: new THREE.HemisphereLight(), uniforms: {} };
ctx.town = town;

const triCount = (o) => { const g = o.geometry; if (!g || !g.attributes.position) return 0; return (g.index ? g.index.count / 3 : g.attributes.position.count / 3) * (o.isInstancedMesh ? o.count : 1); };
let failed = false, total = 0;
log(`town ${seed}: ${town.data.summary}  (surface grid ${town.grid.ok ? 'rasterised' : 'stubbed: no canvas in node'})`);
for (const name of names.length ? names : MODULES) {
  const before = new Set(); scene.traverse(o => before.add(o));
  const upd0 = ctx._updates.length, t0 = Date.now();
  let err = null;
  try { const mod = await imp(`src/world/${name}.js`); await mod.build(ctx); } catch (e) { err = e; failed = true; }
  const ms = Date.now() - t0;
  let tris = 0, meshes = 0, nan = 0; const mats = new Set();
  scene.updateMatrixWorld(true);
  scene.traverse(o => {
    if (before.has(o) || !(o.isMesh || o.isLine || o.isPoints)) return;
    meshes++; tris += triCount(o); (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m && mats.add(m));
    const a = o.geometry?.attributes?.position?.array; if (a) for (let i = 0; i < a.length; i += 97) if (!Number.isFinite(a[i])) { nan++; break; }
  });
  let updErr = null;
  try { for (let i = 0, t = 0; i < 300; i++, t += 1 / 30) for (const f of ctx._updates.slice(upd0)) f(1 / 30, t); } catch (e) { updErr = e; failed = true; }
  total += tris;
  log(`${err || updErr ? 'FAIL' : 'ok  '} ${name.padEnd(10)} ${String(ms).padStart(5)} ms  ${String(Math.round(tris)).padStart(8)} tris  ${String(meshes).padStart(4)} meshes  ${String(mats.size).padStart(3)} mats${nan ? `  NaN in ${nan} meshes!` : ''}`);
  if (err) log('   build error:', err.stack.split('\n').slice(0, 6).join('\n   '));
  if (updErr) log('   update error:', updErr.stack.split('\n').slice(0, 6).join('\n   '));
  if (nan) failed = true;
}
try { const b = batchStatic(ctx.staticRoot, { mat: ctx.mat }); log(`batch: ${b.sources} static meshes -> ${b.merged}`); } catch (e) { failed = true; log('batch error:', e.stack.split('\n').slice(0, 4).join('\n  ')); }
log(`total ${Math.round(total)} tris, colliders ${ctx.physics.count}`);
log(failed ? 'RESULT: FAIL' : 'RESULT: OK');
process.exit(failed ? 1 : 0);
