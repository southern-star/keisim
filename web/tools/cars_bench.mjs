// Frame time of the detailed cars in a town: today's (cars.js, ?cardetail=1) against the redesign (cars2.js,
// ?cardetail=2) with each glass (src/world/carglass.js). One page of ego mode, one camera, 24 cars driving and 12
// parked within 60 m; every car carries both detailed models and the page switches between them, so the
// configurations take turns frame block after frame block and share whatever else the computer is doing.
//   node tools/cars_bench.mjs [--town 1000] [--w 1280 --h 720] [--q high] [--frames 40] [--rounds 25] [--gl hw|soft]
// Prints ms per frame (median of the rounds, and the fastest round), draw calls and triangles.
// CarRedesign_Handoff.md asks for this in pilot mode (criterion 8); this is the same scene load without the model.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { launch, serveTree, rng, sampleDims, sampleColor } from './golden.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] !== undefined && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
const W = Number(args.w || 1280), H = Number(args.h || 720), N = Number(args.frames || 40), R = Number(args.rounds || 25);

/** 24 cars in three lanes ahead of the camera and 12 parked beside them, 6 .. 60 m away. */
function scene(x0, y0) {
  const r = rng(77), yaw = 0.3, c = Math.cos(yaw), s = Math.sin(yaw), veh = [];
  const at = (f, l) => [x0 + c * f - s * l, y0 + s * f + c * l];
  for (let i = 0; i < 36; i++) {
    const [L, Wd, Hh, kind] = sampleDims(r), parked = i >= 24;
    const row = parked ? i - 24 : Math.floor(i / 3), lane = parked ? (i % 2 ? 7.5 : -7.5) : (i % 3 - 1) * 3.5;
    const [x, y] = at(6 + (parked ? 4.5 : 6.5) * row + 2 * r(), lane + 0.6 * r());
    veh.push([i, x, y, yaw + (!parked && i % 3 === 0 ? Math.PI : 0), L, Wd, Hh, ...sampleColor(r), kind, i % 5 === 0 ? 1 : 0]);
  }
  return { cam: { pos: [x0, y0, 1.3], fwd: [c, s, -0.05], vfov: 61.6 }, t: 0, veh, ped: [], bike: [], labels: false, reset: true };
}

const browser = await launch(args.gl || 'hw', W, H), server = serveTree(root);
await new Promise((res) => server.listen(0, '127.0.0.1', res));
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html?shot=1&ego=1&town=${args.town || '1000'}&w=${W}&h=${H}&q=${args.q || 'high'}&cardetail=2`, { waitUntil: 'load', timeout: 180000 });
  await page.waitForFunction('window.__ready === true', { timeout: 900000, polling: 100 });
  const [x0, y0] = await page.evaluate(() => { const T = window.__town, c = T.views[0]; return [c.x + T.origin[0], T.origin[1] - c.z]; });
  const res = await page.evaluate((req, n, rounds) => {
    window.__ego(req);
    const ctx = window.__ctx, A = ctx.services.actors, cache = ctx.mat.cache;
    const gl = document.querySelector('canvas').getContext('webgl2'), ext = gl.getExtension('WEBGL_debug_renderer_info');
    const glass = { solid: cache.get('glass|car'), tint: cache.get('glass|car2:tint'), clear: cache.get('glass|car2:clear') };
    const isGlass = (m) => Object.values(glass).includes(m.material);
    const cars = [];
    for (const v of req.veh) {
      const s = A.cars.get(v[0]);
      const old = A.detailModel(1, v[10], v[4], v[5], v[6], [v[7], v[8], v[9]], v[0]).g;      // today's model beside the new one
      s.g.add(old);
      cars.push({ plain: s.plain, now: s.detail, old, panes: s.detail.children.filter(isGlass) });
    }
    const set = (which, kind) => {
      for (const c of cars) {
        c.plain.visible = which === 'plain'; c.old.visible = which === 'old'; c.now.visible = which === 'now';
        if (kind) for (const m of c.panes) { m.material = glass[kind]; m.layers.set(kind === 'solid' ? 0 : 1); }
      }
    };
    const configs = [['no cars', () => set('none')], ['plain cars', () => set('plain')], ['today (cars.js)', () => set('old')],
      ['redesign, solid glass', () => set('now', 'solid')], ['redesign, tint glass', () => set('now', 'tint')], ['redesign, clear glass', () => set('now', 'clear')]];
    const out = configs.map(([name]) => ({ name, ms: [] }));
    for (const [, fn] of configs) { fn(); window.__bench(5); }                                 // compile everything first
    for (let k = 0; k < rounds; k++) configs.forEach(([, fn], i) => { fn(); const b = window.__bench(n); out[i].ms.push(b.ms); out[i].calls = b.calls; out[i].tris = b.triangles; });
    return { gpu: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : '?', out, errors: window.__errors };
  }, scene(x0, y0), N, R);
  if (res.errors.length) throw new Error(JSON.stringify(res.errors));
  console.log(`GPU: ${res.gpu}; ${W} x ${H}, 36 cars, ${N} frames a round, ${R} rounds taking turns`);
  const med = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
  const none = med(res.out[0].ms), today = med(res.out[2].ms);
  for (const [i, o] of res.out.entries()) {
    const m = med(o.ms);
    console.log(`${o.name.padEnd(22)} median ${m.toFixed(2)} ms/frame (fastest ${Math.min(...o.ms).toFixed(2)})  the cars ${(m - none).toFixed(2)} ms`
      + `${i > 2 ? `  ${m >= today ? '+' : ''}${(100 * (m / today - 1)).toFixed(1)} % of the frame against today` : ''}  calls ${o.calls}, triangles ${o.tris}`);
  }
} finally { await browser.close(); server.close(); }
