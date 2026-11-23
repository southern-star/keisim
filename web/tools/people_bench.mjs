// Frame time of the detailed pedestrians (src/world/people.js, ?peddetail=1) against the plain ones, in a town:
// PedRedesign_Handoff.md criterion 6 (20 people within 45 m; at most 3 ms more on an RTX 3060). One page of ego mode
// with ?peddetail=1: every person carries both models and the page switches between them (view('model') shows the
// plain ones, view('main') the detailed ones), so the two take turns block after block and share whatever else the
// computer is doing. Three figures: drawing a frame of people who stand still (GPU), actors.update() with moving
// phases, which poses the detailed people (CPU), and a whole ego frame of walking people as KeiSim asks for it
// (window.__ego: update, draw, read the picture back), which is what criterion 6 is about.
//   node tools/people_bench.mjs [--town 1000] [--w 1280 --h 720] [--q high] [--people 20] [--frames 40] [--rounds 25] [--gl hw|soft]
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { launch, serveTree } from './golden.mjs';
import { findRoad, street } from './people_scene.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] !== undefined && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
const W = Number(args.w || 1280), H = Number(args.h || 720), N = Number(args.frames || 40), R = Number(args.rounds || 25), P = Number(args.people || 20);

const browser = await launch(args.gl || 'hw', W, H), server = serveTree(root);
await new Promise((res) => server.listen(0, '127.0.0.1', res));
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html?shot=1&ego=1&town=${args.town || '1000'}&w=${W}&h=${H}&q=${args.q || 'high'}&peddetail=1`, { waitUntil: 'load', timeout: 180000 });
  await page.waitForFunction('window.__ready === true', { timeout: 900000, polling: 100 });
  const road = await page.evaluate(findRoad), S = street(road, { near: P, far: 0 });
  const res = await page.evaluate((q, n, rounds) => {
    window.__ego(q);
    const ctx = window.__ctx, A = ctx.services.actors, gl = document.querySelector('canvas').getContext('webgl2'), ext = gl.getExtension('WEBGL_debug_renderer_info');
    const c = new window.THREE.Vector3(); ctx.camera.getWorldPosition(c);
    const dist = [...A.peds.values()].map((s) => s.g.position.distanceTo(c));
    const none = () => { for (const s of A.peds.values()) s.g.visible = false; };
    const all = () => { for (const s of A.peds.values()) s.g.visible = true; };
    const configs = [['no people', () => { none(); }], ['plain', () => { all(); A.view('model'); }], ['detailed', () => { all(); A.view('main'); }]];
    const out = configs.map(([name]) => ({ name, ms: [], upd: [], ego: [] }));
    let tt = q.t;
    for (const [, fn] of configs) { fn(); window.__bench(5); }
    let ph = 0;
    for (let k = 0; k < rounds; k++) configs.forEach(([, fn], i) => {
      fn();
      const b = window.__bench(n); out[i].ms.push(b.ms); out[i].calls = b.calls; out[i].tris = b.triangles;
      if (i === 0) return;
      // actors.update() with the phases moving on: the detailed people are posed (the plain ones swing two boxes)
      const t0 = performance.now();
      for (let j = 0; j < 50; j++) { ph += 0.3; ctx.time = (ctx.time || 0) + 0.05; A.update([], q.ped.map((p) => { const r = p.slice(); r[14] = p[14] + ph; return r; }), []); }
      out[i].upd.push((performance.now() - t0) / 50);
      // whole ego frames of walking people (view('model') / view('main') stays as set)
      const t1 = performance.now();
      for (let j = 0; j < n; j++) { ph += 0.3; tt += 0.05; window.__ego({ cam: q.cam, t: tt, veh: [], ped: q.ped.map((p) => { const r = p.slice(); r[14] = p[14] + ph; return r; }), bike: [], labels: false }); }
      out[i].ego.push((performance.now() - t1) / n);
    });
    return { gpu: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : '?', out, near: dist.filter((d) => d < 45).length, far: Math.max(...dist), errors: window.__errors };
  }, { cam: S.cam, t: 5, veh: [], ped: S.ped, bike: [], labels: false, reset: true }, N, R);
  if (res.errors.length) throw new Error(JSON.stringify(res.errors));
  console.log(`GPU: ${res.gpu}; ${W} x ${H}, ${res.near} people within 45 m (the farthest ${res.far.toFixed(0)} m), ${N} frames a round, ${R} rounds taking turns`);
  const med = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
  for (const o of res.out) {
    console.log(`${o.name.padEnd(10)} frame: median ${med(o.ms).toFixed(2)} ms (fastest ${Math.min(...o.ms).toFixed(2)})  calls ${o.calls}, triangles ${o.tris}`
      + (o.upd.length ? `  update: median ${med(o.upd).toFixed(3)} ms  ego frame, walking: median ${med(o.ego).toFixed(2)} ms (fastest ${Math.min(...o.ego).toFixed(2)})` : ''));
  }
  const [, pl, de] = res.out;
  console.log(`detailed - plain: frame ${(med(de.ms) - med(pl.ms)).toFixed(2)} ms by the medians, ${(Math.min(...de.ms) - Math.min(...pl.ms)).toFixed(2)} ms by the fastest rounds; update ${(med(de.upd) - med(pl.upd)).toFixed(3)} ms; ego frame, walking ${(med(de.ego) - med(pl.ego)).toFixed(2)} ms by the medians, ${(Math.min(...de.ego) - Math.min(...pl.ego)).toFixed(2)} ms by the fastest rounds`
    + '  (the handoff asks for at most 3 ms on an RTX 3060)');
} finally { await browser.close(); server.close(); }
