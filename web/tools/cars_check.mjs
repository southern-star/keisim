// Checks of the car redesign (CarRedesign_Handoff.md, chapter 6) on the cars of ?cardetail=2 (src/world/cars2.js).
//   node tools/cars_check.mjs [--n 500] [--no-browser] [--base origin/main] [--gl hw|soft]
// Without a browser (--no-browser): 2 the box, 3 repeatability, 4 the shares of the body types (printed: the new
// types come in P3), 5 the plan area, and the budget (triangles, meshes against ?cardetail=1).
// With one: 1 the default pictures against the base tree (tools/golden.mjs) and 6 the labels of ?cardetail=2.
// Not yet: 7 wheels (P4), 8 speed (P5). Body types that are not lofted yet are counted, not judged.
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { rng, sampleDims, sampleColor, compare, launch, serveTree, frames } from './golden.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] !== undefined && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
const N = Number(args.n || 500);
const { buildCar2, carType2, SUBTYPES } = await import(pathToFileURL(path.join(root, 'src/world/cars2.js')).href);
const TRUCKS = new Set(['keitruck', 'truck', 'flatbed', 'reefer', 'delivery']);   // their beds and boxes are boxes
const { buildCar } = await import(pathToFileURL(path.join(root, 'src/world/cars.js')).href);

let failed = false;
const line = (ok, name, text) => { if (!ok) failed = true; console.log(`${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(16)} ${text}`); };

// ---- the cars: N of KeiSim's sizes and colours
const r = rng(20261121), cars = [];
for (let id = 0; id < N; id++) { const [L, W, H, kind] = sampleDims(r); cars.push({ id, kind, L, W, H, rgb: sampleColor(r) }); }
const geos = (c) => c.parts.map(([g, key, o]) => ({ g, key, extra: !!(o && o.extra) })).concat(c.tails.map((g) => ({ g, key: 'tail', extra: false })));
const tris = (c) => geos(c).reduce((n, { g }) => n + (g.index ? g.index.count : g.attributes.position.count) / 3, 0);
const sig = (c) => {
  let h = 2166136261;
  for (const { g, key } of geos(c)) {
    const p = g.attributes.position.array;
    for (let i = 0; i < key.length; i++) h = Math.imul(h ^ key.charCodeAt(i), 16777619);
    for (let i = 0; i < p.length; i += 7) h = Math.imul(h ^ Math.round(p[i] * 1e4), 16777619);
    h = Math.imul(h ^ p.length, 16777619);
  }
  return `${c.type}|${c.face || ''}|${c.trim || ''}|${tris(c)}|${h >>> 0}`;
};

const types = new Map(), lofted = [];
let worst = 0, worstCar = null, maxTris = 0, nan = 0, repeat = 0, looks = new Map();
const byType = new Map(), faces = new Map(), trims = new Map();
let withDriver = 0, lessEmpty = 0;
for (const c of cars) {
  const m = buildCar2(c.kind, c.L, c.W, c.H, c.rgb, c.id);
  types.set(m.type, (types.get(m.type) || 0) + 1);
  const s1 = sig(m);
  if (s1 !== sig(buildCar2(c.kind, c.L, c.W, c.H, c.rgb, c.id))) repeat++;
  if (!m.lofted) continue;
  lofted.push({ c, m });
  looks.set(s1, (looks.get(s1) || 0) + 1);
  maxTris = Math.max(maxTris, tris(m));
  if (!byType.has(m.type)) byType.set(m.type, { n: 0, tris: 0, plan: [] });
  const bt = byType.get(m.type); bt.n++; bt.tris = Math.max(bt.tris, tris(m));
  const fk = `${m.type} ${m.face.split('/')[0].split('+').slice(0, 2).join('+')}`, rk = `${m.type} /${m.face.split('/')[1].split('+')[0]}`;
  faces.set(fk, (faces.get(fk) || 0) + 1); faces.set(rk, (faces.get(rk) || 0) + 1);
  {                                                                  // wheels and paint by type; a parked car: nobody at the wheel
    const [wheel, roof, pillar] = m.trim.split('|');
    const paint = roof !== 'body' ? (roof === 'plateW' ? 'white roof' : 'black roof') : pillar !== 'body' ? 'black pillars' : 'one colour';
    for (const k of [`${m.type} wheels: ${wheel}`, `${m.type} paint: ${paint}`]) trims.set(k, (trims.get(k) || 0) + 1);
    const empty = buildCar2(c.kind, c.L, c.W, c.H, c.rgb, c.id, { driver: false });
    withDriver++; if (tris(empty) < tris(m) && empty.trim.endsWith('-') && m.trim.endsWith('d')) lessEmpty++;
  }
  // 2: the box of everything but the marked extras (door mirrors, roof fittings)
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const { g, extra } of geos(m)) {
    const p = g.attributes.position.array, n = g.attributes.normal.array;
    for (let i = 0; i < p.length; i++) {
      if (!Number.isFinite(p[i]) || !Number.isFinite(n[i])) nan++;
      if (extra) continue;
      const k = i % 3; if (p[i] < lo[k]) lo[k] = p[i]; if (p[i] > hi[k]) hi[k] = p[i];
    }
  }
  const want = [[-c.W / 2, 0, -c.L / 2], [c.W / 2, c.H, c.L / 2]];
  const err = Math.max(...[0, 1, 2].map((k) => Math.max(Math.abs(lo[k] - want[0][k]), Math.abs(hi[k] - want[1][k]))));
  if (err > worst) { worst = err; worstCar = c; }
  m.fitMax = Math.max(...m.fit.map((v) => Math.abs(v - 1)));
}
const share = [...types].sort((a, b) => b[1] - a[1]).map(([t, n]) => `${t} ${(100 * n / N).toFixed(1)} %`).join(', ');
console.log(`${N} cars of KeiSim's sizes: ${lofted.length} lofted (${[...new Set(lofted.map((x) => x.m.type))].join(', ') || 'none'}), the others still cars.js`);
line(lofted.length > 0 && worst <= 0.01 && nan === 0, '2 box', `largest distance of a lofted body's box from L x W x H: ${(worst * 1000).toFixed(1)} mm`
  + (worstCar ? ` (car ${worstCar.id}: ${worstCar.L.toFixed(2)} x ${worstCar.W.toFixed(2)} x ${worstCar.H.toFixed(2)})` : '') + (nan ? `, ${nan} non-finite numbers` : '')
  + `; largest fitting correction ${(100 * Math.max(0, ...lofted.map((x) => x.m.fitMax))).toFixed(2)} %`);
line(repeat === 0, '3 repeatable', `${N - repeat}/${N} cars come out the same when built twice (type, face, wheels, paint, plate, triangles, vertices)`);
line(lessEmpty === withDriver, 'driver', `${lessEmpty}/${withDriver} cars lose their driver's triangles when built as parked (opt.driver: false)`);
// 4: every type of the table among the N cars; its share of its size class within 8 points of the table. The shares
// are judged on 20,000 cars (types only: no geometry), as a class of 30 cars scatters by more than 8 points by chance;
// the N cars' own shares are printed beside them.
{
  const all = Object.values(SUBTYPES).flat().map((e) => e[0]), missing = [...new Set(all)].filter((t) => !types.has(t));
  const tally = (list) => { const m = new Map(); for (const c of list) { const { base, type } = carType2(c.kind, c.L, c.W, c.H, c.rgb, c.id); if (!m.has(base)) m.set(base, new Map()); const b = m.get(base); b.set(type, (b.get(type) || 0) + 1); } return m; };
  const r2 = rng(424242), big = [];
  for (let id = 0; id < 20000; id++) { const [L, W, H, kind] = sampleDims(r2); big.push({ id, kind, L, W, H, rgb: sampleColor(r2) }); }
  const tb = tally(big), ts = tally(cars);
  let worst = 0, worstN = 0; const rows = [];
  for (const [base, list] of Object.entries(SUBTYPES)) {
    const nb = [...(tb.get(base) || new Map()).values()].reduce((a, v) => a + v, 0), ns = [...(ts.get(base) || new Map()).values()].reduce((a, v) => a + v, 0);
    const parts = list.map(([type, want]) => {
      const got = 100 * ((tb.get(base) || new Map()).get(type) || 0) / Math.max(nb, 1), small = 100 * ((ts.get(base) || new Map()).get(type) || 0) / Math.max(ns, 1);
      worst = Math.max(worst, Math.abs(got - 100 * want)); worstN = Math.max(worstN, Math.abs(small - 100 * want));
      return `${type} ${got.toFixed(1)} % (table ${(100 * want).toFixed(0)}, of the ${N}: ${small.toFixed(0)})`;
    });
    rows.push(`       ${base.padEnd(9)} ${String(ns).padStart(3)} of the ${N}: ${parts.join(', ')}`);
  }
  line(missing.length === 0 && worst <= 8, '4 types', `${types.size} of ${new Set(all).size} body types among the ${N} cars${missing.length ? ` (missing: ${missing.join(', ')})` : ''}; shares within their size class off the table by at most ${worst.toFixed(1)} points on 20,000 cars (wanted: 8; the ${N} cars alone: ${worstN.toFixed(1)})`);
  for (const r of rows) console.log(r);
}

// 5: the plan seen from straight above, as a share of L x W (1 cm cells)
function planShare({ c, m }) {
  const res = 0.01, nx = Math.ceil(c.W / res) + 20, nz = Math.ceil(c.L / res) + 20, x0 = -c.W / 2 - 0.1, z0 = -c.L / 2 - 0.1;
  const cell = new Uint8Array(nx * nz);
  for (const { g, key, extra } of geos(m)) {
    if (extra || key === 'tire' || key === 'hub') continue;
    const p = g.attributes.position.array;
    for (let i = 0; i < p.length; i += 9) {
      const ax = (p[i] - x0) / res, az = (p[i + 2] - z0) / res, bx = (p[i + 3] - x0) / res, bz = (p[i + 5] - z0) / res, cx = (p[i + 6] - x0) / res, cz = (p[i + 8] - z0) / res;
      const d = (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
      if (Math.abs(d) < 1e-9) continue;
      for (let u = Math.max(0, Math.floor(Math.min(ax, bx, cx))); u <= Math.min(nx - 1, Math.ceil(Math.max(ax, bx, cx))); u++) {
        for (let v = Math.max(0, Math.floor(Math.min(az, bz, cz))); v <= Math.min(nz - 1, Math.ceil(Math.max(az, bz, cz))); v++) {
          const px = u + 0.5, pz = v + 0.5;
          const w0 = ((bx - px) * (cz - pz) - (bz - pz) * (cx - px)) / d, w1 = ((cx - px) * (az - pz) - (cz - pz) * (ax - px)) / d;
          if (w0 >= 0 && w1 >= 0 && w0 + w1 <= 1) cell[v * nx + u] = 1;
        }
      }
    }
  }
  let n = 0; for (const v of cell) n += v;
  return n * res * res / (c.L * c.W);
}
const seen = new Map(), plans = [];
for (const x of lofted) {
  const k = seen.get(x.m.type) || 0; seen.set(x.m.type, k + 1);
  if (k >= 12) continue;                                             // 12 cars of every type
  const p = planShare(x); byType.get(x.m.type).plan.push(p);
  if (!TRUCKS.has(x.m.type)) plans.push(p);
}
if (plans.length) {
  const lo = Math.min(...plans), hi = Math.max(...plans);
  line(lo >= 0.90 && hi <= 0.97, '5 plan', `the body from above covers ${(100 * lo).toFixed(1)} .. ${(100 * hi).toFixed(1)} % of L x W on ${plans.length} cars of the ${[...byType.keys()].filter((t) => !TRUCKS.has(t)).length} car types (wanted: 90 .. 97 %)`);
}
for (const [t, b] of byType) console.log(`       ${t.padEnd(9)} ${String(b.n).padStart(3)} cars, at most ${b.tris} triangles, plan ${(100 * Math.min(...b.plan)).toFixed(1)} .. ${(100 * Math.max(...b.plan)).toFixed(1)} %${TRUCKS.has(t) ? '  (bed / box are boxes: not judged)' : ''}`);
// budget: triangles, and meshes (one per material key, the lamps apart) against today's detailed car
const meshes = (c) => new Set(c.parts.map((p) => p[1])).size + (c.tails.length ? 1 : 0);
const extraMeshes = Math.max(0, ...lofted.map(({ c, m }) => meshes(m) - meshes(buildCar(c.kind, c.L, c.W, c.H, c.rgb))));
line(maxTris <= 6000 && extraMeshes <= 5, 'budget', `at most ${maxTris} triangles a lofted car (guide: 6,000); ${extraMeshes} more meshes than ?cardetail=1 at most (guide: 5)`);
const same = [...looks.values()].filter((n) => n > 1).reduce((a, n) => a + n, 0);
{
  const ft = new Map();
  for (const [k, n] of [...faces].sort()) { const [t, f] = k.split(' '); if (!ft.has(t)) ft.set(t, []); ft.get(t).push(`${f} ${n}`); }
  for (const [t, a] of ft) console.log(`       faces ${t.padEnd(9)} ${a.join(', ')}`);
  const tt = new Map();
  for (const [k, n] of [...trims].sort()) { const i = k.indexOf(' '), t = k.slice(0, i); if (!tt.has(t)) tt.set(t, []); tt.get(t).push(`${k.slice(i + 1)} ${n}`); }
  for (const [t, a] of tt) console.log(`       trims ${t.padEnd(9)} ${a.join(', ')}`);
}
line(same <= 0.03 * lofted.length, '4 look-alikes', `${same} of ${lofted.length} cars share their whole geometry with another (wanted: 3 % at most)`);
{
  const w = new Map();
  for (const { m } of lofted) { const e = m.trim.split('|')[4]; for (const k of e ? e.split('+') : []) w.set(k, (w.get(k) || 0) + 1); }
  console.log(`       working cars and small things among the ${N}: ${[...w].sort().map(([k, n]) => `${k} ${n}`).join(', ') || 'none'}`);
}

// ---- with a browser: the default pictures and the labels
if (!args['no-browser']) {
  const g = await compare({ base: args.base, gl: args.gl });
  for (const l of g.lines) console.log(`     ${l}`);
  line(g.ok, '1 default views', 'ego mode\'s default and ?cardetail=1 against the base tree, RGB and labels');
  const browser = await launch(args.gl || 'hw'), server = serveTree(root);
  await new Promise((res) => server.listen(0, '127.0.0.1', res));
  // with the glass of cars.js and with the see-through glass (src/world/carglass.js): a pane must stay a vehicle
  for (const glass of ['solid', 'clear']) try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/index.html?shot=1&ego=1&town=1000&w=320&h=160&q=medium&cardetail=2&carglass=${glass}`, { waitUntil: 'load', timeout: 180000 });
    await page.waitForFunction('window.__ready === true', { timeout: 900000, polling: 100 });
    const [x0, y0, errs] = await page.evaluate(() => { const T = window.__town, c = T.views[0]; return [c.x + T.origin[0], T.origin[1] - c.z, window.__errors]; });
    if (errs.length) throw new Error(`page errors: ${JSON.stringify(errs)}`);
    let bad = 0, carPx = 0, withCars = 0, loftedSeen = 0;
    for (const q of frames(6, x0, y0)) {
      const sem = async (veh) => Buffer.from(await page.evaluate((req) => window.__ego(req).sem, { ...q, veh, ped: [], bike: [] }), 'base64');
      const a = await sem(q.veh);
      loftedSeen += await page.evaluate(() => [...window.__ctx.services.actors.cars.values()].filter((s) => s.g.visible && s.detail && s.detail.visible).length);
      const b = await sem([]);
      let n = 0;
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { n++; if (a[i] !== 11) bad++; }
      carPx += n; if (n > 50) withCars++;
    }
    if (glass === 'clear') {
      const v = frames(1, x0, y0)[0].veh[0], row = (id) => [id, ...v.slice(1)];
      const n = await page.evaluate((a, b) => {
        const A = window.__ctx.services.actors; A.update([a, b], [], []);
        const count = (id) => { let k = 0; A.cars.get(id).detail.traverse((o) => { if (o.isMesh) k += o.geometry.attributes.position.count / 3; }); return k; };
        return [count(a[0]), count(b[0])];
      }, row(7), row(100007));
      line(n[1] < n[0], 'driver, parked', `the same car as id 7 and as a parked car's id 100007: ${n[0]} and ${n[1]} triangles (the parked one has nobody at the wheel)`);
    }
    await page.close();
    line(bad === 0 && withCars >= 5 && loftedSeen > 0, `6 labels ${glass}`, `${carPx} pixels change when the cars of ?cardetail=2 are drawn, ${bad} of them not class 11 (vehicle); cars in ${withCars} of 6 frames, ${loftedSeen} detailed models shown`);
  } catch (e) { line(false, `6 labels ${glass}`, String(e.message || e)); }
  await browser.close(); server.close();
}
console.log('     7 wheels, 8 speed: not yet (P4, P5)');
console.log(failed ? 'RESULT: FAIL' : 'RESULT: OK');
process.exit(failed ? 1 : 0);
