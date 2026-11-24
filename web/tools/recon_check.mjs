// ReconView's fidelity (index.html?pilot=mc&recon=oracle; docs/recon_view.md): opens the page in headless Chrome,
// drives `--sec` seconds through window.__pilotRun and, every `--every` seconds, compares the reconstruction with the
// town itself from the chase camera (8 m behind the car, 4 m up; 320x160 depth images of the same model frame):
//   coverage     of the town's pixels within 80 m, the share the reconstruction draws
//   seen         the same over the pixels a camera of the rig sees in that frame (not covered there: seen, but not built)
//   error        median |R - G| / G where both have a surface
//   wrong        of the pixels the reconstruction draws, the share with no surface in the town or more than 25 % off:
//                through (the town's surface is nearer: something the cameras did not see hides the place from the chase
//                camera) + phantom (the town's is farther or none: a triangle stretched between two objects)
//   node tools/recon_check.mjs [--town 1000] [--sec 60] [--every 2] [--recon oracle]
//                              [--recon model]   (?recon=model: the model's own depth, models/keipilot_mc_depth.onnx; every
//                              measurement then also has the same frame with the town's depth, under "oracle")
//                              [--layout split]   (&layout=split: the main view from the chase camera beside the view)
//                              [--accum 0|1]   (&accum=: the keyframes of passed places, on by default with --recon model;
//                              every measurement then has the current frame alone at the top and, under "accum", the
//                              same frame with the keyframes; the pictures: _recon.png with them, _current.png without)
//                              [--keep noveg|nobuild|flat|ground] [--far 14]   (&accumkeep=, &accumfar=: what of the
//                              keyframes the view draws, by default every static class within 10 m; --far 14: all a
//                              keyframe holds, as G4 was measured; every measurement has all of src/recon.js ACCUM_VARIANTS on
//                              the same frame under accum.variants, the pictures as _acc_<name>.png)
//                              [--fill oracle|model] [--fillshow all] [--filltint 1]   (&fill=: what the view leaves dark is
//                              filled in, src/fill.js, docs/recon_fill.md; every measurement then has, under "fill", the
//                              same frame with this frame's fill and with the one before it, the holes only and all of the
//                              fill alone; the pictures: _fill_holes.png, _fill_all.png, _fill_holesTint.png, _fill_image.png
//                              (the fill's own image) and the maps _fill_<name>Map.png; --fill model: the fill of
//                              models/reconfill.onnx, and under fill.oracle the town's own fill of the same frame)
//                              [--fillmodel models/x.onnx] [--fillworker own]   (&fillmodel=, &fillworker=: another fill
//                              model; the fill model in a worker of its own, not in the driving model's)
//                              [--measure 0]   (no measurements: the page draws what a visitor's draws; the digests only)
//                              [--size 1920x1080]   (the window; 1280x720) [--name NAME]   (--out DIR/NAME.json)
//                              [--recon off]   (the page without ?recon: checks that it loads nothing of ReconView and
//                              drives the same way; no measurements, the digest only)
//                              [--out DIR]   (DIR/<town>.json: every measurement)
//                              [--shotdir DIR]   (the pictures of --shots go here instead of --out)
//                              [--shots 10,24,38 | all]   (at these seconds: DIR/<town>_t010_recon.png, the reconstruction
//                              from the chase camera, _town.png, the town itself from there, _map.png, what every pixel of
//                              the measurement is (green: drawn right, orange: through, red: phantom, blue: seen by a camera
//                              but not drawn, grey: not seen), and _page.png, the page)
//                              [--ground class|height|both] [--ratio 1.15] [--scaled 0]   (the view's triangle rule,
//                              src/recon.js; every measurement also has the numbers of the rules in its VARIANTS)
//                              [--realtime SEC]   (first let the page drive in real time for SEC s, as a visitor sees it, and
//                              print frames per second, model calls per second, the model frame's and the mesh's times;
//                              with --recon off: the same page without ReconView, to compare)
//                              [--drag 1]   (drag in the view and double-click it: the camera turns and comes back)
//                              [--fonts block]   (a diagnosis: the page's web fonts are not loaded, so every text texture of
//                              the town is drawn with the system's font; see docs/recon_view.md on runs that differ)
//                              [--traffic 0] [--cars N] [--bikes N] [--parked N] [--model models/x.onnx] [--tdir .towns]
//                              [--ort node_modules/onnxruntime-web/dist/] [--ep webgpu|wasm] [--gl hw|soft]
// The run starts from the town's time 0 with the car at its start, whatever time the page took to load (the page's own
// frame loop drives on in real time until a test takes over; after --realtime or --drag the page is loaded again
// first, a longer real-time drive leaves more behind than the restart puts back). The last line is a digest of the
// car's poses and the model's input images at every measurement: the same arguments give one of two digests (KeiView
// draws a few frames in one of two ways from run to run; a page loaded a second time, as after --realtime, gives a
// third: docs/recon_view.md 4.4), and runs of the same digest are the same to the bit. "drive" is the digest of the
// car's whole state, the model's plan and its input images at every measurement (gate G3).
// Needs models/keipilot_mc.onnx and network access to cdn.jsdelivr.net (onnxruntime-web, three.js).
import crypto from 'node:crypto';
import fs from 'node:fs';
import puppeteer from 'puppeteer-core';
import { createServer } from './serve.mjs';

const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
const town = args.town || '1000', sec = Number(args.sec || 60), every = Number(args.every || 2);
const shots = args.shots === 'all' ? null : new Set((args.shots || '').split(',').filter(Boolean).map(Number));
const exe = process.env.CHROME || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  'C:/Program Files/Google/Chrome/Application/chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find((p) => fs.existsSync(p));
if (!exe) { console.error('No Chrome found - set CHROME=/path/to/browser'); process.exit(1); }
const glFlags = args.gl === 'soft' ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
  : ['--enable-gpu', '--enable-unsafe-webgpu',
    ...(process.platform === 'linux' ? ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'] : [])];

const [VW, VH] = (args.size || '1280x720').split('x').map(Number);     // --size 1920x1080: the window
const server = createServer();
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const browser = await puppeteer.launch({
  executablePath: exe, headless: true,
  args: [...glFlags, '--ignore-gpu-blocklist', '--no-sandbox', '--no-first-run', '--disable-extensions',
    ...(process.env.HTTPS_PROXY ? [`--proxy-server=${process.env.HTTPS_PROXY}`] : []), `--window-size=${VW},${VH}`],
  defaultViewport: { width: VW, height: VH, deviceScaleFactor: 1 },
  protocolTimeout: 900000,
});
let failed = false;
try {
  const page = await browser.newPage();
  const off = args.recon === 'off', loaded = [];
  if (args.fonts === 'block') await page.setRequestInterception(true);
  page.on('request', (r) => {
    if (/\/(recon|depth|unproject|fill)\.js|OrbitControls/.test(r.url())) loaded.push(r.url().split('/').pop());
    if (args.fonts === 'block') { if (/fonts\.(googleapis|gstatic)\.com/.test(r.url())) r.abort(); else r.continue(); }
  });
  page.on('console', (m) => { if (m.type() === 'error') console.log(`[error] ${m.text().slice(0, 300)}`); });
  page.on('pageerror', (e) => { console.log(`[pageerror] ${e.message}`); failed = true; });
  const q = new URLSearchParams({ pilot: 'mc', town, ...(off ? {} : { recon: args.recon || 'oracle' }), ...(args.ep ? { ep: args.ep } : {}), ...(args.tdir ? { tdir: args.tdir } : {}),
    ...(args.traffic ? { traffic: args.traffic } : {}), ...(args.cars ? { cars: args.cars } : {}),
    ...(args.bikes ? { bikes: args.bikes } : {}), ...(args.parked ? { parked: args.parked } : {}),
    ...(args.model ? { model: args.model } : {}), ...(args.ort ? { ort: args.ort } : {}),
    ...(args.layout && !off ? { layout: args.layout } : {}), ...(args.accum && !off ? { accum: args.accum } : {}),
    ...(args.keep && !off ? { accumkeep: args.keep } : {}), ...(args.far && !off ? { accumfar: args.far } : {}),
    ...(args.fill && !off ? { fill: args.fill } : {}), ...(args.fillshow && !off ? { fillshow: args.fillshow } : {}),
    ...(args.filltint && !off ? { filltint: args.filltint } : {}), ...(args.fillmodel && !off ? { fillmodel: args.fillmodel } : {}),
    ...(args.fillworker && !off ? { fillworker: args.fillworker } : {}),
    ...(args.ground ? { reconground: args.ground } : {}), ...(args.ratio ? { reconratio: args.ratio } : {}),
    ...(args.scaled ? { reconscaled: args.scaled } : {}) });
  const t0 = Date.now();
  const open = async () => {
    await page.goto(`http://127.0.0.1:${server.address().port}/index.html?${q}`, { waitUntil: 'load', timeout: 180000 });
    await page.waitForFunction('window.__pilotRun || /読み込めません/.test(document.getElementById("loadlabel").textContent)', { timeout: 600000, polling: 200 });
    if (!(await page.evaluate(() => !!window.__pilotRun))) throw new Error(await page.evaluate(() => document.getElementById('loadlabel').textContent));
  };
  await open();
  if (!off && !(await page.evaluate(() => !!window.__pilot.recon))) throw new Error('the page has no ReconView (?recon)');
  if (!off && !!args.fill !== (await page.evaluate(() => !!window.__pilot.recon.fill)) || (!args.fill && loaded.includes('fill.js'))) {
    throw new Error(args.fill ? 'the page has no fill (?fill)' : 'the page has a fill without ?fill');
  }
  if (off) {
    const has = await page.evaluate(() => !!window.__pilot.recon || !!document.getElementById('recon-view'));
    console.log(`without ?recon: ${has || loaded.length ? `FAIL: ReconView is there (${loaded.join(', ')})` : 'no ReconView, none of its modules requested'}`);
    if (has || loaded.length) failed = true;
  }
  if (args.realtime) {
    const fps = [], cap = [], inf = [], build = [], fprep = [], finf = [];
    let fillEnd = null;
    const now = () => page.evaluate(() => { const st = window.__pilot.state(), rc = window.__pilot.recon; return { n: st.nInfer, t: st.t, w: Date.now(), fills: rc && rc.fill ? rc.fill.state().made : 0 }; });
    const a = await now();
    for (let k = 0; k < Number(args.realtime) * 2; k++) {
      await new Promise((r) => setTimeout(r, 500));
      const r = await page.evaluate(() => { const st = window.__pilot.state(), rc = window.__pilot.recon; return { fps: window.__stats.fps, cap: st.captureMs, inf: st.inferMs, build: rc ? rc.state().buildMs : 0, fill: rc && rc.fill ? rc.fill.state() : null }; });
      if (k >= 4) { fps.push(r.fps); cap.push(r.cap); inf.push(r.inf); build.push(r.build); if (r.fill && r.fill.inferMs !== undefined) { fprep.push(r.fill.prepMs); finf.push(r.fill.inferMs); } }
      if (k === Number(args.realtime) * 2 - 1 && r.fill) fillEnd = r.fill;
    }
    const b = await now(), wall = (b.w - a.w) / 1000, mean = (x) => x.reduce((p, q) => p + q, 0) / x.length;
    console.log(`real time ${wall.toFixed(0)} s${off ? ' without ReconView' : ''}: ${mean(fps).toFixed(1)} fps (min ${Math.min(...fps).toFixed(1)}), ${((b.n - a.n) / wall).toFixed(1)} model calls/s, ` +
      `inference ${mean(inf).toFixed(1)} ms, model frame ${mean(cap).toFixed(1)} ms${off ? '' : `${args.recon === 'model' ? '' : ' (with its depth)'}, mesh ${mean(build).toFixed(1)} ms`}, town time x${((b.t - a.t) / wall).toFixed(2)}` +
      (fillEnd ? `; fill ${((b.fills - a.fills) / wall).toFixed(1)} fills/s${finf.length ? `, its input ${mean(fprep).toFixed(1)} ms, its model ${mean(finf).toFixed(1)} ms (${fillEnd.where}), turns skipped ${fillEnd.skipped}` : ''}` : ''));
  }
  if (args.drag && !off) {
    const cam = () => page.evaluate(() => window.__pilot.recon.viewCamera());
    const box = await (await page.$('#recon-view')).boundingBox(), c0 = await cam();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2 - 30, { steps: 8 });
    await page.mouse.up();
    await new Promise((r) => setTimeout(r, 300));
    const c1 = await cam();
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2, { clickCount: 2 });
    await new Promise((r) => setTimeout(r, 300));
    const c2 = await cam(), d = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    const ok = d(c0, c1) > 1 && d(c0, c2) < 1e-6;
    console.log(`drag: the camera moved ${d(c0, c1).toFixed(1)} m; double-click: ${d(c0, c2) < 1e-6 ? 'back at the chase camera' : `${d(c0, c2).toFixed(2)} m from it`} ${ok ? 'ok' : 'FAIL'}`);
    if (!ok) failed = true;
  }
  if (args.realtime || args.drag) await open();             // a fresh page for the measured run
  // from the town's time 0 and the car's start: stop the page's own loop, let the model call in flight end, take one
  // model frame (the following ones come every 0.1 s from this one on), then put time and car back
  const backend = await page.evaluate(async () => {
    await window.__pilotRun(0);
    await window.__pilot.idle();
    window.__pilot.capture(0, 1, true);
    await window.__pilot.idle();
    window.__sim(0);
    window.__pilot.reset();
    return window.__pilot.backend;
  });
  console.log(`town ${town}: ready in ${((Date.now() - t0) / 1000).toFixed(1)} s, backend ${backend}, recon ${args.recon || 'oracle'}`);
  console.log('   t     s  km/h   coverage    seen  (seen share)   error   wrong = through + phantom    0-10 m  10-30 m  30-80 m (coverage)   triangles wide / narrow');
  const pc = (a, b) => (b ? `${(100 * a / b).toFixed(1)} %`.padStart(7) : '      -');
  const rows = [], digest = crypto.createHash('sha1'), drive = crypto.createHash('sha1');
  const measure = !off && args.measure !== '0';             // --measure 0: the page as a visitor has it, digests only
  if (args.out) fs.mkdirSync(args.out, { recursive: true });
  for (let t = every; t <= sec + 1e-6; t += every) {
    const r = await page.evaluate(async (d) => {
      await window.__pilotRun(d.every - 0.1);
      if (window.__pilot.recon && d.measure) window.__pilot.recon.arm();  // measure at the next model frame
      const s = await window.__pilotRun(0.1);
      let h = 2166136261;                                   // FNV-1a of the model's input images
      for (const im of window.__pilot.images()) for (let i = 0; i < im.data.length; i++) h = Math.imul(h ^ im.data[i], 16777619);
      const rc = window.__pilot.recon;
      return { s: s.s, car: s.car, resets: s.resets, images: h >>> 0, m: rc ? rc.measured() : null, ms: rc ? rc.state().buildMs : 0,
        plan: s.plan ? { path: s.plan.path, target: s.plan.target, raw: s.plan.raw, tl: s.plan.tl } : null };
    }, { every, measure });
    digest.update(JSON.stringify([r.car.x, r.car.y, r.car.yaw, r.car.v, r.images]));
    drive.update(JSON.stringify([r.car, r.images, r.plan]));
    const m = r.m;
    rows.push({ t, s: r.s, car: r.car, images: r.images, resets: r.resets, ...(m || {}) });
    if (!m) { console.log(`${t.toFixed(0).padStart(4)} ${r.s.toFixed(0).padStart(5)} ${(r.car.v * 3.6).toFixed(0).padStart(5)}   ${!measure ? '' : '(no measurement: the drive restarted)'}`); continue; }
    console.log(`${t.toFixed(0).padStart(4)} ${r.s.toFixed(0).padStart(5)} ${(r.car.v * 3.6).toFixed(0).padStart(5)}   ${pc(m.covered, m.omega)} ${pc(m.seenCovered, m.seen)}  (${pc(m.seen, m.omega)})  ` +
      `${m.error === null ? '    -' : `${(100 * m.error).toFixed(2)} %`.padStart(6)} ${pc(m.wrong, m.drawn)} ${pc(m.through, m.drawn)} ${pc(m.phantom, m.drawn)}       ${m.bands.map((b) => pc(b.covered, b.omega)).join('  ')}` +
      `              ${m.stats.map((st) => st.triangles).join(' / ')}`);
    if ((args.out || args.shotdir) && (shots === null ? !!args.shots : shots.has(t))) {
      fs.mkdirSync(args.shotdir || args.out, { recursive: true });
      const tag = `${args.shotdir || args.out}/${town}_t${String(Math.round(t)).padStart(3, '0')}`;
      await page.screenshot({ path: `${tag}_page.png` });     // the page as the last step drew it, before snapshot() draws its own
      const png = await page.evaluate(() => window.__pilot.recon.snapshot(window.__pilot.state().car));
      for (const k of ['recon', 'town', 'map', 'oracle', 'oracleMap', 'current', 'currentMap']) if (png[k]) fs.writeFileSync(`${tag}_${k}.png`, Buffer.from(png[k].split(',')[1], 'base64'));
      for (const [k, v] of Object.entries(png.accum || {})) fs.writeFileSync(`${tag}_acc_${k}.png`, Buffer.from(v.split(',')[1], 'base64'));
      for (const [k, v] of Object.entries(png.fill || {})) if (v) fs.writeFileSync(`${tag}_fill_${k}.png`, Buffer.from(v.split(',')[1], 'base64'));
    }
  }
  const total = (name, get) => {
    const ms = rows.map(get).filter((m) => m && m.omega), sum = (k) => ms.reduce((a, m) => a + (m[k] || 0), 0);
    const errs = ms.filter((m) => m.error != null).map((m) => m.error).sort((a, b) => a - b);
    console.log(`${name} ${ms.length} frames: coverage ${pc(sum('covered'), sum('omega'))}, seen ${pc(sum('seenCovered'), sum('seen'))} (seen share ${pc(sum('seen'), sum('omega'))}), ` +
      `error (median of the frames) ${errs.length ? (100 * errs[errs.length >> 1]).toFixed(2) : '-'} %, wrong ${pc(sum('wrong'), sum('drawn'))} = through ${pc(sum('through'), sum('drawn'))} + phantom ${pc(sum('phantom'), sum('drawn'))}; ` +
      `drawn right: ${pc(sum('good'), sum('omega'))} of all, ${pc(sum('seenGood'), sum('seen'))} of the seen`);
  };
  total('all', (r) => r);
  if (rows.some((r) => r.oracle)) total('the same frames with the town\'s depth (oracle):', (r) => r.oracle);
  if (rows.some((r) => r.accum)) {
    total('the same frames with the keyframes (accumulation):', (r) => r.accum);
    const a = (k) => rows.reduce((x, r) => x + (r.accum ? r.accum[k] : 0), 0), fin = [...rows].reverse().find((r) => r.accum).accum;
    console.log(`  the keyframes add ${a('added')} pixels where the current frame draws nothing: right ${pc(a('addedGood'), a('added'))}, through ${pc(a('addedThrough'), a('added'))}, phantom ${pc(a('addedPhantom'), a('added'))}; ` +
      `clearly in front of the current frame's surface at ${a('before')} pixels (${pc(a('before'), rows.reduce((x, r) => x + (r.drawn || 0), 0))} of its pixels; right there ${pc(a('beforeGood'), a('before'))})`);
    for (const name of Object.keys(fin.variants || {})) total(`  keyframes drawn as ${name.padEnd(8)}`, (r) => (r.accum ? r.accum.variants[name] : null));
    console.log(`  keyframes made ${fin.made}, kept vertices ${fin.vertices}, of them inside a car's or rider's box ${fin.ghostVertices} (${pc(fin.ghostVertices, fin.vertices)}), keyframes with ${10} or more of those: ${fin.ghostKeyframes}`);
  }
  if (rows.some((r) => r.fill)) {
    const NAMES = { holes: 'the holes filled, this frame\'s fill', holesPrev: 'the holes filled, the fill before', all: 'the fill alone, this frame\'s', allPrev: 'the fill alone, the one before' };
    const part = (name, get) => {
      if (!rows.some((r) => r.fill && get(r.fill))) return;
      total(`  ${name}:`, (r) => (r.fill ? get(r.fill) : null));
      const a = (key) => rows.reduce((x, r) => x + (r.fill && get(r.fill) ? get(r.fill)[key] : 0), 0);
      console.log(`    it draws ${a('added')} pixels where the view draws nothing: right ${pc(a('addedGood'), a('added'))}, through ${pc(a('addedThrough'), a('added'))}, phantom ${pc(a('addedPhantom'), a('added'))}`);
    };
    for (const [k, name] of Object.entries(NAMES)) part(name, (f) => f[k]);
    part('the holes filled, the town\'s own fill of this frame', (f) => f.oracle && f.oracle.holes);
    part('the town\'s own fill alone', (f) => f.oracle && f.oracle.all);
    const fl = rows.filter((r) => r.fill && r.fill.flicker), fn = fl.reduce((x, r) => x + r.fill.flicker.pixels, 0);
    if (fn) console.log(`  flicker: this fill against the one before, from the same camera, at the holes both fill with the same surface: colours ${(fl.reduce((x, r) => x + r.fill.flicker.sum, 0) / fn).toFixed(2)} apart (0..255; ${fn} pixels of ${fl.length} frames)`);
    const stale = rows.filter((r) => r.fill && r.fill.fresh === false).length;
    if (stale) { console.log(`  FAIL: ${stale} measurements without this frame's fill`); failed = true; }
    const fs2 = rows.filter((r) => r.fill), mean = (get) => fs2.reduce((x, r) => x + get(r.fill), 0) / fs2.length, aged = fs2.filter((r) => r.fill.age);
    console.log(`  the fill: ${mean((f) => f.stats.triangles).toFixed(0)} triangles of ${mean((f) => f.stats.valid).toFixed(0)} valid vertices, cut at edges ${mean((f) => f.stats.edge).toFixed(0)}, cells over a hole ${mean((f) => f.stats.holeCells).toFixed(0)}, ` +
      `${mean((f) => f.ms).toFixed(1)} ms; the fill before: ${aged.length ? `${(aged.reduce((x, r) => x + r.fill.age.frames, 0) / aged.length).toFixed(1)} model frames, ${(aged.reduce((x, r) => x + r.fill.age.metres, 0) / aged.length).toFixed(2)} m earlier` : '-'}`);
  }
  const vs = rows.find((r) => r.variants);
  for (const name of vs ? Object.keys(vs.variants) : []) {
    const v = (k) => rows.reduce((a, r) => a + (r.variants ? r.variants[name][k] : 0), 0);
    console.log(`  rule ${name.padEnd(10)} coverage ${pc(v('covered'), v('omega'))}, seen ${pc(v('seenCovered'), v('seen'))}, wrong ${pc(v('wrong'), v('drawn'))} = through ${pc(v('through'), v('drawn'))} + phantom ${pc(v('phantom'), v('drawn'))}; drawn right ${pc(v('good'), v('omega'))} / ${pc(v('seenGood'), v('seen'))}`);
  }
  const hex = digest.digest('hex').slice(0, 16), driveHex = drive.digest('hex').slice(0, 16);
  console.log(`digest ${hex}  drive ${driveHex}`);
  if (args.out) fs.writeFileSync(`${args.out}/${args.name || town}.json`, JSON.stringify({ town, sec, every, recon: args.recon || 'oracle', backend, digest: hex, drive: driveHex,
    args: Object.fromEntries(Object.entries(args).filter(([k]) => !['out', 'shots', 'shotdir'].includes(k))), rows }, null, 1));
} catch (e) {
  console.error(e); failed = true;
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
