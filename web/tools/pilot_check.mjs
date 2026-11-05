// Headless check of the in-browser KeiPilot demo (index.html?pilot=1): opens the page in Chrome, drives `--sec`
// seconds of simulated time through window.__pilotRun (every model call awaited, as in KeiSim) and prints the
// inference backend and time plus one line per `--every` seconds: route position, speed, target speed, light,
// next turn, lateral deviation from the route, restarts, and the other cars (count / contacts with the ego / contacts
// among themselves).
//   node tools/pilot_check.mjs [--town 1000] [--sec 60] [--every 2] [--ep webgpu|wasm] [--gl hw|soft] [--shot out.png]
//                              [--traffic 0] [--cars 20]   (no other cars / their number)
//                              [--model models/keipilot_v24.onnx]   (another ONNX under web/; default models/keipilot.onnx)
//                              [--tdir .towns]   (a town exported there by scripts/import_osm.py, not one under towns/)
//                              [--url https://southern-star.github.io/keisim/]   (a deployed copy instead of web/)
//                              [--ort node_modules/onnxruntime-web/dist/]   (onnxruntime-web from web/, e.g. without CDN
//                              access: npm install --no-save onnxruntime-web@1.20.1)
//                              [--ramp N]   (towns with ramps: start before ramp N; -1: the usual start)
//                              [--frames DIR]   (a screenshot at every report: DIR/f0001.png, ...)
//                              [--pilot mc]   (the multi-camera demo, models/keipilot_mc.onnx: ?pilot=mc)
//                              [--images DIR]   (the model's input images at every report: DIR/i0001_c0.png, ...)
//                              [--realtime SEC]   (first let the page drive in real time for SEC s and print its speed:
//                              frames per second, model calls per second, inference and model-frame times)
//                              [--only ground,roads,signals,actors]   (build only these modules: fast without a GPU)
//                              [--infer main]   (the model on the page's main thread, as before src/infer_worker.js)
//                              [--readback sync]   (read the model frame back at once, waiting for the GPU, as before)
//                              [--hz N]   (at most N model calls a second in real time)
// Needs models/keipilot.onnx (scripts/export_onnx.py) and network access to cdn.jsdelivr.net (onnxruntime-web).
import fs from 'node:fs';
import puppeteer from 'puppeteer-core';
import { createServer } from './serve.mjs';

const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
const town = args.town || '1000', sec = Number(args.sec || 60), every = Number(args.every || 2);
const exe = process.env.CHROME || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  'C:/Program Files/Google/Chrome/Application/chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find((p) => fs.existsSync(p));
if (!exe) { console.error('No Chrome found - set CHROME=/path/to/browser'); process.exit(1); }
const glFlags = args.gl === 'soft' ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
  : ['--enable-gpu', '--enable-unsafe-webgpu',       // Linux: Vulkan gives headless Chrome the GPU's WebGPU adapter
    ...(process.platform === 'linux' ? ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'] : [])];

const server = args.url ? null : createServer();
if (server) await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = args.url || `http://127.0.0.1:${server.address().port}/index.html`;
const browser = await puppeteer.launch({
  executablePath: exe, headless: true,
  args: [...glFlags, '--ignore-gpu-blocklist', '--no-sandbox', '--no-first-run', '--disable-extensions',
    ...(process.env.HTTPS_PROXY ? [`--proxy-server=${process.env.HTTPS_PROXY}`] : []), '--window-size=1280,720'],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 },
  protocolTimeout: 900000,
});
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warn') console.log(`[${m.type()}] ${m.text().slice(0, 300)}`); });
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
  const q = new URLSearchParams({ pilot: args.pilot || '1', town, ...(args.ep ? { ep: args.ep } : {}), ...(args.tdir ? { tdir: args.tdir } : {}),
    ...(args.traffic ? { traffic: args.traffic } : {}), ...(args.cars ? { cars: args.cars } : {}),
    ...(args.model ? { model: args.model } : {}), ...(args.ort ? { ort: args.ort } : {}), ...(args.ramp ? { ramp: args.ramp } : {}),
    ...(args.only ? { only: args.only } : {}), ...(args.infer ? { infer: args.infer } : {}),
    ...(args.readback ? { readback: args.readback } : {}), ...(args.hz ? { hz: args.hz } : {}) });
  const t0 = Date.now();
  await page.goto(`${base}?${q}`, { waitUntil: 'load', timeout: 180000 });
  await page.waitForFunction('window.__pilotRun || /読み込めません/.test(document.getElementById("loadlabel").textContent)', { timeout: 600000, polling: 200 });
  const label = await page.evaluate(() => document.getElementById('loadlabel').textContent);
  if (!(await page.evaluate(() => !!window.__pilotRun))) throw new Error(label);
  const info = await page.evaluate(() => ({ backend: window.__pilot.backend, adapter: window.__pilot.adapter, inferIn: window.__pilot.inferIn,
    stopLines: window.__pilot.lanes.filter((L) => L.stopSign != null).length,
    rig: window.__pilot.rig,
    gl: (() => { try { const g = document.getElementById('scene').getContext('webgl2'); const d = g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : '?'; } catch (e) { return 'n/a'; } })() }));
  console.log(`town ${town}: ready in ${((Date.now() - t0) / 1000).toFixed(1)} s, backend ${info.backend}${info.adapter ? ` (${info.adapter})` : ''}${info.inferIn === 'main' ? ' on the main thread' : info.inferIn ? ' in a worker' : ''}, WebGL ${info.gl}${info.rig ? `, cameras ${info.rig.join(' + ')}` : ''}`);
  if (info.stopLines) console.log(`止まれ lines (other cars stop once): ${info.stopLines}`);
  if (args.realtime) {          // the page's own frame loop, as a visitor sees it
    const fps = [], infer = [], cap = [], read = [];
    const n0 = await page.evaluate(() => window.__pilot.state().nInfer), w0 = Date.now();
    for (let k = 0; k < Number(args.realtime) * 2; k++) {
      await new Promise((r) => setTimeout(r, 500));
      const r = await page.evaluate(() => { const s = window.__pilot.state(); return { fps: window.__stats.fps, infer: s.inferMs, cap: s.captureMs, read: s.readMs }; });
      if (k >= 4) { fps.push(r.fps); infer.push(r.infer); cap.push(r.cap); read.push(r.read || 0); }   // after 2 s of warm-up
    }
    const calls = (await page.evaluate(() => window.__pilot.state().nInfer)) - n0, wall = (Date.now() - w0) / 1000;
    const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1], mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    console.log(`real time ${wall.toFixed(0)} s: ${mean(fps).toFixed(1)} fps (min ${Math.min(...fps).toFixed(1)}), ${(calls / wall).toFixed(1)} model calls/s, ` +
      `inference ${mean(infer).toFixed(1)} ms (median ${med(infer).toFixed(1)}), model frame ${mean(cap).toFixed(1)} ms (median ${med(cap).toFixed(1)})` +
      (mean(read) > 0 ? `, read back after ${mean(read).toFixed(1)} ms` : ''));
  }
  console.log('   t     s  km/h  target  light  next turn       dev  restarts  infer  frame   cars/ego/npc contacts');
  const names = ['red', 'yellow', 'green', 'none'];
  for (let t = every; t <= sec + 1e-6; t += every) {
    const s = await page.evaluate((d) => window.__pilotRun(d).then((r) => ({
      s: r.s, v: r.car.v, target: r.plan ? r.plan.target : null, tl: r.plan ? r.plan.tl : null,
      turn: (r.inputs.turn || 'straight') + (r.inputs.ramp ? '/ramp' : ''), dist: r.inputs.dist, dev: r.deviation, resets: r.resets, ms: r.inferMs, cap: r.captureMs, hold: r.holding,
      traffic: r.traffic, stand: r.useStand ? r.standM : null })), every);
    if (args.images) {
      fs.mkdirSync(args.images, { recursive: true });
      const urls = await page.evaluate(() => window.__pilot.images().map((im) => {
        const c = document.createElement('canvas'); c.width = im.width; c.height = im.height;
        c.getContext('2d').putImageData(im, 0, 0); return c.toDataURL('image/png'); }));
      urls.forEach((u, k) => fs.writeFileSync(`${args.images}/i${String(Math.round(t / every)).padStart(4, '0')}_c${k}.png`, Buffer.from(u.split(',')[1], 'base64')));
      const st = await page.evaluate(() => { const r = window.__pilot.state(); return { car: r.captured, t: r.captured && r.captured.t }; });   // the pose and time of those images
      fs.writeFileSync(`${args.images}/i${String(Math.round(t / every)).padStart(4, '0')}.json`, JSON.stringify(st));
    }
    if (args.frames) { fs.mkdirSync(args.frames, { recursive: true }); await page.screenshot({ path: `${args.frames}/f${String(Math.round(t / every)).padStart(4, '0')}.png` }); }
    const k = s.tl ? s.tl.indexOf(Math.max(...s.tl)) : 3;
    console.log(`${t.toFixed(0).padStart(4)} ${s.s.toFixed(0).padStart(5)} ${(s.v * 3.6).toFixed(0).padStart(5)} ${s.target === null ? '     -' : (s.target * 3.6).toFixed(0).padStart(6)}` +
      `  ${names[k].padEnd(6)} ${s.turn.padEnd(8)} ${s.dist === null ? '' : `${s.dist.toFixed(0)} m`.padStart(6)} ${s.dev.toFixed(2).padStart(6)} ${String(s.resets).padStart(6)} ${s.ms.toFixed(0).padStart(6)} ms ${s.cap.toFixed(0).padStart(4)} ms${s.hold ? ' hold' : ''}   ${s.traffic ? `${s.traffic.n}/${s.traffic.contacts}/${s.traffic.npcContacts}${s.traffic.stopSignStops ? ` 止まれ ${s.traffic.stopSignStops}` : ''}${s.traffic.laneChanges !== undefined ? `  lane changes ${s.traffic.laneChanges} ramps ${s.traffic.rampPassages}` : ''}` : '-'}${s.stand === null ? '' : `   stood ${s.stand.toFixed(1)} m ago`}`);
  }
  if (args.shot) { await page.screenshot({ path: args.shot }); console.log(`screenshot: ${args.shot}`); }
} finally {
  await browser.close();
  if (server) server.close();
}
