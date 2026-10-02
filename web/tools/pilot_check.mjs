// Headless check of the in-browser KeiPilot demo (index.html?pilot=1): opens the page in Chrome, drives `--sec`
// seconds of simulated time through window.__pilotRun (every model call awaited, as in KeiSim) and prints the
// inference backend and time plus one line per `--every` seconds: route position, speed, target speed, light,
// next turn, lateral deviation from the route and restarts.
//   node tools/pilot_check.mjs [--town 1000] [--sec 60] [--every 2] [--ep webgpu|wasm] [--gl hw|soft] [--shot out.png]
//                              [--url https://southern-star.github.io/keisim/]   (a deployed copy instead of web/)
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
  const q = new URLSearchParams({ pilot: '1', town, ...(args.ep ? { ep: args.ep } : {}) });
  const t0 = Date.now();
  await page.goto(`${base}?${q}`, { waitUntil: 'load', timeout: 180000 });
  await page.waitForFunction('window.__pilotRun || /読み込めません/.test(document.getElementById("loadlabel").textContent)', { timeout: 600000, polling: 200 });
  const label = await page.evaluate(() => document.getElementById('loadlabel').textContent);
  if (!(await page.evaluate(() => !!window.__pilotRun))) throw new Error(label);
  const info = await page.evaluate(() => ({ backend: window.__pilot.backend, adapter: window.__pilot.adapter,
    gl: (() => { try { const g = document.getElementById('scene').getContext('webgl2'); const d = g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : '?'; } catch (e) { return 'n/a'; } })() }));
  console.log(`town ${town}: ready in ${((Date.now() - t0) / 1000).toFixed(1)} s, backend ${info.backend}${info.adapter ? ` (${info.adapter})` : ''}, WebGL ${info.gl}`);
  console.log('   t     s  km/h  target  light  next turn       dev  restarts  infer');
  const names = ['red', 'yellow', 'green', 'none'];
  for (let t = every; t <= sec + 1e-6; t += every) {
    const s = await page.evaluate((d) => window.__pilotRun(d).then((r) => ({
      s: r.s, v: r.car.v, target: r.plan ? r.plan.target : null, tl: r.plan ? r.plan.tl : null,
      turn: r.inputs.turn || 'straight', dist: r.inputs.dist, dev: r.deviation, resets: r.resets, ms: r.inferMs })), every);
    const k = s.tl ? s.tl.indexOf(Math.max(...s.tl)) : 3;
    console.log(`${t.toFixed(0).padStart(4)} ${s.s.toFixed(0).padStart(5)} ${(s.v * 3.6).toFixed(0).padStart(5)} ${s.target === null ? '     -' : (s.target * 3.6).toFixed(0).padStart(6)}` +
      `  ${names[k].padEnd(6)} ${s.turn.padEnd(8)} ${s.dist === null ? '' : `${s.dist.toFixed(0)} m`.padStart(6)} ${s.dev.toFixed(2).padStart(6)} ${String(s.resets).padStart(6)} ${s.ms.toFixed(0).padStart(6)} ms`);
  }
  if (args.shot) { await page.screenshot({ path: args.shot }); console.log(`screenshot: ${args.shot}`); }
} finally {
  await browser.close();
  if (server) server.close();
}
