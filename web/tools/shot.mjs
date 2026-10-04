// Headless screenshots of the real renderer (puppeteer-core + a local Chromium/Edge/Chrome).
// Adapted from Sakuragaoka Station tools/shot.mjs (MIT). Starts its own server.
//   node tools/shot.mjs --town 1000 [--views 1,2,3] [--cams "x,z,yaw,pitch;x,y,z,yaw,pitch"] [--only ground,roads]
//                       [--out shots/t1000] [--t 20] [--w 1280 --h 720] [--q high] [--bench 3] [--gl auto|soft|hw]
//                       [--tdir .towns]   (towns exported by KeiSim on demand, e.g. town styles: --town 1017-<hash>)
//   --views  1-based indices of the town's preset views (default: all presets, unless --cams is given)
//   --cams   ';'-separated cameras. 4 numbers = walking eye (x,z,yawDeg,pitchDeg); 5 = free camera (x,y,z,yaw,pitch).
//            yaw 0 = north(-Z), 90 = west, 180 = south, -90 = east.
//   --gl     soft = SwiftShader (CPU; works without a GPU, e.g. cloud containers), hw = GPU. auto: soft on Linux.
// Browser: $CHROME, else Playwright's Chromium (/opt/pw-browsers), Edge / Chrome on Windows, Chrome on macOS.
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';
import { createServer, root } from './serve.mjs';

const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
const W = Number(args.w || 1280), H = Number(args.h || 720);
const town = args.town || '1000';
const out = args.out || `shots/t${town}`;
fs.mkdirSync(path.dirname(path.resolve(root, out)), { recursive: true });

function findBrowser() {
  if (process.env.CHROME) return process.env.CHROME;
  const c = [];
  try { for (const d of fs.readdirSync('/opt/pw-browsers')) if (d.startsWith('chromium-')) c.push(`/opt/pw-browsers/${d}/chrome-linux/chrome`); } catch (e) { /* none */ }
  c.push('/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');
  return c.find(p => fs.existsSync(p));
}
const exe = findBrowser();
if (!exe) { console.log('No Chromium/Edge/Chrome found — set CHROME=/path/to/browser'); process.exit(1); }
const gl = args.gl && args.gl !== 'auto' ? args.gl : (process.platform === 'linux' ? 'soft' : 'hw');
const glFlags = gl === 'soft' ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
  : process.platform === 'win32' ? ['--use-angle=d3d11', '--enable-gpu'] : ['--enable-gpu'];

const server = createServer();
await new Promise(r => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const browser = await puppeteer.launch({
  executablePath: exe, headless: true,
  args: [...glFlags, '--ignore-gpu-blocklist', '--enable-webgl', '--no-sandbox', '--no-first-run', '--disable-extensions',
    ...(process.env.HTTPS_PROXY ? [`--proxy-server=${process.env.HTTPS_PROXY}`] : []), `--window-size=${W},${H}`],
  defaultViewport: { width: W, height: H, deviceScaleFactor: 1 },
  protocolTimeout: 900000,
});
const logs = [];
const T0 = Date.now(); const el = () => `${((Date.now() - T0) / 1000).toFixed(1)}s`;
try {
  const page = await browser.newPage();
  page.on('console', (m) => { const t = m.type(); if (t === 'error' || t === 'warning' || t === 'warn') logs.push(`[console.${t}] ${m.text()}`); });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  page.on('requestfailed', (r) => logs.push(`[requestfailed] ${r.url()} ${r.failure()?.errorText}`));
  const q = new URLSearchParams({ shot: '1', town, w: String(W), h: String(H), t: String(args.t || 0), q: args.q || 'high',
    ...(args.tdir ? { tdir: args.tdir } : {}) });
  if (args.only) q.set('only', args.only);
  await page.goto(`http://127.0.0.1:${port}/index.html?${q}`, { waitUntil: 'load', timeout: 120000 });
  await page.waitForFunction('window.__ready === true', { timeout: 900000, polling: 250 });
  console.log(`ready at ${el()}`);
  const info = await page.evaluate(() => ({ errors: window.__errors, stats: window.__stats, views: window.__town ? window.__town.views : [],
    gl: (() => { try { const g = document.getElementById('scene').getContext('webgl2'); const d = g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'unknown'; } catch (e) { return 'n/a'; } })() }));
  const shots = [];
  if (args.cams) for (const c of args.cams.split(';').map(s => s.trim()).filter(Boolean)) shots.push({ cam: c.split(',').map(Number), label: c });
  const viewIdx = args.views ? args.views.split(',').map(v => Number(v) - 1) : (args.cams ? [] : info.views.map((_, i) => i));
  for (const i of viewIdx) shots.push({ view: i, label: `view ${i + 1} (${info.views[i]?.label})` });
  for (let i = 0; i < shots.length; i++) {
    const s = shots[i];
    if (s.cam) await page.evaluate((v) => { if (v.length === 4) window.__setCam(v[0], null, v[1], v[2], v[3]); else window.__setCam(v[0], v[1], v[2], v[3], v[4]); }, s.cam);
    else await page.evaluate((k) => window.__view(k), s.view);
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(r)))));
    const file = path.resolve(root, `${out}_${i}.png`);
    await page.screenshot({ path: file });
    const st = await page.evaluate(() => ({ calls: window.__stats.calls, triangles: window.__stats.triangles }));
    console.log(`[${el()}] saved ${path.relative(root, file)}  ${s.label}  calls=${st.calls} tris=${st.triangles}`);
    if (args.bench) console.log('  bench:', JSON.stringify(await page.evaluate((n) => window.__bench(n), Number(args.bench) || 10)));
  }
  console.log('gpu:', info.gl);
  console.log('module stats:', JSON.stringify(info.stats.modules));
  console.log('batch:', JSON.stringify(info.stats.batch && { merged: info.stats.batch.merged, sources: info.stats.batch.sources }));
  if (info.errors && info.errors.length) { console.log('MODULE ERRORS:'); for (const e of info.errors) console.log(` - [${e.module}] ${e.message.split('\n').slice(0, 8).join('\n   ')}`); process.exitCode = 1; }
} catch (e) {
  console.log('SHOT FAILED:', e.message);
  process.exitCode = 1;
} finally {
  const noisy = /ImageUtils\.getDataURL|toNonIndexed/;
  const shown = logs.filter(l => !noisy.test(l));
  if (shown.length) { console.log('PAGE LOGS:'); for (const l of shown.slice(0, 40)) console.log(' ', l.slice(0, 500)); }
  if (logs.some(l => l.includes('ERR_CERT_AUTHORITY_INVALID'))) console.log('hint: the browser does not trust this network\'s TLS proxy (web fonts fall back to system fonts); add its CA to ~/.pki/nssdb with certutil.');
  await browser.close();
  server.close();
}
