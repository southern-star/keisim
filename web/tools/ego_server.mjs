// KeiView ego-camera render server for KeiSim: headless Chrome (GPU) driven over stdin/stdout.
//   node tools/ego_server.mjs [--w 320 --h 160] [--q medium] [--gl hw|soft] [--tdir .towns]
// One JSON request per line on stdin, one JSON reply per line on stdout (logs go to stderr):
//   {"id":1,"op":"town","town":1000}                        -> {"id":1,"ok":true,"ms":8800,"errors":[]}
//   {"id":2,"op":"render","cam":{"pos":[x,y,z],"fwd":[fx,fy,fz],"vfov":61.6},"t":12.3,
//    "veh":[[id,x,y,yaw,L,W,H,r,g,b,kind,brake],...],"ped":[[id,x,y,yaw,h,...9 rgb...,phase],...],
//    "labels":true,"reset":false}                           -> {"id":2,"ok":true,"rgb":"<b64>","sem":"<b64>","w":320,"h":160,"ms":6.1}
//   {"id":3,"op":"quit"}
// Images are raw RGB / uint8 class ids in OpenGL row order (bottom-up), base64 encoded.
import fs from 'node:fs';
import readline from 'node:readline';
import puppeteer from 'puppeteer-core';
import { createServer } from './serve.mjs';

const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
const W = Number(args.w || 320), H = Number(args.h || 160);
const Q = args.q || 'medium';
const TDIR = args.tdir || '.towns';
const log = (...a) => process.stderr.write(`[ego_server] ${a.join(' ')}\n`);
const reply = (o) => process.stdout.write(JSON.stringify(o) + '\n');

function findBrowser() {
  if (process.env.CHROME) return process.env.CHROME;
  const c = [];
  try { for (const d of fs.readdirSync('/opt/pw-browsers')) if (d.startsWith('chromium-')) c.push(`/opt/pw-browsers/${d}/chrome-linux/chrome`); } catch (e) { /* none */ }
  c.push('/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
    'C:/Program Files/Google/Chrome/Application/chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');
  return c.find(p => fs.existsSync(p));
}
const exe = findBrowser();
if (!exe) { reply({ id: 0, ok: false, error: 'No Chromium/Chrome found - set CHROME=/path/to/browser' }); process.exit(1); }
const gl = args.gl || 'hw';
const glFlags = gl === 'soft' ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
  : process.platform === 'win32' ? ['--use-angle=d3d11', '--enable-gpu'] : ['--enable-gpu'];

const server = createServer();
await new Promise(r => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const browser = await puppeteer.launch({
  executablePath: exe, headless: true,
  args: [...glFlags, '--ignore-gpu-blocklist', '--enable-webgl', '--no-sandbox', '--no-first-run', '--disable-extensions',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
    ...(process.env.HTTPS_PROXY ? [`--proxy-server=${process.env.HTTPS_PROXY}`] : []), `--window-size=${W},${H}`],
  defaultViewport: { width: W, height: H, deviceScaleFactor: 1 },
  protocolTimeout: 900000,
});
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error') log(`[console.error] ${m.text()}`); });
page.on('pageerror', (e) => log(`[pageerror] ${e.message}`));

let current = null;
async function loadTown(town) {
  const t0 = Date.now();
  const q = new URLSearchParams({ shot: '1', ego: '1', town: String(town), w: String(W), h: String(H), q: Q, tdir: TDIR });
  await page.goto(`http://127.0.0.1:${port}/index.html?${q}`, { waitUntil: 'load', timeout: 180000 });
  await page.waitForFunction('window.__ready === true', { timeout: 900000, polling: 100 });
  const info = await page.evaluate(() => ({ errors: window.__errors || [], stats: window.__stats,
    gpu: (() => { try { const g = document.getElementById('scene').getContext('webgl2'); const d = g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'unknown'; } catch (e) { return 'n/a'; } })() }));
  current = town;
  return { ms: Date.now() - t0, errors: info.errors.map(e => `[${e.module}] ${String(e.message).split('\n')[0]}`), gpu: info.gpu,
    modules: info.stats && info.stats.modules };
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
reply({ id: 0, ok: true, ready: true, w: W, h: H });
for await (const line of rl) {
  if (!line.trim()) continue;
  let q;
  try { q = JSON.parse(line); } catch (e) { reply({ id: -1, ok: false, error: 'bad json' }); continue; }
  try {
    if (q.op === 'town') {
      const r = (q.town === current && !q.force) ? { ms: 0, errors: [], cached: true } : await loadTown(q.town);
      reply({ id: q.id, ok: true, ...r });
    } else if (q.op === 'render') {
      if (current === null) throw new Error('no town loaded');
      const r = await page.evaluate((req) => window.__ego(req), q);
      reply({ id: q.id, ok: true, ...r });
    } else if (q.op === 'quit') {
      reply({ id: q.id, ok: true });
      break;
    } else {
      reply({ id: q.id, ok: false, error: `unknown op ${q.op}` });
    }
  } catch (e) {
    reply({ id: q.id, ok: false, error: String(e && e.message || e) });
  }
}
await browser.close();
server.close();
