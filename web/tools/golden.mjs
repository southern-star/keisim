// Pixel-exact comparison of KeiView's default pictures between two trees: a base (git ref, default origin/main: the
// tree before the work) and this one. Used by the car and pedestrian redesigns (acceptance 1 of both): the plain
// actors of ego mode and the detailed cars of ?cardetail=1 must not change by a single pixel, RGB and labels.
//   node tools/golden.mjs [--base origin/main] [--town 1000] [--frames 6] [--gl hw|soft]
//                         [--modes "|cardetail=1"]   ('|'-separated extra query strings, '' = ego mode's default)
// The base tree is unpacked with git archive into the OS temp folder; both trees are served by this script (three.js
// from this tree's node_modules) and drawn by the same headless browser, one after the other.
// Each frame has cars (every size class), pedestrians and bicycles in front of the camera.
import fs from 'node:fs';
import os from 'node:os';
import http from 'node:http';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const here = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg' };

export function findBrowser() {
  if (process.env.CHROME) return process.env.CHROME;
  const c = [];
  try { for (const d of fs.readdirSync('/opt/pw-browsers')) if (d.startsWith('chromium-')) c.push(`/opt/pw-browsers/${d}/chrome-linux/chrome`); } catch (e) { /* none */ }
  c.push('/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
    'C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');
  return c.find((p) => fs.existsSync(p));
}

/** Static server of a web/ tree; three.js always from this tree's node_modules. */
export function serveTree(root) {
  const local = fs.existsSync(path.join(here, 'node_modules/three/build/three.module.js'));
  return http.createServer((req, res) => {
    let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (p.endsWith('/')) p += 'index.html';
    const file = p.startsWith('/node_modules/') ? path.join(here, p) : path.join(root, p);
    if (!file.startsWith(root) && !file.startsWith(here)) { res.writeHead(403); return res.end(); }
    fs.readFile(file, (err, data) => {
      if (err) { res.writeHead(404); return res.end('not found'); }
      if (local && p === '/index.html') data = Buffer.from(String(data).replaceAll('https://cdn.jsdelivr.net/npm/three@0.170.0/', '/node_modules/three/'));
      res.writeHead(200, { 'Content-Type': TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream', 'Cache-Control': 'no-store' });
      res.end(data);
    });
  });
}

/** web/ of a git ref, unpacked once per commit into the temp folder. */
export function baseTree(ref) {
  const repo = path.resolve(here, '..');
  const sha = execFileSync('git', ['rev-parse', ref], { cwd: repo }).toString().trim();
  const dir = path.join(os.tmpdir(), `keiview-base-${sha.slice(0, 12)}`);
  if (!fs.existsSync(path.join(dir, 'web/index.html'))) {
    fs.mkdirSync(dir, { recursive: true });
    const tarFile = path.join(dir, 'web.tar');
    execFileSync('git', ['archive', '--format=tar', '-o', tarFile, sha, 'web'], { cwd: repo });
    execFileSync('tar', ['-xf', 'web.tar'], { cwd: dir });
    fs.rmSync(tarFile);
  }
  return { dir: path.join(dir, 'web'), sha };
}

// ---- the frames: actors in front of the camera, sizes and colours as KeiSim draws them (keisim/traffic.py)
export function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const CAR_COLORS = [[236, 236, 232], [200, 204, 208], [40, 42, 48], [120, 124, 130], [176, 36, 40], [36, 70, 140], [228, 200, 60], [60, 120, 80],
  [232, 232, 232], [90, 60, 44], [150, 190, 214]];
/** One car's size in the classes and shares of keisim/traffic.py _sample_dims: [L, W, H, kind]. */
export function sampleDims(r) {
  const u = r(), f = (lo, hi) => lo + (hi - lo) * r();
  if (u < 0.62) return [f(4.0, 4.8), f(1.68, 1.85), f(1.4, 1.6), 0];
  if (u < 0.78) return [f(3.35, 3.45), 1.48, f(1.6, 1.8), 0];
  if (u < 0.92) return [f(4.6, 5.0), f(1.8, 1.95), f(1.75, 2.0), 0];
  return [f(6.0, 7.6), f(2.0, 2.25), f(2.6, 3.0), 1];
}
export function sampleColor(r) {
  const c = CAR_COLORS[Math.floor(r() * CAR_COLORS.length)], k = 0.85 + 0.3 * r();
  return c.map((v) => Math.max(0, Math.min(255, Math.round(v * k + (r() * 16 - 8)))));
}
/** Ego requests for a camera standing at (x0, y0) (KeiSim world), turned a little further in every frame. */
export function frames(n, x0, y0) {
  const out = [];
  for (let k = 0; k < n; k++) {
    const r = rng(1000 + k), yaw = k * 1.1, c = Math.cos(yaw), s = Math.sin(yaw);
    const at = (f, l) => [x0 + c * f - s * l, y0 + s * f + c * l];
    const veh = [], ped = [], bike = [];
    for (let i = 0; i < 9; i++) {
      const [L, W, H, kind] = sampleDims(r), [x, y] = at(7 + 6.5 * Math.floor(i / 3) + 2 * r(), (i % 3 - 1) * 3.4 + r() - 0.5);
      veh.push([i, x, y, yaw + (i % 4 === 3 ? Math.PI : 0) + (r() - 0.5) * 0.5, L, W, H, ...sampleColor(r), kind, i % 3 === 0 ? 1 : 0]);
    }
    for (let i = 0; i < 6; i++) {
      const [x, y] = at(4 + 2.2 * i + r(), (i % 2 ? 1 : -1) * (1.6 + 2.5 * r()));
      const col = () => [Math.round(60 + 180 * r()), Math.round(60 + 180 * r()), Math.round(60 + 180 * r())];
      ped.push([i, x, y, yaw + 6.28 * r(), i === 5 ? 1.0 + 0.35 * r() : 1.5 + 0.35 * r(), ...col(), ...col(), 224, 180, 150, 6.28 * r()]);
    }
    for (let i = 0; i < 3; i++) {
      const [x, y] = at(6 + 5 * i, 5.2 - 10.4 * (i % 2));
      const col = () => [Math.round(40 + 200 * r()), Math.round(40 + 200 * r()), Math.round(40 + 200 * r())];
      bike.push([i, x, y, yaw + (i % 2 ? Math.PI : 0), i, 1.55 + 0.25 * r(), ...col(), ...col(), ...col(), 224, 180, 150,
        0.1 * (r() - 0.5), 0.2 * (r() - 0.5), 6.28 * r(), 6.28 * r(), i === 2 ? 1 : 0, 0, 0.3 * r()]);
    }
    out.push({ cam: { pos: [x0, y0, 1.3], fwd: [c, s, -0.06], vfov: 61.6 }, t: 3.5 * k, veh, ped, bike, labels: true, reset: k === 0 });
  }
  return out;
}

/** Hashes of the RGB and label images of every frame of every mode, drawn from the tree at `root`. */
export async function render(browser, root, { town = '1000', n = 6, modes = [''], w = 320, h = 160 } = {}) {
  const server = serveTree(root);
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const out = {};
  try {
    for (const mode of modes) {
      const page = await browser.newPage();
      const errs = [];
      page.on('pageerror', (e) => errs.push(e.message));
      await page.goto(`http://127.0.0.1:${server.address().port}/index.html?shot=1&ego=1&town=${town}&w=${w}&h=${h}&q=medium${mode ? '&' + mode : ''}`,
        { waitUntil: 'load', timeout: 180000 });
      await page.waitForFunction('window.__ready === true', { timeout: 900000, polling: 100 });
      const origin = await page.evaluate(() => { const T = window.__town, c = T.views[0]; return [c.x + T.origin[0], T.origin[1] - c.z, window.__errors.length]; });
      if (origin[2] || errs.length) throw new Error(`page errors in ${root}: ${errs.join('; ')} ${JSON.stringify(await page.evaluate(() => window.__errors))}`);
      const hashes = [];
      for (const q of frames(n, origin[0], origin[1])) {
        const r = await page.evaluate((req) => { const o = window.__ego(req); return { rgb: o.rgb, sem: o.sem }; }, q);
        const sem = Buffer.from(r.sem, 'base64');
        const count = (id) => { let c = 0; for (const v of sem) if (v === id) c++; return c; };
        hashes.push({ rgb: crypto.createHash('sha256').update(r.rgb).digest('hex').slice(0, 16),
          sem: crypto.createHash('sha256').update(r.sem).digest('hex').slice(0, 16), veh: count(11), ped: count(12) });
      }
      out[mode] = hashes;
      await page.close();
    }
  } finally { server.close(); }
  return out;
}

export async function launch(gl = 'hw', w = 320, h = 160) {
  const exe = findBrowser();
  if (!exe) throw new Error('No Chromium/Edge/Chrome found - set CHROME=/path/to/browser');
  const glFlags = gl === 'soft' ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
    : process.platform === 'win32' ? ['--use-angle=d3d11', '--enable-gpu'] : ['--enable-gpu'];
  return puppeteer.launch({ executablePath: exe, headless: true,
    args: [...glFlags, '--ignore-gpu-blocklist', '--enable-webgl', '--no-sandbox', '--no-first-run', '--disable-extensions', `--window-size=${w},${h}`],
    defaultViewport: { width: w, height: h, deviceScaleFactor: 1 }, protocolTimeout: 900000 });
}

/** Compare this tree with the base ref; returns { ok, lines }. */
export async function compare({ base = 'origin/main', town = '1000', n = 6, modes = ['', 'cardetail=1'], gl = 'hw' } = {}) {
  const b = baseTree(base), lines = [`base ${base} (${b.sha.slice(0, 7)}) against this tree, town ${town}, ${n} frames`];
  const browser = await launch(gl);
  let ok = true;
  try {
    const A = await render(browser, b.dir, { town, n, modes }), B = await render(browser, here, { town, n, modes });
    for (const mode of modes) {
      let same = 0, withCars = 0, withPeds = 0;
      A[mode].forEach((a, i) => {
        const c = B[mode][i];
        if (a.rgb === c.rgb && a.sem === c.sem) same++;
        else lines.push(`  DIFF ${mode || 'default'} frame ${i}: rgb ${a.rgb === c.rgb ? 'same' : 'differs'}, labels ${a.sem === c.sem ? 'same' : 'differ'}`);
        if (a.veh > 50) withCars++;
        if (a.ped > 20) withPeds++;
      });
      const good = same === n && withCars >= 5 && withPeds >= 5;
      ok = ok && good;
      lines.push(`${good ? 'ok  ' : 'FAIL'} ${(mode || 'ego default').padEnd(14)} ${same}/${n} frames identical (RGB and labels); cars in ${withCars}, pedestrians in ${withPeds}`);
    }
  } finally { await browser.close(); }
  return { ok, lines };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = {};
  for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] !== undefined && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
  const r = await compare({ base: args.base, town: args.town, n: Number(args.frames || 6), gl: args.gl, modes: args.modes !== undefined ? args.modes.split('|') : undefined });
  for (const l of r.lines) console.log(l);
  console.log(r.ok ? 'RESULT: OK' : 'RESULT: FAIL');
  process.exit(r.ok ? 0 : 1);
}
