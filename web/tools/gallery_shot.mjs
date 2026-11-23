// Stills of the model galleries (index.html?gallery=cars | people, src/gallery.js) with a headless browser.
//   node tools/gallery_shot.mjs --gallery cars [--cams "az,el,dist,x,y,z;..."] [--t 0] [--out shots/cars] [--w 1280 --h 720]
//                               [--q high] [--query "only=compact&n=3"] [--gl hw|soft] [--eval "JS"]
//   --cams   ';'-separated orbit cameras: azimuth and elevation in degrees (az 0 looks at a model's front, 90 at its
//            left side, el 90 straight down), distance and target in metres. Default: the gallery's own camera.
//   --t      the gallery's time in seconds, one or several ('0,0.2,0.4': every camera at every time)
//   --query  further URL parameters for the gallery
//   --eval   an expression evaluated in the page after the stills, printed as JSON
// Writes <out>_<n>.png and prints draw calls and triangles of each.
import fs from 'node:fs';
import path from 'node:path';
import { createServer, root } from './serve.mjs';
import { launch } from './golden.mjs';

const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] !== undefined && !/^--[a-z]/.test(process.argv[i + 1]) ? process.argv[++i] : '1'; args[k] = v; } }
const W = Number(args.w || 1280), H = Number(args.h || 720), kind = args.gallery || 'cars';
const out = path.resolve(root, args.out || `shots/gallery_${kind}`);
fs.mkdirSync(path.dirname(out), { recursive: true });

const server = createServer();
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const browser = await launch(args.gl || 'hw', W, H);
const logs = [];
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error') logs.push(`[console.error] ${m.text()}`); });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  const q = new URLSearchParams({ shot: '1', gallery: kind, w: String(W), h: String(H), q: args.q || 'high' });
  for (const [k, v] of new URLSearchParams(args.query || '')) q.set(k, v);
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html?${q}`, { waitUntil: 'load', timeout: 120000 });
  await page.waitForFunction('window.__ready === true', { timeout: 600000, polling: 200 });
  const errors = await page.evaluate(() => window.__errors);
  const cams = args.cams ? args.cams.split(';').map((s) => s.trim()).filter(Boolean).map((s) => s.split(',').map(Number)) : [null];
  const times = String(args.t ?? '0').split(',').map(Number);
  let n = 0;
  for (const cam of cams) for (const t of times) {
    await page.evaluate((c, tt) => { if (c) window.__orbit(...c); window.__sim(tt); }, cam, t);
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(r)))));
    const file = `${out}_${n++}.png`;
    await page.screenshot({ path: file });
    const st = await page.evaluate(() => ({ calls: window.__stats.calls, triangles: window.__stats.triangles }));
    console.log(`saved ${path.relative(root, file)}  cam ${cam ? cam.join(',') : 'default'}  t ${t}  calls=${st.calls} tris=${st.triangles}`);
  }
  if (args.eval) console.log(JSON.stringify(await page.evaluate(args.eval)));
  if (errors && errors.length) { console.log('ERRORS:'); for (const e of errors) console.log(` - [${e.module}] ${String(e.message).split('\n').slice(0, 6).join('\n   ')}`); process.exitCode = 1; }
} catch (e) {
  console.log('SHOT FAILED:', e.message); process.exitCode = 1;
} finally {
  if (logs.length) { console.log('PAGE LOGS:'); for (const l of logs.slice(0, 30)) console.log(' ', l.slice(0, 600)); }
  await browser.close(); server.close();
}
