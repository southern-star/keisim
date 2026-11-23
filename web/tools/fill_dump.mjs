// ReconFill's training data (docs/recon_fill.md): opens index.html?pilot=mc&recon=model&fill=oracle in headless Chrome,
// drives `--sec` seconds through window.__pilotRun and, every `--every` seconds, writes a sample of that model frame,
// all from ReconView's chase camera (320x160, top row first; depths as 16 bit codes, src/unproject.js, 0: no surface):
//   in_rgb, in_dep        the view without a fill (the current frame and the keyframes): what a fill model is given
//   loose_rgb, loose_dep  the current frame alone by a wide edge ratio (src/fill.js FILL.loose)
//   gt_rgb, gt_dep        the town itself: the colours by the model frames' pipeline, the depth of the depth pass
//   prev_rgb, prev_dep    the town of the fill before this one (FILL.every model frames earlier), as a mesh, drawn from
//                         this frame's chase camera: what "the last fill, seen from now" is when the fill is right
// into DIR/NAME/<array>.npy (N x 160 x 320 x 3 uint8, N x 160 x 320 uint16; numpy.load(..., mmap_mode='r')) and
// DIR/NAME/meta.json (per sample: the town's time, the car's pose and speed, the keyframes, the fill before's pose).
//   node tools/fill_dump.mjs --out DIR [--name NAME] [--town 1000] [--tdir .towns] [--sec 600] [--every 1]
//                            [--mindist 1 --maxwait 10]   (no sample while the car has moved less than --mindist metres
//                            since the last one, unless --maxwait seconds have passed: a red light gives few samples)
//                            [--preview 4]   (the first samples' images as DIR/NAME/preview_<i>.png, to look at)
//                            [--traffic 0] [--cars N] [--bikes N] [--parked N] [--model models/x.onnx]
//                            [--ort node_modules/onnxruntime-web/dist/] [--ep webgpu|wasm] [--gl hw|soft]
// Headers and meta.json are written again every 100 samples, so a run that dies keeps what it had (meta.json says
// "done": true only at the end). The last line: samples, seconds of wall time per sample, bytes per sample.
// Needs models/keipilot_mc_depth.onnx and network access to cdn.jsdelivr.net (onnxruntime-web, three.js).
import fs from 'node:fs';
import zlib from 'node:zlib';
import puppeteer from 'puppeteer-core';
import { createServer } from './serve.mjs';

const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
if (!args.out) { console.error('--out DIR is needed'); process.exit(1); }
const town = args.town || '1000', sec = Number(args.sec || 600), every = Number(args.every || 1);
const minDist = Number(args.mindist ?? 1), maxWait = Number(args.maxwait ?? 10), preview = Number(args.preview || 0);
const dir = `${args.out}/${args.name || town}`;
const exe = process.env.CHROME || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  'C:/Program Files/Google/Chrome/Application/chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find((p) => fs.existsSync(p));
if (!exe) { console.error('No Chrome found - set CHROME=/path/to/browser'); process.exit(1); }
const glFlags = args.gl === 'soft' ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
  : ['--enable-gpu', '--enable-unsafe-webgpu',
    ...(process.platform === 'linux' ? ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'] : [])];

const W = 320, H = 160, HEADER = 128;
const ARRAYS = { in_rgb: 3, in_dep: 0, loose_rgb: 3, loose_dep: 0, gt_rgb: 3, gt_dep: 0, prev_rgb: 3, prev_dep: 0 };   // channels; 0: uint16
/** A .npy header of HEADER bytes for n samples of an array (version 1.0, C order). */
function npyHeader(n, ch) {
  const dict = `{'descr': '${ch ? '|u1' : '<u2'}', 'fortran_order': False, 'shape': (${n}, ${H}, ${W}${ch ? `, ${ch}` : ''}), }`;
  const buf = Buffer.alloc(HEADER, 0x20);
  buf.write('\x93NUMPY', 0, 'latin1'); buf[6] = 1; buf[7] = 0;
  buf.writeUInt16LE(HEADER - 10, 8);
  buf.write(dict, 10, 'latin1');
  buf[HEADER - 1] = 0x0a;
  return buf;
}
/** An 8 bit RGB PNG (the previews). */
function png(rgb, w, h) {
  const raw = Buffer.alloc(h * (1 + 3 * w));
  for (let y = 0; y < h; y++) rgb.copy(raw, y * (1 + 3 * w) + 1, 3 * y * w, 3 * (y + 1) * w);
  const chunk = (type, data) => {
    const len = Buffer.alloc(4), body = Buffer.concat([Buffer.from(type, 'latin1'), data]), crc = Buffer.alloc(4);
    len.writeUInt32BE(data.length); crc.writeUInt32BE(zlib.crc32(body) >>> 0);
    return Buffer.concat([len, body, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}
/** A depth image's codes as a picture: near bright, no surface black. */
function depthRgb(buf) {
  const out = Buffer.alloc(3 * W * H);
  for (let i = 0; i < W * H; i++) { const q = buf.readUInt16LE(2 * i), v = q ? 255 - Math.min(230, Math.round(q / 65535 * 330)) : 0; out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = v; }
  return out;
}

const server = createServer();
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const browser = await puppeteer.launch({
  executablePath: exe, headless: true,
  args: [...glFlags, '--ignore-gpu-blocklist', '--no-sandbox', '--no-first-run', '--disable-extensions',
    ...(process.env.HTTPS_PROXY ? [`--proxy-server=${process.env.HTTPS_PROXY}`] : []), '--window-size=1280,720'],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 1 },
  protocolTimeout: 900000,
});
let failed = false;
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) console.log(`[error] ${m.text().slice(0, 300)}`); });
  page.on('pageerror', (e) => { console.log(`[pageerror] ${e.message}`); failed = true; });
  const q = new URLSearchParams({ pilot: 'mc', town, recon: 'model', fill: 'oracle', ...(args.ep ? { ep: args.ep } : {}), ...(args.tdir ? { tdir: args.tdir } : {}),
    ...(args.traffic ? { traffic: args.traffic } : {}), ...(args.cars ? { cars: args.cars } : {}),
    ...(args.bikes ? { bikes: args.bikes } : {}), ...(args.parked ? { parked: args.parked } : {}),
    ...(args.model ? { model: args.model } : {}), ...(args.ort ? { ort: args.ort } : {}) });
  const t0 = Date.now();
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html?${q}`, { waitUntil: 'load', timeout: 180000 });
  await page.waitForFunction('window.__pilotRun || /読み込めません/.test(document.getElementById("loadlabel").textContent)', { timeout: 600000, polling: 200 });
  if (!(await page.evaluate(() => !!window.__pilotRun))) throw new Error(await page.evaluate(() => document.getElementById('loadlabel').textContent));
  if (!(await page.evaluate(() => !!(window.__pilot.recon && window.__pilot.recon.fill)))) throw new Error('the page has no fill (?recon=model&fill=oracle)');
  const backend = await page.evaluate(async () => {          // from the town's time 0 and the car's start (tools/recon_check.mjs)
    await window.__pilotRun(0);
    await window.__pilot.idle();
    window.__pilot.capture(0, 1, true);
    await window.__pilot.idle();
    window.__sim(0);
    window.__pilot.reset();
    return window.__pilot.backend;
  });
  console.log(`town ${town}: ready in ${((Date.now() - t0) / 1000).toFixed(1)} s, backend ${backend}`);
  fs.mkdirSync(dir, { recursive: true });
  const fds = Object.fromEntries(Object.entries(ARRAYS).map(([k, ch]) => { const fd = fs.openSync(`${dir}/${k}.npy`, 'w'); fs.writeSync(fd, npyHeader(0, ch)); return [k, fd]; }));
  const meta = [], t1 = Date.now();
  /** The headers for the samples so far, and meta.json. */
  const save = (done) => {
    for (const [k, ch] of Object.entries(ARRAYS)) fs.writeSync(fds[k], npyHeader(meta.length, ch), 0, HEADER, 0);
    fs.writeFileSync(`${dir}/meta.json`, JSON.stringify({ town, sec, every, backend, minDist, maxWait, width: W, height: H, done,
      args: Object.fromEntries(Object.entries(args).filter(([k]) => k !== 'out')), samples: meta }, null, 1));
  };
  let last = null, lastT = -Infinity, skipped = 0, noPrev = 0, gz = 0;
  for (let t = every; t <= sec + 1e-6; t += every) {
    const s = await page.evaluate(async (d) => {
      const st = await window.__pilotRun(d.every - 0.1);
      const moved = d.last ? Math.hypot(st.car.x - d.last[0], st.car.y - d.last[1]) : Infinity;
      const take = moved >= d.minDist || d.waited >= d.maxWait;
      if (take) window.__pilot.recon.fill.armSample();
      const s2 = await window.__pilotRun(0.1);
      const sample = take ? window.__pilot.recon.fill.takeSample() : null;
      return { take, sample, car: s2.car, s: s2.s, resets: s2.resets, traffic: s2.traffic ? { n: s2.traffic.n, contacts: s2.traffic.contacts } : null };
    }, { every, last, minDist, maxWait, waited: t - lastT });
    if (!s.take) { skipped++; continue; }
    if (!s.sample || !s.sample.prev_rgb) { noPrev++; continue; }   // the drive restarted, or no fill before this one yet
    for (const k of Object.keys(ARRAYS)) {
      const buf = Buffer.from(s.sample[k], 'base64');
      if (buf.length !== W * H * (ARRAYS[k] || 2)) throw new Error(`${k}: ${buf.length} bytes`);
      fs.writeSync(fds[k], buf);
      if (meta.length < 50) gz += zlib.gzipSync(buf).length;      // how well the first samples would pack
      if (meta.length < preview) s.sample[`_${k}`] = buf;
    }
    if (meta.length < preview) {
      const row = (a, b) => Buffer.concat(Array.from({ length: H }, (_, y) => Buffer.concat([a.subarray(3 * y * W, 3 * (y + 1) * W), b.subarray(3 * y * W, 3 * (y + 1) * W)])));
      const sheet = Buffer.concat(['in', 'loose', 'gt', 'prev'].map((k) => row(s.sample[`_${k}_rgb`], depthRgb(s.sample[`_${k}_dep`]))));
      fs.writeFileSync(`${dir}/preview_${meta.length}.png`, png(sheet, 2 * W, 4 * H));
    }
    const p = s.sample.pose, pp = s.sample.prevPose;
    meta.push({ t, s: s.s, pose: p, v: s.car.v, keyframes: s.sample.keyframes, prevPose: pp, prevFrames: s.sample.prevFrames, resets: s.resets });
    last = [p.x, p.y]; lastT = t;
    if (meta.length % 100 === 0) {
      save(false);
      console.log(`  ${meta.length} samples, t ${t.toFixed(0)} s, ${((Date.now() - t1) / 1000 / meta.length).toFixed(2)} s per sample`);
    }
  }
  save(true);
  let bytes = 0;
  for (const k of Object.keys(ARRAYS)) { fs.closeSync(fds[k]); bytes += fs.statSync(`${dir}/${k}.npy`).size; }
  const wall = (Date.now() - t1) / 1000, n = meta.length, packed = Math.min(n, 50);
  console.log(`${dir}: ${n} samples of ${sec} s (${skipped} times the car had not moved, ${noPrev} without a fill before), ${wall.toFixed(0)} s: ${n ? (wall / n).toFixed(2) : '-'} s per sample, ` +
    `${n ? (bytes / n / 1e6).toFixed(3) : '-'} MB per sample${packed ? ` (gzip, each array of the first ${packed} samples on its own: ${(gz / packed / 1e6).toFixed(3)} MB)` : ''}`);
} catch (e) {
  console.error(e); failed = true;
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
