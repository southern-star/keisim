// Two ONNX files in the browser's onnxruntime-web on the same inputs (ReconView G2, docs/recon_view.md): are the
// outputs they share the same to the bit? The model with the depth head (scripts/export_onnx.py) has to drive exactly
// as the released one.
//   node tools/onnx_same.mjs [--a models/keipilot_mc.onnx] [--b models/keipilot_mc_depth.onnx] [--ep webgpu|wasm] [--n 8]
//                            [--ort node_modules/onnxruntime-web/dist/]
// Inputs: images of noise, of smooth shades and of blocks; the vectors from a normal distribution (seeded). Prints per
// output of A how many of the N inputs gave different values (and the largest difference), for A run twice (is the
// backend itself repeatable?) and for A against B. Exit code 1 when A and B differ.
import fs from 'node:fs';
import puppeteer from 'puppeteer-core';
import { createServer } from './serve.mjs';

const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
const exe = process.env.CHROME || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  'C:/Program Files/Google/Chrome/Application/chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find((p) => fs.existsSync(p));
if (!exe) { console.error('No Chrome found - set CHROME=/path/to/browser'); process.exit(1); }
const server = createServer();
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const browser = await puppeteer.launch({
  executablePath: exe, headless: true,
  args: ['--enable-gpu', '--enable-unsafe-webgpu', ...(process.platform === 'linux' ? ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'] : []),
    '--ignore-gpu-blocklist', '--no-sandbox', '--no-first-run', '--disable-extensions',
    ...(process.env.HTTPS_PROXY ? [`--proxy-server=${process.env.HTTPS_PROXY}`] : [])],
  protocolTimeout: 900000,
});
let failed = false;
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => { console.log(`[pageerror] ${e.message}`); failed = true; });
  const base = `http://127.0.0.1:${server.address().port}/`;
  await page.goto(`${base}package.json`, { waitUntil: 'load' });          // any document of this origin
  const version = /ORT_VERSION = '([\d.]+)'/.exec(fs.readFileSync(new URL('../src/pilot.js', import.meta.url), 'utf8'))[1];
  const r = await page.evaluate(async (o) => {
    const ort = await import(`${o.ort}${o.ep === 'webgpu' ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs'}`);
    ort.env.wasm.wasmPaths = o.ort;
    const open = async (url) => ort.InferenceSession.create(new Uint8Array(await (await fetch(url)).arrayBuffer()), { executionProviders: [o.ep] });
    const a = await open(o.a), b = await open(o.b);
    let seed = 12345;
    const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    const normal = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    const dims = { imgs: [1, 2, 3, 160, 320], cond: [1, 7], speed: [1, 3], stand: [1, 3] };
    function feeds(kind) {
      const f = {};
      for (const name of a.inputNames) {
        const d = dims[name], data = new Float32Array(d.reduce((x, y) => x * y, 1));
        if (name !== 'imgs') for (let i = 0; i < data.length; i++) data[i] = normal();
        else if (kind === 0) for (let i = 0; i < data.length; i++) data[i] = Math.floor(256 * rnd());
        else {                                              // smooth shades, or blocks of 16 pixels
          const p = Array.from({ length: 12 }, () => rnd());
          for (let c = 0; c < 6; c++) for (let v = 0; v < 160; v++) for (let u = 0; u < 320; u++) {
            const x = kind === 1 ? u / 320 : Math.floor(u / 16) / 20, y = kind === 1 ? v / 160 : Math.floor(v / 16) / 10;
            data[(c * 160 + v) * 320 + u] = Math.round(255 * (0.5 + 0.5 * Math.sin(6.28 * (p[c] + 3 * p[c + 6] * x + 2 * p[(c + 3) % 12] * y))));
          }
        }
        f[name] = { data, dims: d };
      }
      return f;
    }
    const run = async (s, f) => {
      const t = {};
      for (const [k, v] of Object.entries(f)) t[k] = new ort.Tensor('float32', v.data.slice(), v.dims);
      const out = await s.run(t), res = {};
      for (const k of s.outputNames) res[k] = out[k].data.slice();
      return res;
    };
    const diff = (x, y) => {
      const p = new Uint8Array(x.buffer, x.byteOffset, x.byteLength), q = new Uint8Array(y.buffer, y.byteOffset, y.byteLength);
      let same = p.length === q.length, max = 0;
      for (let i = 0; same && i < p.length; i++) same = p[i] === q[i];
      if (!same) for (let i = 0; i < Math.min(x.length, y.length); i++) max = Math.max(max, Math.abs(Number(x[i]) - Number(y[i])));
      return { same, max };
    };
    const names = a.outputNames.filter((k) => b.outputNames.includes(k));
    const twice = Object.fromEntries(names.map((k) => [k, { n: 0, max: 0 }])), ab = Object.fromEntries(names.map((k) => [k, { n: 0, max: 0 }]));
    for (let i = 0; i < o.n; i++) {
      const f = feeds(i % 3);
      const a1 = await run(a, f), a2 = await run(a, f), b1 = await run(b, f);
      for (const k of names) {
        let d = diff(a1[k], a2[k]); if (!d.same) { twice[k].n++; twice[k].max = Math.max(twice[k].max, d.max); }
        d = diff(a1[k], b1[k]); if (!d.same) { ab[k].n++; ab[k].max = Math.max(ab[k].max, d.max); }
      }
    }
    return { names, twice, ab, onlyB: b.outputNames.filter((k) => !a.outputNames.includes(k)) };
  }, { a: args.a || 'models/keipilot_mc.onnx', b: args.b || 'models/keipilot_mc_depth.onnx', ep: args.ep || 'webgpu', n: Number(args.n || 8),
    ort: args.ort ? `${base}${args.ort}` : `https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/` });
  const n = Number(args.n || 8);
  console.log(`${args.ep || 'webgpu'}, onnxruntime-web ${version}, ${n} inputs; outputs only in B: ${r.onlyB.join(', ') || '-'}`);
  for (const k of r.names) {
    console.log(`  ${k.padEnd(12)} A twice: ${r.twice[k].n ? `${r.twice[k].n} differ (max ${r.twice[k].max.toExponential(2)})` : 'the same'};  A and B: ${r.ab[k].n ? `${r.ab[k].n} differ (max ${r.ab[k].max.toExponential(2)})` : 'the same'}`);
    if (r.ab[k].n) failed = true;
  }
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
