// Offline replay of captured driving frames through the existing ReconView mesh and keyframe implementation.
import fs from 'node:fs';
import readline from 'node:readline';
import puppeteer from 'puppeteer-core';
import { createServer } from './serve.mjs';

const gl = process.argv.includes('--soft') ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] : ['--enable-gpu'];
const chrome = process.env.CHROME || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(p => fs.existsSync(p));
if (!chrome) throw new Error('Set CHROME to a local Chromium executable');
const server = createServer();
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
const reply = obj => process.stdout.write(JSON.stringify(obj) + '\n');
try {
  browser = await puppeteer.launch({ executablePath: chrome, headless: true,
    args: [...gl, '--no-sandbox', '--enable-webgl', '--ignore-gpu-blocklist', '--disable-background-networking'] });
  const page = await browser.newPage();
  const origin = `http://127.0.0.1:${server.address().port}`;
  // Same-origin module loading, with a blank document and local three.js. No town or pilot is running here.
  await page.goto(origin + '/package.json');
  await page.setContent(`<script type="importmap">{"imports":{"three":"${origin}/node_modules/three/build/three.module.js","three/addons/":"${origin}/node_modules/three/examples/jsm/"}}</script><body></body>`);
  const gpu = await page.evaluate(async () => {
    const THREE = await import('three');
    const { createRecon } = await import('/src/recon.js');
    const { rigShots } = await import('/src/rig.js');
    const rig = rigShots('wide_narrow');
    const renderer = new THREE.WebGLRenderer();
    const gl = renderer.getContext(), debug = gl.getExtension('WEBGL_debug_renderer_info');
    const gpu = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : 'unknown';
    const w = 256, h = 128;
    function mode(name) {
      let capture;
      const recon = createRecon({ renderer, mode: name, rig, scene: new THREE.Scene(),
        ctx: { mat: { cache: new Map() } }, T: { origin: [0, 0], P: ([x, y]) => [x, -y] },
        views: rig.map(r => ({ w: r.shots[0].width, h: r.shots[0].height })), params: new URLSearchParams(),
        // Reuse the same bare geometry and observers exposed to ReconFill, leaving the fill disabled.
        createFill(view) {
          const rt = new THREE.WebGLRenderTarget(w, h);
          rt.texture.colorSpace = THREE.SRGBColorSpace;
          const rgba = new Uint8Array(w * h * 4);
          capture = (pose, accumulated) => view.bare(() => {
            view.carFrame(pose.x, pose.y, pose.yaw);
            view.observed(true, accumulated);
            view.rr.setRenderTarget(rt);
            view.rr.setClearColor(0x12131c, 0);
            view.rr.render(view.rscene, view.measCam);
            view.rr.readRenderTargetPixels(rt, 0, 0, w, h, rgba);
            view.rr.setRenderTarget(null);
            view.rr.setClearColor(0x12131c, 1);
            view.observed(true);
            // Binary coverage in alpha. Native GL row order is converted to top-first by the Python caller.
            let s = '';
            for (let off = 0; off < rgba.length; off += 8192) s += String.fromCharCode(...rgba.subarray(off, off + 8192));
            return btoa(s);
          });
          return null;
        }
      });
      return { recon, capture };
    }
    const oracle = mode('oracle'), model = mode('model');
    const bytes = str => Uint8Array.from(atob(str), ch => ch.charCodeAt(0));
    window.replayRealityFrame = q => {
      if (q.reset) { oracle.recon.reset(); model.recon.reset(); }
      const [x, y, yaw] = q.pose, pose = {x, y, yaw};
      const rgb = q.rgb.map(bytes), seg = bytes(q.seg);
      const dep = q.dep.map(s => new Uint16Array(bytes(s).buffer));
      const predicted = new Float32Array(bytes(q.predicted).buffer);
      oracle.recon.frame(pose, {dep}, rgb, seg);
      model.recon.frame(pose, {}, rgb, seg, predicted);
      return { oracle: oracle.capture(pose, false), model: model.capture(pose, false),
        model_accum: model.capture(pose, true) };
    };
    return gpu;
  });
  reply({ ready: true, gpu });
  for await (const line of readline.createInterface({input: process.stdin})) {
    const q = JSON.parse(line);
    try { reply({ id: q.id, images: await page.evaluate(q => window.replayRealityFrame(q), q) }); }
    catch (e) { reply({ id: q.id, error: String(e) }); break; }
  }
} finally {
  if (browser) await browser.close();
  server.close();
}
