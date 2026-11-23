// onnxruntime-web in a worker for src/pilot.js: the driving model runs off the page's main thread, so its WebGPU
// dispatches and buffer read-backs no longer wait for the frames the page draws in between (in the page itself an
// inference took 3-5 times as long while the town was being drawn as with the drawing stopped). WASM runs here as
// well, in place of onnxruntime-web's own proxy worker.
// Messages: {type: 'init', ort, gpu, force, bytes} -> {type: 'ready', backend, adapter, inputNames, outputNames} or
// {type: 'error'}; {type: 'run', id, feeds: {name: {data, dims}}} -> {type: 'out', id, out: {name: {data, dims}}}.
// A second model in the same worker (src/fill.js, ?fill=model): {type: 'load', id, key, bytes} -> {type: 'loaded', id,
// inputNames, outputNames}, then {type: 'run', id, key, feeds}. From then on the runs of all sessions go one after
// another, in the order they were asked for.

/** A hardware WebGPU adapter, or null (as src/pilot.js picks it: SwiftShader is far slower than WASM). */
async function gpuAdapter() {
  if (!self.navigator.gpu) return null;
  for (const opt of [{ powerPreference: 'high-performance' }, {}, { powerPreference: 'low-power' }]) {
    try {
      const a = await self.navigator.gpu.requestAdapter(opt);
      const info = (a && a.info) || {};
      if (a && !a.isFallbackAdapter && !info.isFallbackAdapter && !/swiftshader/i.test(`${info.vendor} ${info.architecture}`)) return a;
    } catch (e) { /* try the next preference */ }
  }
  return null;
}

let ort = null, session = null, provider = '';
const extra = new Map();                                    // more sessions: key -> session
let chain = Promise.resolve();

async function init({ ort: dir, gpu, force, bytes }) {
  const adapter = gpu ? await gpuAdapter() : null;
  const webgpu = !!adapter || (gpu && force);               // ?ep=webgpu: try it even without a hardware adapter
  ort = await import(`${dir}${webgpu ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs'}`);
  ort.env.wasm.wasmPaths = dir;
  if (adapter) ort.env.webgpu.adapter = adapter;
  let backend = '';
  for (const ep of webgpu ? ['webgpu', 'wasm'] : ['wasm']) {
    try { session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] }); backend = ep; break; } catch (e) { console.warn(`[pilot worker] ${ep} unavailable`, e); }
  }
  if (!session) throw new Error('no execution provider');
  provider = backend;
  const info = adapter && adapter.info;
  return { backend, adapter: backend === 'webgpu' && info ? `${info.vendor} ${info.architecture}`.trim() : '',
    inputNames: session.inputNames, outputNames: session.outputNames };
}

async function run(feeds, s = session) {
  const inputs = {};
  for (const [k, f] of Object.entries(feeds)) inputs[k] = new ort.Tensor('float32', f.data, f.dims);
  const out = await s.run(inputs), res = {}, transfer = [];
  for (const [k, t] of Object.entries(out)) {
    let d = t.data;
    if (d.byteOffset !== 0 || d.byteLength !== d.buffer.byteLength) d = d.slice();   // its own buffer, to transfer
    res[k] = { data: d, dims: t.dims };
    transfer.push(d.buffer);
  }
  return { res, transfer };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') self.postMessage({ type: 'ready', ...(await init(m)) });
    else if (m.type === 'load') {
      const s = await ort.InferenceSession.create(m.bytes, { executionProviders: [provider] });
      extra.set(m.key, s);
      self.postMessage({ type: 'loaded', id: m.id, inputNames: s.inputNames, outputNames: s.outputNames });
    } else if (m.type === 'run') {
      let job;
      if (extra.size) {                                     // one run at a time across the sessions
        job = chain.then(() => run(m.feeds, m.key ? extra.get(m.key) : session));
        chain = job.catch(() => {});
      } else job = run(m.feeds);
      const { res, transfer } = await job;
      self.postMessage({ type: 'out', id: m.id, out: res }, transfer);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String((err && err.message) || err) });
  }
};
