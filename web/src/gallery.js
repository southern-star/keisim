// ギャラリー — index.html?gallery=cars | people: KeiView's models on a flat ground, without a town, to look at them
// from every side (the car and pedestrian redesigns: src/gallery_cars.js, src/gallery_people.js).
//   drag: turn the camera   wheel / pinch: distance   right drag or shift + drag: move the target   Space: pause
//   ?cam=az,el,dist[,x,y,z]  camera (degrees, metres; az 0 looks at a model's front, 90 at its left side)
//   ?shot=1&t=2.5             a still for tools/gallery_shot.mjs: time stands at t, window.__orbit / window.__sim move it
import * as THREE from 'three';

const KINDS = { cars: './gallery_cars.js', people: './gallery_people.js' };

export async function installGallery({ kind, ctx, scene, camera, renderer, pipeline, sky, sunDir, params, shot, canvas }) {
  if (!KINDS[kind]) throw new Error(`?gallery=${kind}: one of ${Object.keys(KINDS).join(', ')}`);
  // the town the actor modules ask for: KeiSim's frame (x east, y north) on a flat ground at the origin
  ctx.town = { origin: [0, 0], P: ([x, y]) => [x, -y], rotYOf: (yaw) => Math.atan2(Math.cos(yaw), -Math.sin(yaw)), heightAt: () => 0, views: [] };
  window.__town = ctx.town;

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600).rotateX(-Math.PI / 2), ctx.mat.toon('#8f9196', { paint: 0.03 }));
  ground.receiveShadow = true; ground.name = 'gallery ground';
  scene.add(ground);
  const lines = [];                                    // a 1 m grid near the models, bolder every 5 m
  for (let k = -40; k <= 40; k++) {
    const y = k % 5 === 0 ? 0.006 : 0.004;
    lines.push(k, y, -40, k, y, 40, -40, y, k, 40, y, k);
  }
  const grid = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(lines, 3)),
    new THREE.LineBasicMaterial({ color: 0xa9abb0, transparent: true, opacity: 0.55, depthWrite: false }));
  grid.layers.set(ctx.LAYER_NO_OUTLINE); grid.userData.labelSkip = true;
  scene.add(grid);

  const mod = await import(KINDS[kind]);
  const show = await mod.setup({ ctx, scene, params, shot });

  // ---- orbit camera
  const orbit = { az: 35, el: 14, dist: 9, x: 0, y: 0.8, z: 0, ...(show.orbit || {}) };
  const cam = (params.get('cam') || '').split(',').filter((s) => s !== '').map(Number);
  ['az', 'el', 'dist', 'x', 'y', 'z'].forEach((k, i) => { if (Number.isFinite(cam[i])) orbit[k] = cam[i]; });
  function place() {
    orbit.el = Math.min(89, Math.max(-5, orbit.el)); orbit.dist = Math.min(120, Math.max(0.6, orbit.dist));
    const a = orbit.az * Math.PI / 180, e = orbit.el * Math.PI / 180;
    camera.position.set(orbit.x + orbit.dist * Math.cos(e) * Math.sin(a), orbit.y + orbit.dist * Math.sin(e), orbit.z + orbit.dist * Math.cos(e) * Math.cos(a));
    camera.up.set(0, 1, 0);
    if (orbit.el > 88.5) camera.up.set(-Math.sin(a), 0, -Math.cos(a));      // straight down: the model's front up the screen
    camera.lookAt(orbit.x, orbit.y, orbit.z);
    camera.updateMatrixWorld();
  }
  camera.fov = Number(params.get('fov') || 32); camera.near = 0.1; camera.far = 2500; camera.updateProjectionMatrix();
  let drag = null;
  const pts = new Map();
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, pinch: 0 };
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag || !pts.has(e.pointerId)) return;
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2) {                              // pinch: distance
      const [p, q] = [...pts.values()], d = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (drag.pinch) orbit.dist *= drag.pinch / Math.max(d, 1);
      drag.pinch = d;
      return;
    }
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) {
      const a = orbit.az * Math.PI / 180, k = orbit.dist * 0.0016;
      orbit.x -= dx * k * Math.cos(a); orbit.z += dx * k * Math.sin(a); orbit.y = Math.max(0, orbit.y + dy * k);
    } else { orbit.az -= dx * 0.35; orbit.el += dy * 0.3; }
  });
  const up = (e) => { pts.delete(e.pointerId); if (!pts.size) drag = null; else if (drag) drag.pinch = 0; };
  canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); orbit.dist *= Math.exp(e.deltaY * 0.0012); }, { passive: false });

  // ---- time and the frame loop (its own: no player, no town modules)
  let t = Number(params.get('t') || 0), paused = false, last = performance.now();
  addEventListener('keydown', (e) => {
    if (e.code === 'Space') { paused = !paused; e.preventDefault(); }
    if (show.key) show.key(e.code);
  });
  function draw(dt) {
    ctx.time = t; ctx.shared.uTime.value = t;
    if (show.update) show.update(dt, t);
    place();
    sky.update(t, camera);
    renderer.info.reset();
    pipeline.render(scene, camera, sunDir, t);
    const s = window.__stats; if (s) { s.calls = renderer.info.render.calls; s.triangles = renderer.info.render.triangles; }
  }
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = shot || paused ? 0 : Math.min(0.1, (now - last) / 1000); last = now;
    t += dt;
    try { draw(dt); } catch (e) { if (!frame.err) { frame.err = 1; console.error(e); window.__errors.push({ module: 'gallery', message: String(e && e.stack || e) }); } }
  }
  window.__orbit = (az, el, dist, x = orbit.x, y = orbit.y, z = orbit.z) => { Object.assign(orbit, { az, el, dist, x, y, z }); };
  window.__sim = (target) => { const dt = target - t; t = target; draw(Math.max(0, dt)); };
  window.__gallery = show;

  document.body.classList.add(shot ? 'shot' : 'playing', 'noui');
  document.title = `KeiView gallery — ${kind}`;
  if (!shot) {
    const cap = document.createElement('div');
    cap.style.cssText = 'position:fixed;left:12px;bottom:12px;z-index:5;padding:8px 12px;border-radius:8px;background:rgba(251,248,242,.86);color:#33304a;font:500 13px/1.6 "Noto Sans JP",sans-serif;max-width:min(92vw,720px)';
    cap.innerHTML = `<b>${show.title || kind}</b><br>${show.caption || ''}<br>ドラッグ: 回す　ホイール: 距離　右ドラッグ / Shift+ドラッグ: 注視点　Space: 一時停止`;
    document.body.appendChild(cap);
  }
  try { renderer.compile(scene, camera); } catch (e) { console.warn(e); }
  draw(0);
  requestAnimationFrame(frame);
}
