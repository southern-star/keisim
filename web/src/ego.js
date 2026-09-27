// Ego-camera mode (KeiSim driving data and closed-loop evaluation).
//   index.html?shot=1&ego=1&town=<seed>&w=320&h=160
// window.__ego(request) renders one KeiSim camera frame on demand (there is no animation loop):
// the anime-styled RGB image through the normal pipeline, plus an optional pixel-exact semantic
// label image with the same classes as KeiSim's own renderer (keisim/config.py SEM_NAMES).
import * as THREE from 'three';

export const SEM = {
  sky: 0, road: 1, marking: 2, sidewalk: 3, terrain: 4, building: 5, vegetation: 6, pole: 7,
  tl_red: 8, tl_yellow: 9, tl_green: 10, vehicle: 11, pedestrian: 12,
};

// roads.js materials (sRGB hex) -> class. Anything else in that module counts as road.
const ROAD_COLORS = { '6f7176': SEM.road, 'b9b8b1': SEM.road, 'd5d0c5': SEM.sidewalk, 'c7c5bd': SEM.sidewalk, 'f2f0ea': SEM.marking };
const MODULE_SEM = { ground: SEM.terrain, roads: SEM.road, houses: SEM.building, trees: SEM.vegetation,
  signals: SEM.pole, poles: SEM.pole };

const _box = new THREE.Box3();

/** Resolve every static mesh's class before static batching (batch2 keeps userData.sem in its merge key). */
export function tagSemantics(ctx) {
  ctx.staticRoot.updateMatrixWorld(true);
  const moduleOf = (o) => { for (let p = o; p; p = p.parent) if (p.userData.module) return p.userData.module; return null; };
  ctx.staticRoot.traverse((o) => {
    if (!o.isMesh) return;
    const mod = moduleOf(o);
    let sem = MODULE_SEM[mod] ?? SEM.building;
    const m = Array.isArray(o.material) ? o.material[0] : o.material;
    const hex = m && m.color ? m.color.getHexString() : '';
    if (mod === 'roads') sem = ROAD_COLORS[hex] ?? SEM.road;
    else if (mod === 'houses') {
      if (o.geometry) {
        if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
        _box.copy(o.geometry.boundingBox).applyMatrix4(o.matrixWorld);
        if (_box.max.y - _box.min.y < 0.35 && _box.max.y < 0.8) sem = SEM.terrain;   // lots, fields, car parks, paths
      }
      if (m && m.color && sem === SEM.building) {
        const c = m.color.clone().convertLinearToSRGB();
        if (c.g > c.r * 1.08 && c.g > c.b * 1.05) sem = SEM.vegetation;            // hedges, gardens, crops
      }
    }
    o.userData.sem = sem;
  });
  // dynamic objects (never batched): keep explicit classes (actors), else use the module's class
  ctx.dynamicRoot.traverse((o) => {
    if (!o.isMesh || o.userData.labelColors) return;
    let has = false; for (let p = o; p; p = p.parent) if (p.userData.sem !== undefined) { has = true; break; }
    if (!has) o.userData.sem = MODULE_SEM[moduleOf(o)] ?? SEM.building;
  });
}

function b64(u8) {
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}

export function installEgo({ ctx, renderer, scene, camera, pipeline, sky, sunDir, W, H }) {
  const T = ctx.town;
  const [ox, oy] = T.origin;
  const gl = renderer.getContext();
  const rgba = new Uint8Array(W * H * 4), rgb = new Uint8Array(W * H * 3), sem = new Uint8Array(W * H);
  const rtLabel = new THREE.WebGLRenderTarget(W, H, { type: THREE.UnsignedByteType, samples: 0,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false });
  const glassMats = new Set();
  for (const [k, m] of ctx.mat.cache) if (k.startsWith('glass|')) glassMats.add(m);

  // ---------------------------------------------------------------- label materials
  const labelMats = new Map();
  const white = new THREE.MeshBasicMaterial({ toneMapped: false, fog: false });
  white.color.setRGB(1, 1, 1, THREE.LinearSRGBColorSpace);
  function labelMaterial(id, src) {
    const cut = src && src.alphaTest > 0 && (src.map || src.alphaMap) ? (src.alphaMap || src.map) : null;
    const po = src && src.polygonOffset ? `${src.polygonOffsetFactor},${src.polygonOffsetUnits}` : '';
    const side = src ? src.side : THREE.FrontSide;
    const key = `${id}|${cut ? cut.uuid + '@' + src.alphaTest : ''}|${po}|${side}|${src ? src.depthWrite : true}`;
    let m = labelMats.get(key);
    if (m) return m;
    m = new THREE.MeshBasicMaterial({ side, toneMapped: false, fog: false, depthWrite: src ? src.depthWrite : true });
    m.color.setRGB(id / 255, 0, 0, THREE.LinearSRGBColorSpace);
    if (po) { m.polygonOffset = true; m.polygonOffsetFactor = src.polygonOffsetFactor; m.polygonOffsetUnits = src.polygonOffsetUnits; }
    if (cut) {
      m.alphaMap = cut; m.alphaTest = src.alphaTest;
      if (!src.alphaMap) {   // cut-out lives in the colour map's alpha channel
        m.onBeforeCompile = (sh) => { sh.fragmentShader = sh.fragmentShader.replace('#include <alphamap_fragment>', 'diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).a;'); };
        m.customProgramCacheKey = () => 'label-alpha-a';
      }
    }
    labelMats.set(key, m);
    return m;
  }
  const semOf = (o) => { for (let p = o; p; p = p.parent) if (p.userData.sem !== undefined) return p.userData.sem; return undefined; };

  function renderLabels() {
    const swaps = [], hidden = [];
    scene.traverseVisible((o) => {
      if (!(o.isMesh || o.isPoints || o.isLine || o.isSprite)) return;
      const src = Array.isArray(o.material) ? o.material[0] : o.material;
      if (o.name === 'sky' || o.userData.labelSkip || !o.isMesh) { hidden.push(o); return; }
      if (o.userData.labelColors) { swaps.push([o, o.material, o.instanceColor]); return; }
      if (src && (src.isShaderMaterial || src.isRawShaderMaterial) && !glassMats.has(src)) { hidden.push(o); return; }
      const id = semOf(o) ?? SEM.building;
      swaps.push([o, o.material, o.instanceColor, id]);
    });
    for (const o of hidden) o.visible = false;
    for (const [o, mat, ic, id] of swaps) {
      if (o.userData.labelColors) { o.material = white; o.instanceColor = o.userData.labelColors; continue; }
      o.material = labelMaterial(id, Array.isArray(mat) ? mat[0] : mat);
      if (o.isInstancedMesh) o.instanceColor = null;
    }
    const bg = scene.background, fog = scene.fog;
    scene.background = null; scene.fog = null;
    renderer.shadowMap.needsUpdate = false;
    camera.layers.enableAll();
    renderer.setRenderTarget(rtLabel);
    renderer.setClearColor(0x000000, 1.0);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.readRenderTargetPixels(rtLabel, 0, 0, W, H, rgba);
    renderer.setRenderTarget(null);
    scene.background = bg; scene.fog = fog;
    for (const [o, mat, ic] of swaps) { o.material = mat; o.instanceColor = ic; }
    for (const o of hidden) o.visible = true;
    for (let i = 0, k = 0; k < sem.length; i += 4, k++) sem[k] = rgba[i];
  }

  // ---------------------------------------------------------------- lighting (per episode)
  const base = { sun: sunDir.clone(), exposure: pipeline.compMat.uniforms.uExposure.value,
    fog: scene.fog ? scene.fog.density : 0 };
  /** light: {az, el} sun azimuth / elevation (deg, azimuth CCW from KeiSim +x), exposure, fog density. */
  function setLighting(light) {
    if (!light) { sunDir.copy(base.sun); pipeline.compMat.uniforms.uExposure.value = base.exposure; if (scene.fog) scene.fog.density = base.fog; }
    else {
      if (light.el !== undefined) {
        const az = light.az * Math.PI / 180, el = light.el * Math.PI / 180;
        sunDir.set(Math.cos(el) * Math.cos(az), Math.sin(el), -Math.cos(el) * Math.sin(az)).normalize();
      }
      if (light.exposure !== undefined) pipeline.compMat.uniforms.uExposure.value = light.exposure;
      if (light.fog !== undefined && scene.fog) scene.fog.density = light.fog;
    }
    ctx.shared.uSunDir.value.copy(sunDir);
    if (sky.uniforms && sky.uniforms.uSun) sky.uniforms.uSun.value.copy(sunDir);
  }

  // ---------------------------------------------------------------- per-frame state
  function setState(q) {
    if (q.light !== undefined) setLighting(q.light);
    const t = q.t || 0;
    ctx.time = t; ctx.shared.uTime.value = t;
    ctx.shared.uGust.value = 0.5 + 0.28 * Math.sin(t * 0.37) + 0.14 * Math.sin(t * 1.13 + 1.7) + 0.08 * Math.sin(t * 2.9 + 0.4);
    // camera: KeiSim world (x east, y north, z up) -> three.js (X = x - ox, Y = z, Z = -(y - oy))
    const [cx, cy, cz] = q.cam.pos, [fx, fy, fz] = q.cam.fwd;
    camera.fov = q.cam.vfov; camera.aspect = W / H; camera.near = 0.1; camera.far = 2500;
    camera.updateProjectionMatrix();
    camera.position.set(cx - ox, cz, -(cy - oy));
    camera.up.set(0, 1, 0);
    camera.lookAt(camera.position.x + fx, camera.position.y + fz, camera.position.z - fy);
    camera.updateMatrixWorld();
    ctx.player.position.set(camera.position.x, 0, camera.position.z);
    if (ctx.services.actors) ctx.services.actors.update(q.veh || [], q.ped || []);
    for (const fn of ctx._updates) { try { fn(0, t); } catch (e) { console.error('update error', e); } }
    sky.update(t, camera);
  }

  /** Render one ego frame. Images are bottom-up rows (OpenGL order), RGB / class id. */
  window.__ego = (q) => {
    const t0 = performance.now();
    if (q.reset && ctx.services.actors) ctx.services.actors.reset();
    setState(q);
    renderer.info.reset();
    pipeline.render(scene, camera, sunDir, q.t || 0);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) { rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2]; }
    const out = { w: W, h: H, rgb: b64(rgb), sem: null, calls: renderer.info.render.calls };
    if (q.labels) { renderLabels(); out.sem = b64(sem); }
    out.ms = +(performance.now() - t0).toFixed(2);
    return out;
  };
  window.__egoWarmup = () => {          // compile every program variant once (both passes)
    const c = T.views && T.views[0] ? T.views[0] : { x: 0, z: 0, yaw: 0 };
    const x = c.x + ox, y = oy - c.z;
    window.__ego({ cam: { pos: [x, y, 1.55], fwd: [1, 0, -0.1], vfov: 61.6 }, t: 0, labels: true });
    return true;
  };
}
