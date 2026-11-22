// The depth pass (ReconView, docs/recon_view.md): the scene drawn once more with every mesh's distance from the
// camera in place of its colour, as 16 bit codes (src/unproject.js, keisim/render/rays.py: q = round(65535 ln(r / 0.25)
// / ln(1600)), 0: no surface; r is the straight distance from the camera centre, not z). It follows the rules of
// the label pass of src/ego.js, so depth and label of a pixel are of the same object: the same objects hidden (sky,
// labelSkip, everything that is not a mesh, shader materials other than glass), the same cut-outs, polygon offsets,
// sides and depth writes. It needs no class ids, so it also runs in pilot mode (src/recon.js).
import * as THREE from 'three';
import { DEPTH_FAR, DEPTH_NEAR } from './unproject.js';

/** The glass materials (core/materials.js): the shader materials the label and depth passes draw. */
export function glassMaterials(ctx) {
  const out = new Set();
  for (const [k, m] of ctx.mat.cache) if (k.startsWith('glass|')) out.add(m);
  return out;
}

/** What the label and depth passes draw of the visible scene, and what they hide. */
export function passObjects(scene, glassMats) {
  const draw = [], hidden = [];
  scene.traverseVisible((o) => {
    if (!(o.isMesh || o.isPoints || o.isLine || o.isSprite)) return;
    const src = Array.isArray(o.material) ? o.material[0] : o.material;
    if (o.name === 'sky' || o.userData.labelSkip || !o.isMesh) { hidden.push(o); return; }
    if (!o.userData.labelColors && src && (src.isShaderMaterial || src.isRawShaderMaterial) && !glassMats.has(src)) { hidden.push(o); return; }
    draw.push(o);
  });
  return { draw, hidden };
}

const cutOf = (src) => (src && src.alphaTest > 0 && (src.map || src.alphaMap) ? (src.alphaMap || src.map) : null);

/** Materials that cover the same pixels share this key. */
export function flatKey(src) {
  const cut = cutOf(src);
  const po = src && src.polygonOffset ? `${src.polygonOffsetFactor},${src.polygonOffsetUnits}` : '';
  return `${cut ? cut.uuid + '@' + src.alphaTest : ''}|${po}|${src ? src.side : THREE.FrontSide}|${src ? src.depthWrite : true}`;
}

/** An unlit material that covers the pixels src covers: its cut-out, polygon offset, side and depth write.
 *  userData.alphaInMap: the cut-out is the alpha channel of src's colour map (ALPHA_IN_MAP in the fragment shader). */
export function flatLike(src) {
  const cut = cutOf(src);
  const m = new THREE.MeshBasicMaterial({ side: src ? src.side : THREE.FrontSide, toneMapped: false, fog: false, depthWrite: src ? src.depthWrite : true });
  if (src && src.polygonOffset) { m.polygonOffset = true; m.polygonOffsetFactor = src.polygonOffsetFactor; m.polygonOffsetUnits = src.polygonOffsetUnits; }
  if (cut) {
    m.alphaMap = cut; m.alphaTest = src.alphaTest;
    if (!src.alphaMap) m.userData.alphaInMap = true;
  }
  return m;
}
export const ALPHA_IN_MAP = ['#include <alphamap_fragment>', 'diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).a;'];

const ENCODE = /* glsl */`
  { float r = length(vDepthPos);
    float q = r > ${DEPTH_FAR.toFixed(1)} ? 0.0 : clamp(floor(65535.0 * log(max(r, 1e-6) / ${DEPTH_NEAR}) / ${Math.log(DEPTH_FAR / DEPTH_NEAR)} + 0.5), 1.0, 65535.0);
    float hi = floor(q / 256.0);
    gl_FragColor = vec4(hi / 255.0, (q - hi * 256.0) / 255.0, 0.0, 1.0); }`;

export function createDepthPass({ renderer, scene, glassMats }) {
  const mats = new Map();
  function material(src) {
    const key = flatKey(src);
    let m = mats.get(key);
    if (m) return m;
    m = flatLike(src);
    const inMap = !!m.userData.alphaInMap;
    m.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vDepthPos;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvDepthPos = mvPosition.xyz;');
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vDepthPos;')
        .replace('#include <dithering_fragment>', ENCODE);
      if (inMap) sh.fragmentShader = sh.fragmentShader.replace(...ALPHA_IN_MAP);
    };
    m.customProgramCacheKey = () => (inMap ? 'depth-alpha-a' : 'depth');
    mats.set(key, m);
    return m;
  }

  let swaps = null, hidden = null, saved = null;
  return {
    /** A render target for draw(): 8 bit RGBA, no filtering, no multisampling. */
    target: (w, h) => new THREE.WebGLRenderTarget(w, h, { type: THREE.UnsignedByteType, samples: 0,
      minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false }),
    /** Swap every drawn mesh's material for its depth material and hide the rest, until end(). */
    begin() {
      const objs = passObjects(scene, glassMats);
      hidden = objs.hidden;
      swaps = objs.draw.map((o) => [o, o.material]);
      for (const o of hidden) o.visible = false;
      for (const [o, mat] of swaps) o.material = material(Array.isArray(mat) ? mat[0] : mat);
      saved = { bg: scene.background, fog: scene.fog, auto: renderer.shadowMap.autoUpdate, target: renderer.getRenderTarget(),
        clear: renderer.getClearColor(new THREE.Color()), alpha: renderer.getClearAlpha() };
      scene.background = null; scene.fog = null;
      renderer.shadowMap.autoUpdate = false; renderer.shadowMap.needsUpdate = false;
    },
    /** The camera's depth image into rt (w x h); with rgba (Uint8Array 4wh) and out (Uint16Array wh): read back too,
     *  codes in GL row order (bottom row first). */
    draw(camera, rt, w, h, rgba, out) {
      camera.layers.enableAll();
      renderer.setRenderTarget(rt);
      renderer.setClearColor(0x000000, 1.0);
      renderer.clear();
      renderer.render(scene, camera);
      if (!out) return null;
      renderer.readRenderTargetPixels(rt, 0, 0, w, h, rgba);
      for (let i = 0, k = 0; k < out.length; i += 4, k++) out[k] = (rgba[i] << 8) | rgba[i + 1];
      return out;
    },
    end() {
      renderer.setRenderTarget(saved.target);
      renderer.setClearColor(saved.clear, saved.alpha);
      renderer.shadowMap.autoUpdate = saved.auto;
      scene.background = saved.bg; scene.fog = saved.fog;
      for (const [o, mat] of swaps) o.material = mat;
      for (const o of hidden) o.visible = true;
      swaps = hidden = saved = null;
    },
  };
}
