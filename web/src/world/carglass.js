// 車のガラス (redesign) — see-through window glass for the cars of ?cardetail=2 (src/world/cars2.js), next to the
// almost opaque glass of cars.js carGlass() that ?cardetail=1 keeps. Same look (sky where the pane faces up and at
// grazing angles, diagonal highlight streaks), with two numbers opened up: how opaque the tint is and how much sky it
// shows. Unlike carGlass() it writes depth: ego mode's label pass copies depthWrite, so a pane stays class 11
// (vehicle) whatever is drawn behind it, now that there is no dark box right behind the glass.
// Cached in M.cache under 'glass|...' like the other glass, so ego mode labels it (src/ego.js).
import * as THREE from 'three';
import { carGlass } from './cars.js';

/** kind -> [opacity of the tint, share of the sky reflection]. 'solid' is cars.js' glass itself. The cars of
 *  ?cardetail=2 take 'clear' unless ?carglass= says otherwise (src/main.js). */
export const CAR_GLASS = { tint: [0.42, 0.55], clear: [0.16, 0.3] };

export function carGlass2(M, kind = 'solid') {
  if (!CAR_GLASS[kind]) return carGlass(M);
  const key = `glass|car2:${kind}`;
  if (M.cache.has(key)) return M.cache.get(key);
  const [opacity, sky] = CAR_GLASS[kind];
  const m = new THREE.ShaderMaterial({
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
      uTint: { value: M.color('#1b2430') }, uSkyTop: { value: M.color(M.palette.skyMid) }, uSkyLow: { value: M.color('#e9eef2') },
      uOpacity: { value: opacity }, uSky: { value: sky },
    }]),
    vertexShader: /* glsl */`
      #include <common>
      #include <fog_pars_vertex>
      varying vec3 vW; varying vec3 vN;
      void main(){
        vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz;
        vN = normalize(mat3(modelMatrix) * normal);
        vec4 mvPosition = viewMatrix * wp;
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */`
      #include <common>
      #include <fog_pars_fragment>
      uniform vec3 uTint; uniform vec3 uSkyTop; uniform vec3 uSkyLow; uniform float uOpacity; uniform float uSky;
      varying vec3 vW; varying vec3 vN;
      void main(){
        vec3 V = normalize(cameraPosition - vW);
        vec3 N = normalize(vN); if (dot(N, V) < 0.0) N = -N;
        float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);
        vec3 R = reflect(-V, N);
        vec3 sky = mix(uSkyLow, uSkyTop, smoothstep(-0.1, 0.6, R.y));
        float k = clamp(uSky * (0.06 + 0.5 * fres + 0.5 * smoothstep(-0.05, 0.5, R.y)), 0.0, 0.85);
        vec3 col = mix(uTint, sky, k);
        float s = dot(vW, normalize(vec3(0.62, 0.78, 0.62))) * 0.9;
        float band = abs(fract(s * 0.5) - 0.5);
        float streak = smoothstep(0.035, 0.0, abs(band - 0.18)) * 0.9 + smoothstep(0.012, 0.0, abs(band - 0.25)) * 0.6;
        col += vec3(1.0) * streak * 0.45;
        // what the pane reflects hides what is behind it: the sky's share and the streaks add to the tint's opacity
        gl_FragColor = vec4(col, clamp(uOpacity + k * 0.45 + fres * 0.3 + streak * 0.3, 0.0, 1.0));
        #include <fog_fragment>
      }`,
    transparent: true, depthWrite: true, fog: true,
  });
  M.cache.set(key, m);
  return m;
}
