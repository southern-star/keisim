// 車と歩行者 — KeiSim's NPC vehicles and pedestrians in the same cel style, posed every frame from
// the simulator state (ego mode, see src/ego.js). Pool slot i follows KeiSim actor i; a slot is only
// rebuilt when that actor's kind / size / colour changes (episode start or respawn), so posing a
// frame is just transforms. Models face local +Z with their origin on the ground at the centre.
// With ctx.mergeActors (pilot mode) each car is merged into one mesh per material, about a third of the draw calls;
// ego mode keeps the separate parts, so the frames KeiSim renders through KeiView stay exactly as they were.
// With ctx.carDetail (pilot mode) a car also gets a detailed model (src/world/cars.js), shown in the main view when
// it is within DETAIL_DIST of the camera; view('model') shows the plain cars everywhere, for the KeiPilot frame.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { buildCar, carGlass } from './cars.js';

const DETAIL_DIST = 70;           // m: detailed cars closer than this to the camera (main view)

export const SEM_VEHICLE = 11, SEM_PED = 12;

const q8 = (v) => Math.round(Math.min(255, Math.max(0, v)) / 16) * 16;

export function build(ctx) {
  const T = ctx.town;
  const M = ctx.mat;
  const root = new THREE.Group(); root.name = 'actors';
  const carRoot = new THREE.Group(); carRoot.name = 'cars'; carRoot.userData.sem = SEM_VEHICLE;
  const pedRoot = new THREE.Group(); pedRoot.name = 'pedestrians'; pedRoot.userData.sem = SEM_PED;
  root.add(carRoot, pedRoot);

  const mGlass = M.toon('#4b5a76', { paint: 0.02 });
  const mTire = M.toon('#2d2d34', { paint: 0.02 });
  const mTrim = M.toon('#3b3d45', { paint: 0.03 });
  const mChassis = M.toon('#55575e', { paint: 0.03 });
  const mHead = M.emissive('#fff4dc', 1.35);
  const mTail = M.toon('#8a2d33', { paint: 0.02 });
  const mTailOn = M.emissive('#ff4436', 2.4);
  const mPlateW = M.toon('#f2f0ea', { paint: 0.01 });
  const mPlateY = M.toon('#f3c534', { paint: 0.01 });
  const mCargo = M.toon('#e9e7e0', { paint: 0.05 });
  const HAIR = ['#2b2630', '#3d2f2a', '#5a4130', '#1f1d24', '#6b5845'];
  const colorMat = (rgb, paint = 0.04) => M.toon(new THREE.Color(`rgb(${q8(rgb[0])},${q8(rgb[1])},${q8(rgb[2])})`), { paint });

  // ------------------------------------------------------------------ vehicles
  function makeCar(kind, L, W, H, rgb) {
    const g = new THREE.Group();
    const k = ctx.kit(g);
    const body = colorMat(rgb);
    const tails = [];
    const wheel = (x, z, r, w) => k.cyl(r, r, w, mTire, [x, r, z], [0, 0, Math.PI / 2], 14);
    if (kind === 1) {                                   // small truck: cab + cargo box
      const cabL = Math.min(1.9, L * 0.3), cabH = Math.min(H * 0.82, 2.45);
      k.box(W * 0.94, 0.28, L * 0.96, mChassis, [0, 0.42, 0]);
      k.rbox(W, cabH - 0.45, cabL, 0.1, body, [0, 0.45 + (cabH - 0.45) / 2, L / 2 - cabL / 2]);
      k.box(W * 0.9, (cabH - 0.45) * 0.42, 0.04, mGlass, [0, 0.45 + (cabH - 0.45) * 0.68, L / 2 + 0.005]);
      k.rbox(W * 1.01, H - 0.55, L - cabL - 0.12, 0.05, mCargo, [0, 0.55 + (H - 0.55) / 2, -(cabL + 0.12) / 2]);
      for (const z of [L / 2 - cabL * 0.55, -L / 2 + 1.1, -L / 2 + 2.0]) for (const s of [-1, 1]) wheel(s * (W / 2 - 0.2), z, 0.42, 0.26);
      for (const s of [-1, 1]) {
        k.box(0.26, 0.14, 0.04, mHead, [s * (W / 2 - 0.25), 0.72, L / 2 + 0.01]);
        tails.push(k.box(0.22, 0.14, 0.04, mTail, [s * (W / 2 - 0.2), 0.62, -L / 2 - 0.01]));
      }
      k.box(0.33, 0.16, 0.02, mPlateW, [0, 0.5, -L / 2 - 0.02]);
      return { g, tails };
    }
    const tall = H > 1.62;
    const kei = L < 3.6;
    const gc = 0.2;                                     // ground clearance
    const hb = tall ? 0.36 * H : 0.43 * H;               // lower body height
    const yb = gc + hb;                                 // belt line
    const cabL = kei ? 0.74 * L : tall ? 0.68 * L : 0.52 * L;
    const cabZ = kei ? -0.06 * L : tall ? -0.08 * L : -0.07 * L;
    const cabH = H - yb;
    k.rbox(W, hb, L, 0.14, body, [0, gc + hb / 2, 0]);
    k.rbox(W * 0.9, cabH * 0.97, cabL, tall ? 0.12 : 0.16, mGlass, [0, yb + cabH * 0.97 / 2, cabZ]);
    k.rbox(W * 0.91, 0.07, cabL * 0.9, 0.03, body, [0, H - 0.035, cabZ]);                 // roof
    for (const s of [-1, 1]) k.box(0.05, cabH * 0.9, 0.09, body, [s * W * 0.445, yb + cabH * 0.45, cabZ + cabL * 0.12]); // B pillars
    k.box(W * 1.005, 0.08, L * 0.99, mTrim, [0, gc + 0.06, 0]);                            // side sill
    const wr = kei ? 0.27 : tall ? 0.33 : 0.31;
    for (const z of [L * 0.33, -L * 0.33]) for (const s of [-1, 1]) wheel(s * (W / 2 - 0.13), z, wr, 0.2);
    for (const s of [-1, 1]) {
      k.box(0.28, 0.12, 0.04, mHead, [s * (W / 2 - 0.26), gc + hb * 0.72, L / 2 + 0.005]);
      tails.push(k.box(0.24, 0.13, 0.04, mTail, [s * (W / 2 - 0.22), gc + hb * 0.74, -L / 2 - 0.005]));
    }
    k.box(0.33, 0.16, 0.02, kei ? mPlateY : mPlateW, [0, gc + hb * 0.38, -L / 2 - 0.012]);
    k.box(0.33, 0.16, 0.02, kei ? mPlateY : mPlateW, [0, gc + hb * 0.3, L / 2 + 0.012]);
    return { g, tails };
  }

  /** One mesh per material for a car built by makeCar (the tail lights stay one mesh of their own: they switch
   *  material when braking). Paint variation is in world space, so the merged car looks the same. */
  function mergeCar({ g, tails }) {
    const parts = new Map();
    for (const o of g.children) {
      if (!o.isMesh) continue;
      const key = tails.includes(o) ? 'tail' : o.material.uuid;
      if (!parts.has(key)) parts.set(key, { mat: o.material, geos: [] });
      o.updateMatrix();
      const geo = o.geometry.index ? o.geometry.toNonIndexed() : o.geometry.clone();
      for (const name of Object.keys(geo.attributes)) if (!['position', 'normal', 'uv'].includes(name)) geo.deleteAttribute(name);
      parts.get(key).geos.push(geo.applyMatrix4(o.matrix));
    }
    const out = { g: new THREE.Group(), tails: [] };
    for (const [key, { mat, geos }] of parts) {
      const mesh = new THREE.Mesh(mergeGeometries(geos, false), mat);
      for (const x of geos) x.dispose();
      mesh.castShadow = true; mesh.receiveShadow = true;
      out.g.add(mesh);
      if (key === 'tail') out.tails.push(mesh);
    }
    return out;
  }

  // detailed cars (cars.js): parts by material key, merged into one mesh per material
  const detailMats = {
    glass: carGlass(M), interior: M.toon('#1d2027', { paint: 0.01 }),
    trim: M.toon('#2a2c33', { paint: 0.02 }), tire: mTire, hub: M.toon('#a9afb8', { paint: 0.02 }), head: mHead,
    plateW: mPlateW, plateY: mPlateY, cargo: mCargo,
  };
  const plain = (g) => {
    const x = g.index ? g.toNonIndexed() : g;
    for (const name of Object.keys(x.attributes)) if (!['position', 'normal', 'uv'].includes(name)) x.deleteAttribute(name);
    if (!x.attributes.uv) x.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(x.attributes.position.count * 2), 2));
    return x;
  };
  function makeDetail(kind, L, W, H, rgb) {
    const { parts, tails } = buildCar(kind, L, W, H, rgb);
    const byKey = new Map();
    for (const [geo, key] of parts) { if (!byKey.has(key)) byKey.set(key, []); byKey.get(key).push(plain(geo)); }
    const g = new THREE.Group(), out = { g, tails: [] };
    for (const [key, geos] of byKey) {
      const mesh = new THREE.Mesh(mergeGeometries(geos, false), key === 'body' ? colorMat(rgb, 0.03) : detailMats[key]);
      mesh.castShadow = key !== 'glass'; mesh.receiveShadow = true;
      g.add(mesh);
    }
    if (tails.length) {
      const mesh = new THREE.Mesh(mergeGeometries(tails.map(plain), false), mTail);
      mesh.castShadow = true; g.add(mesh); out.tails.push(mesh);
    }
    return out;
  }

  // ------------------------------------------------------------------ pedestrians
  function makePed(h, shirt, pants, skin, seed) {
    const g = new THREE.Group();
    const k = ctx.kit(g);
    const mShirt = colorMat(shirt, 0.03), mPants = colorMat(pants, 0.03), mSkin = colorMat(skin, 0.01);
    const mHair = M.toon(HAIR[seed % HAIR.length], { paint: 0.02 });
    const hip = 0.49 * h, shoulder = 0.8 * h;
    const legs = [], arms = [];
    for (const s of [-1, 1]) {
      const pivot = new THREE.Group(); pivot.position.set(s * 0.085, hip, 0); g.add(pivot);
      ctx.kit(pivot).rbox(0.13, hip, 0.15, 0.04, mPants, [0, -hip / 2, 0]);
      legs.push(pivot);
      const ap = new THREE.Group(); ap.position.set(s * 0.215, shoulder - 0.03, 0); g.add(ap);
      ctx.kit(ap).rbox(0.085, 0.3 * h, 0.1, 0.035, mShirt, [0, -0.15 * h, 0]);
      arms.push(ap);
    }
    k.rbox(0.36, shoulder - hip + 0.04, 0.22, 0.07, mShirt, [0, (hip + shoulder) / 2, 0]);
    k.cyl(0.05, 0.05, 0.08, mSkin, [0, shoulder + 0.04, 0], null, 8);
    k.sphere(0.115, mSkin, [0, 0.885 * h, 0.01], 14);
    k.sphere(0.12, mHair, [0, 0.9 * h, -0.025], 14);
    return { g, legs, arms };
  }

  const cars = new Map(), peds = new Map();          // KeiSim actor id -> slot
  const drop = (slot, parent) => {
    parent.remove(slot.g);
    slot.g.traverse((o) => { if (o.isMesh) o.geometry.dispose(); });
  };

  /**
   * veh: [[id, x, y, yaw, L, W, H, r, g, b, kind, brake], ...]  (KeiSim frame, colour RGB 0..255)
   * ped: [[id, x, y, yaw, height, sr, sg, sb, pr, pg, pb, kr, kg, kb, phase], ...]
   * Actors not listed this frame are hidden (callers may send only the ones near the camera).
   */
  function update(veh, ped) {
    for (const s of cars.values()) s.g.visible = false;
    for (const s of peds.values()) s.g.visible = false;
    for (const v of veh) {
      const [id, x, y, yaw, L, W, H, r, gg, b, kind, brake] = v;
      const sig = `${kind}|${L.toFixed(2)}|${W.toFixed(2)}|${H.toFixed(2)}|${q8(r)},${q8(gg)},${q8(b)}`;
      let s = cars.get(id);
      if (!s || s.sig !== sig) {
        if (s) drop(s, carRoot);
        const car = makeCar(kind, L, W, H, [r, gg, b]);
        s = { ...(ctx.mergeActors ? mergeCar(car) : car), sig, brake: -1 };
        if (ctx.carDetail) {                       // plain and detailed models side by side in one group
          const plainG = s.g, det = makeDetail(kind, L, W, H, [r, gg, b]);
          s.g = new THREE.Group(); s.g.add(plainG, det.g);
          s.plain = plainG; s.detail = det.g; s.tails = s.tails.concat(det.tails);
        }
        cars.set(id, s);
        carRoot.add(s.g);
      }
      const [X, Z] = T.P([x, y]);
      s.g.position.set(X, 0, Z);
      s.g.rotation.y = T.rotYOf(yaw);
      s.g.visible = true;
      const bk = brake ? 1 : 0;
      if (bk !== s.brake) { s.brake = bk; for (const m of s.tails) m.material = bk ? mTailOn : mTail; }
    }
    if (ctx.carDetail) view(viewMode);
    for (const p of ped) {
      const [id, x, y, yaw, h] = p;
      const sig = `${h.toFixed(2)}|${p.slice(5, 14).map(q8).join(',')}`;
      let s = peds.get(id);
      if (!s || s.sig !== sig) {
        if (s) drop(s, pedRoot);
        s = { ...makePed(h, p.slice(5, 8), p.slice(8, 11), p.slice(11, 14), id), sig };
        peds.set(id, s);
        pedRoot.add(s.g);
      }
      const [X, Z] = T.P([x, y]);
      s.g.position.set(X, T.heightAt(X, Z), Z);
      s.g.rotation.y = T.rotYOf(yaw);
      s.g.visible = true;
      const sw = Math.sin(p[14]) * 0.45;
      s.legs[0].rotation.x = sw; s.legs[1].rotation.x = -sw;
      s.arms[0].rotation.x = -sw * 0.7; s.arms[1].rotation.x = sw * 0.7;
    }
  }

  /** Which car models are shown: 'main' (detailed within DETAIL_DIST of the camera, plain beyond) or 'model' (plain
   *  everywhere: the KeiPilot frame, as in training). */
  let viewMode = 'main';
  const camPos = new THREE.Vector3();
  function view(mode) {
    viewMode = mode;
    if (!ctx.carDetail) return;
    if (ctx.camera) ctx.camera.getWorldPosition(camPos);
    for (const s of cars.values()) {
      if (!s.detail) continue;
      const near = mode === 'main' && s.g.position.distanceTo(camPos) < DETAIL_DIST;
      s.detail.visible = near; s.plain.visible = !near;
    }
  }

  /** Forget every actor (new episode). */
  function reset() {
    for (const s of cars.values()) drop(s, carRoot);
    for (const s of peds.values()) drop(s, pedRoot);
    cars.clear(); peds.clear();
  }

  ctx.add(root);
  ctx.services.actors = { update, reset, view, cars, peds };
}
