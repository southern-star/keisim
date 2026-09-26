// 信号機 — KeiSim's far-side overhead signals (Japanese horizontal 3-lamp heads on mast arms).
// Lamp states follow KeiSim's junction timing exactly (split phases: green -> yellow -> all-red),
// so for any sim time t the lit lamp matches keisim/roadnet.py Junction.signal_state().
import * as THREE from 'three';

export const TL = { RED: 0, YELLOW: 1, GREEN: 2 };

/** Same as keisim.roadnet.Junction.signal_state(phase, t). */
export function signalState(J, phase, t) {
  if (!J.signalized) return TL.GREEN;
  const per = J.green + J.yellow + J.allred, T = per * J.phases;
  const tau = (((t + J.offset) % T) + T) % T;
  const k = Math.floor(tau / per);
  if (k !== phase) return TL.RED;
  const r = tau - k * per;
  return r < J.green ? TL.GREEN : r < J.green + J.yellow ? TL.YELLOW : TL.RED;
}

// lamp slots left -> right as seen by the driver: 青 (blue-green), 黄, 赤
const SLOT = [TL.GREEN, TL.YELLOW, TL.RED];
const LIT = { [TL.GREEN]: [0.25, 1.95, 1.45], [TL.YELLOW]: [2.3, 1.45, 0.2], [TL.RED]: [2.5, 0.32, 0.22] };
const DIM = { [TL.GREEN]: [0.13, 0.22, 0.2], [TL.YELLOW]: [0.26, 0.21, 0.12], [TL.RED]: [0.26, 0.12, 0.12] };

export function build(ctx) {
  const T = ctx.town;
  const jById = new Map(T.junctions.map(j => [j.id, j]));
  const mPole = ctx.mat.toon('#b4bbc1', { paint: 0.04 });
  const mHousing = ctx.mat.toon('#9aa3aa', { paint: 0.04 });
  const mFace = ctx.mat.toon('#3d3a48', { paint: 0.02 });
  const mBase = ctx.mat.toon('#bdbcb5', { paint: 0.05 });
  const root = new THREE.Group(); root.name = 'signals';
  const K = ctx.kit(root);
  const lamps = [];                                    // {signal, junction, slot, matrix}
  const up = new THREE.Vector3(0, 1, 0);

  for (const s of T.signals) {
    const J = jById.get(s.junction); if (!J) continue;
    const [px, pz] = s.poleP, [hx, hz] = s.headP;
    const g0 = ctx.L.heightAt(px, pz);
    const armY = s.z + 0.55, headY = s.z;
    // mast + footing
    K.cyl(0.26, 0.26, 0.12, mBase, [px, g0 + 0.06, pz], null, 12);
    K.cyl(0.095, 0.12, armY + 0.35 - g0, mPole, [px, g0 + (armY + 0.35 - g0) / 2, pz], null, 10);
    // mast arm: from the pole to just past the head, with a slight upward sweep
    const dx = hx - px, dz = hz - pz, len = Math.hypot(dx, dz);
    if (len > 0.3) {
      const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.075, len + 0.3, 8), mPole);
      arm.position.set((px + hx) / 2 + dx / len * 0.15, armY, (pz + hz) / 2 + dz / len * 0.15);
      arm.quaternion.setFromUnitVectors(up, new THREE.Vector3(dx / len, 0, dz / len));
      arm.castShadow = true; arm.receiveShadow = true; root.add(arm);
      K.box(0.06, armY - headY - 0.2, 0.06, mPole, [hx, (armY + headY + 0.2) / 2, hz]);   // hanger
    }
    // head: housing, face plate, three visors; local +Z faces the approaching traffic
    const head = K.group([hx, headY, hz], s.rotY);
    const k = ctx.kit(head);
    k.rbox(1.28, 0.46, 0.2, 0.06, mHousing, [0, 0, 0]);
    k.box(1.2, 0.38, 0.02, mFace, [0, 0, 0.105]);
    for (let i = 0; i < 3; i++) {
      const x = (i - 1) * 0.4;
      k.box(0.34, 0.03, 0.22, mHousing, [x, 0.17, 0.2], [0.22, 0, 0]);             // visor (庇)
      for (const sx of [-1, 1]) k.box(0.02, 0.2, 0.18, mHousing, [x + sx * 0.165, 0.08, 0.19]);
      lamps.push({ J, phase: s.phase, slot: SLOT[i], local: new THREE.Vector3(x, 0, 0.118), head });
    }
  }
  ctx.addStatic(root);

  // lamps: one instanced mesh, colours updated from the junction timing
  root.updateMatrixWorld(true);
  const disc = new THREE.CircleGeometry(0.15, 20);
  const mLamp = new THREE.MeshBasicMaterial({ color: '#ffffff', toneMapped: false });
  const inst = new THREE.InstancedMesh(disc, mLamp, Math.max(1, lamps.length));
  inst.name = 'signal-lamps';
  const m4 = new THREE.Matrix4(), col = new THREE.Color();
  lamps.forEach((l, i) => {
    const world = l.head.localToWorld(l.local.clone());
    m4.makeRotationY(l.head.rotation.y).setPosition(world);
    inst.setMatrixAt(i, m4);
    inst.setColorAt(i, col.setRGB(...DIM[l.slot]));
  });
  inst.count = lamps.length;
  inst.castShadow = false; inst.receiveShadow = false;
  ctx.noOutline(inst);
  ctx.add(inst);
  const last = new Int8Array(lamps.length).fill(-1);
  ctx.onUpdate((dt, t) => {
    let dirty = false;
    for (let i = 0; i < lamps.length; i++) {
      const l = lamps[i];
      const on = signalState(l.J, l.phase, t) === l.slot ? 1 : 0;
      if (on !== last[i]) { last[i] = on; inst.setColorAt(i, col.setRGB(...(on ? LIT : DIM)[l.slot])); dirty = true; }
    }
    if (dirty && inst.instanceColor) inst.instanceColor.needsUpdate = true;
  });
  ctx.services.signals = { count: T.signals.length, lamps: lamps.length };
}
