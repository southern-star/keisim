// Checks of the detailed pedestrians (src/world/people.js) in node, without a browser — PedRedesign_Handoff.md
// chapter 6, the part of stages P0 to P2:
//   2  the same inputs give the same person twice (type, vertex counts, colours)
//   4  feet do not slide: through one cycle of the phase, with the body moved 1.496 m, the toes of a foot on the
//      ground stay within 3 cm of where they landed
//   5  variety: ids 0 .. 199 with KeiSim's heights and colours give every type a grown-up can be, children give
//      the seventh, and at most 5 % of the people look like another one; the kinds of parts of 4.4 are all there
//   size: at most 2,500 triangles and 16 meshes per person; every mesh is semantic class 12 (pedestrian)
// and what the walk and the standing pose must look like: the knee bends in the swing and is nearly straight under
// the body, the soles are on the ground, the standing person is as tall as KeiSim says, skirts and coats hold the legs.
// Stage P2 (ways of walking): no sliding for every type, the habits differ by type as they should, a walking stick
// stands where it is planted and stays in the hand, a thing in a hand keeps clear of the leg.
// Stage P3 (the wiring, with a browser: ego mode's page with ?peddetail=1):
//   3  the labels: every pixel that the pedestrians change is class 12, where the picture changes too (the label
//      pass must draw the people as posed, skirts too), with things carried and thin parts
//      the switch: detailed within PED_DETAIL_DIST of the camera, plain beyond and everywhere in view('model');
//      the two models of a person are equally tall in the picture; standing after frames with the same phase;
//      nothing is left behind when the people are dropped (bone textures)
//   6  the speed: tools/people_bench.mjs (a measurement, not judged here)
//   node tools/people_check.mjs [--n 200] [--no-browser] [--gl hw|soft]
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
globalThis.window = globalThis; globalThis.self = globalThis;
globalThis.document = { createElement: () => ({ style: {}, getContext: () => null }), createElementNS: () => ({ style: {} }) };
const THREE = await import('three');
const { makePerson, STEP, personLook, personHabit, HABITS, PERSON_TYPES, TYPE_BANDS, KID_BELOW } = await import(pathToFileURL(path.join(root, 'src/world/people.js')).href);
const argv = process.argv.slice(2), N = Number(argv[argv.indexOf('--n') + 1] && argv.includes('--n') ? argv[argv.indexOf('--n') + 1] : 200);

function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
// keisim/traffic.py Pedestrians: SHIRTS x uniform(0.85, 1.1), PANTS, SKIN (there BGR), heights uniform(1.5, 1.85);
// the children that run out (keisim/world.py, parked.py): heights uniform(1.0, 1.35), KID_SHIRTS, PANTS[0], SKIN[1].
// Heights rounded to 1 cm (keisim/render/keiview.py), colours quantised as actors.js does.
const SHIRTS = [[40, 60, 200], [200, 40, 40], [40, 160, 40], [230, 230, 230], [30, 30, 30], [230, 140, 0], [150, 50, 150], [220, 200, 60], [120, 120, 120], [80, 150, 200]];
const PANTS = [[25, 30, 40], [30, 60, 90], [30, 30, 30], [100, 110, 120], [110, 50, 60], [150, 150, 150]];
const SKIN = [[220, 180, 150], [205, 160, 120], [170, 125, 90], [130, 90, 60]];
const KID_SHIRTS = [[230, 60, 40], [30, 120, 230], [240, 200, 40], [200, 60, 200], [90, 200, 60], [240, 240, 240]];
const q8 = (v) => Math.round(Math.min(255, Math.max(0, v)) / 16) * 16;
const color = (c) => new THREE.Color().setRGB(q8(c[0]) / 255, q8(c[1]) / 255, q8(c[2]) / 255, THREE.SRGBColorSpace);
function sample(r, kid = false) {
  if (kid) return { h: +(1.0 + 0.35 * r()).toFixed(2), cols: [color(KID_SHIRTS[Math.floor(r() * KID_SHIRTS.length)]), color(PANTS[0]), color(SKIN[1])] };
  const h = +(1.5 + 0.35 * r()).toFixed(2), k = 0.85 + 0.25 * r();
  return { h, cols: [color(SHIRTS[Math.floor(r() * SHIRTS.length)].map((v) => v * k)), color(PANTS[Math.floor(r() * PANTS.length)]), color(SKIN[Math.floor(r() * SKIN.length)])] };
}
const mat = new THREE.MeshBasicMaterial({ vertexColors: true });
const ctx = { noOutline: (o) => o };
const meshes = (g) => { const out = []; g.traverse((o) => { if (o.isMesh) out.push(o); }); return out; };
const tris = (g) => meshes(g).reduce((n, m) => n + (m.geometry.index ? m.geometry.index.count : m.geometry.attributes.position.count) / 3, 0);
/** Counts and a checksum of every mesh's positions and colours. */
function signature(g) {
  let sum = 0;
  const parts = meshes(g).map((m) => {
    for (const name of ['position', 'color']) { const a = m.geometry.attributes[name].array; for (let i = 0; i < a.length; i++) sum = (sum * 31 + Math.round(a[i] * 4096)) | 0; }
    return `${m.name}:${m.geometry.attributes.position.count}`;
  });
  return `${parts.join(',')}|${sum}`;
}
const semOf = (m) => { for (let o = m; o; o = o.parent) if (o.userData.sem !== undefined) return o.userData.sem; return undefined; };

let failed = false;
const line = (ok, what, detail) => { if (!ok) failed = true; console.log(`${ok ? 'ok  ' : 'FAIL'} ${what.padEnd(26)} ${detail}`); };
const r = rng(12), people = [];
for (let id = 0; id < N; id++) people.push({ id, ...sample(r, id % 10 === 9) });        // every tenth a child (1.00 .. 1.35 m)

// ---- 2: the same inputs twice
{
  let same = 0; const looks = new Set();
  for (const p of people) {
    const A = makePerson(ctx, mat, p.h, p.cols, p.id), B = makePerson(ctx, mat, p.h, p.cols.map((c) => c.clone()), p.id);
    const a = `${A.type}|${A.look.key}|${signature(A.g)}`, b = `${B.type}|${B.look.key}|${signature(B.g)}`;
    if (a === b) same++;
    looks.add(a);
  }
  line(same === N, '2 same inputs, same person', `${same}/${N} identical (type, parts, meshes, vertex counts, positions, colours); ${looks.size} different people`);
}

// ---- size and class
{
  // every type, with more people than the 200 of the other checks: the largest person of each
  const r2 = rng(77), by = new Map(PERSON_TYPES.map((t) => [t, { n: 0, t: 0, m: 0 }]));
  let noSem = 0, bodies = 0, nFine = 0, lost = 0, thin = 0, noThin = 0;
  const NS = Math.max(N, 1200);
  for (let id = 0; id < NS; id++) {
    const p = id < N ? people[id] : { id, ...sample(r2, id % 6 === 5) };
    const P = makePerson(ctx, mat, p.h, p.cols, p.id), all = meshes(P.g), b = by.get(P.type);
    b.n++; b.t = Math.max(b.t, tris(P.g)); b.m = Math.max(b.m, all.length);
    for (const m of all) if (semOf(m) !== 12) noSem++;
    bodies = 3 + 2 + 2 * 5;
    const B = P.bodies; if (!(B.pelvis && B.torso && B.head && B.item && B.hem && B.sides.length === 2)) noSem++;
    nFine += P.fine.length; for (const m of P.fine) if (!all.includes(m)) lost++;
    const wantThin = P.look.cord || (P.look.hand && (P.look.hand.kind === 'cane' || P.look.hand.kind === 'umbrella'));
    if (wantThin) { thin++; if (!P.fine.length) noThin++; }
  }
  const maxT = Math.max(...[...by.values()].map((b) => b.t)), maxM = Math.max(...[...by.values()].map((b) => b.m));
  line(maxT <= 2500 && maxM <= 16, 'size', `at most ${maxT} triangles (limit 2500) and ${maxM} meshes (limit 16) in ${bodies} rigid bodies, over ${NS} people`);
  console.log(`     by type (triangles / meshes): ${[...by].map(([t, b]) => `${t} ${b.t} / ${b.m}`).join(', ')}`);
  line(noSem === 0, 'class', `${noSem} meshes outside semantic class 12 (things carried and thin parts too)`);
  line(lost === 0 && noThin === 0 && thin > 0, 'thin parts', `${thin} people with a stick, an umbrella or a cord: all have them in 'fine' (${nFine} meshes, without outlines)`);
}

// ---- 4: feet do not slide; the shape of the walk
const v = new THREE.Vector3();
function walk(p, steps = 720) {
  const P = makePerson(ctx, mat, p.h, p.cols, p.id), f = P.fit;
  const shoes = ['shoeL', 'shoeR'].map((n) => meshes(P.g).find((m) => m.name === n));
  const bone = (n) => meshes(P.g).find((m) => m.name === n).parent;
  const out = { drift: 0, sink: 0, float: 0, ground: [0, 0], kneeSwing: 0, kneeStand: Math.PI, air: 0, fit: f, gait: P.gait, elbow: [Math.PI, 0], look: P.look, hem: 0,
    type: P.type, habit: P.habit, hipY: [Infinity, -Infinity], armA: [Infinity, -Infinity], lean: 0, roll: 0, over: 0, n: 0,
    caneDrift: 0, caneGap: 0, caneDown: 0, clear: Infinity };
  const B = P.bodies, kind = P.look.hand && P.look.hand.kind, cane = kind === 'cane', carried = kind && !cane;
  const freeSide = B.sides.find((sd) => sd.mode === 'swing'), handSide = kind ? B.sides.find((sd) => sd.sgn === P.look.hand.side) : null;
  const itemMesh = carried ? B.item.children.find((m) => m.name === 'item') : null;
  let caneLand = null; const airMax = [-Infinity, -Infinity], wasDown = [true, true];
  const w1 = new THREE.Vector3(), w2 = new THREE.Vector3(), w3 = new THREE.Vector3(), w4 = new THREE.Vector3();
  const hemMesh = P.bodies.hem.children[0];
  const land = [null, null];
  const d1 = new THREE.Vector3(), d2 = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
  for (let k = 0; k <= steps * 2; k++) {                               // two cycles: every stance is seen whole
    const phase = k / steps * 2 * Math.PI;
    P.pose({ phase, moving: true, dt: 0.01 });
    P.g.position.set(0, 0, phase * STEP); P.g.updateMatrixWorld(true);
    let onGround = 0;
    shoes.forEach((m, i) => {
      const heel = v.set(0, -f.ankle, -f.heel).applyMatrix4(m.matrixWorld).y, ballY = v.set(0, -f.ankle, f.ball).applyMatrix4(m.matrixWorld).y;
      const toe = v.set(0, -f.ankle, f.toe).applyMatrix4(m.matrixWorld), low = Math.min(heel, ballY);
      out.sink = Math.max(out.sink, -Math.min(low, toe.y));
      if (low < 0.001) {
        onGround++; out.ground[i]++;
        if (land[i] === null) land[i] = toe.z;
        out.drift = Math.max(out.drift, Math.abs(toe.z - land[i]));
      } else land[i] = null;
      // the time the gait says this foot stands (the left foot's is centred on phase pi): how far the sole is off the ground
      const ph = phase + (i ? Math.PI : 0) - Math.PI, pw = ph - 2 * Math.PI * Math.floor((ph + Math.PI) / (2 * Math.PI));
      if (Math.abs(pw) < P.gait.D * Math.PI - 0.02) out.float = Math.max(out.float, low);
      // the knee's bend (0: straight), in the swing and under the body
      const n = i ? 'R' : 'L';
      d1.copy(up).applyQuaternion(bone('thigh' + n).quaternion); d2.copy(up).applyQuaternion(bone('shin' + n).quaternion);
      const knee = d1.angleTo(d2);
      if (low < 0.001) { const sh = v.set(0, 0, 0).applyMatrix4(m.matrixWorld).z - P.g.position.z; if (Math.abs(sh) < 0.03) out.kneeStand = Math.min(out.kneeStand, knee); }
      else out.kneeSwing = Math.max(out.kneeSwing, knee);
      d1.copy(up).applyQuaternion(bone('armU' + n).quaternion); d2.copy(up).applyQuaternion(bone('armF' + n).quaternion);
      const e = d1.angleTo(d2); out.elbow[0] = Math.min(out.elbow[0], e); out.elbow[1] = Math.max(out.elbow[1], e);
    });
    if (!onGround) out.air++;
    // the habits as they show: the hips' height, the free hand's way, the trunk's lean and roll, the swinging foot's reach
    out.n++; out.hipY[0] = Math.min(out.hipY[0], B.pelvis.position.y); out.hipY[1] = Math.max(out.hipY[1], B.pelvis.position.y);
    out.lean += B.torso.rotation.x; out.roll += B.torso.rotation.z;
    if (freeSide) { w1.copy(up).applyQuaternion(freeSide.armU.quaternion); const a = Math.atan2(w1.z, -w1.y); out.armA[0] = Math.min(out.armA[0], a); out.armA[1] = Math.max(out.armA[1], a); }
    shoes.forEach((m, i) => {                                          // how far the swinging foot goes ahead of where it then lands
      const z = w1.set(0, 0, 0).applyMatrix4(m.matrixWorld).z - P.g.position.z, down = Math.min(w2.set(0, -f.ankle, -f.heel).applyMatrix4(m.matrixWorld).y, w3.set(0, -f.ankle, f.ball).applyMatrix4(m.matrixWorld).y) < 0.001;
      if (!down) airMax[i] = Math.max(airMax[i], z); else if (!wasDown[i] && k > steps / 2) out.over = Math.max(out.over, airMax[i] - z);
      if (down) airMax[i] = -Infinity; wasDown[i] = down;
    });
    if (cane) {                                                        // the stick: its tip on the ground stays there; its handle stays in the hand
      const tipW = w1.set(0, P.caneLen, 0).applyMatrix4(B.item.matrixWorld), handle = w2.set(0, 0, 0).applyMatrix4(B.item.matrixWorld);
      const gripW = w3.set(0, f.fore + 0.025 * f.s, 0).applyMatrix4(handSide.armF.matrixWorld);
      out.caneGap = Math.max(out.caneGap, handle.distanceTo(gripW));
      if (tipW.y < 0.001) { out.caneDown++; if (caneLand === null) caneLand = tipW.clone(); out.caneDrift = Math.max(out.caneDrift, Math.hypot(tipW.x - caneLand.x, tipW.z - caneLand.z)); } else caneLand = null;
    }
    if (itemMesh && k % 8 === 0) {                                     // a bag in the hand against the leg of its side (thigh and shin as tapered rods)
      const pos = itemMesh.geometry.attributes.position, lo = P.look.bottom === 'long' ? P.look.loose : 1;
      const rods = [[handSide.thigh, 0.075 * lo, 0.056 * lo], [handSide.shin, 0.054 * lo, 0.04 * lo]].map(([b, r0, r1]) => [w2.set(0, 0, 0).applyMatrix4(b.matrixWorld).clone(), w3.set(0, f.leg, 0).applyMatrix4(b.matrixWorld).clone(), r0 * f.s, r1 * f.s]);
      for (let i = 0; i < pos.count; i += 2) {
        w1.fromBufferAttribute(pos, i).applyMatrix4(itemMesh.matrixWorld);
        for (const [a, b, r0, r1] of rods) {
          const ab = w4.copy(b).sub(a), t = Math.min(1, Math.max(0, w2.copy(w1).sub(a).dot(ab) / ab.lengthSq()));
          out.clear = Math.min(out.clear, w3.copy(a).addScaledVector(ab, t).distanceTo(w1) - (r0 + (r1 - r0) * t));
        }
      }
    }
    if (hemMesh && k % 30 === 0) out.hem = Math.max(out.hem, outOfHem(P, hemMesh));
  }
  out.duty = out.ground.map((n) => n / (steps * 2 + 1)); out.air /= steps * 2 + 1;
  out.lean /= out.n; out.roll /= out.n; out.caneDown /= out.n;
  // standing: feet flat and together, the top of the head at the person's height
  P.pose({ phase: 1.0, stand: true }); P.g.position.set(0, 0, 0); P.g.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(P.g, true);            // (from the vertices)
  out.tall = box.max.y; out.low = box.min.y;
  out.feetApart = Math.abs(v.set(0, 0, 0).applyMatrix4(shoes[0].matrixWorld).z - new THREE.Vector3().applyMatrix4(shoes[1].matrixWorld).z);
  return out;
}
/** How far (m) the front or back of a thigh or shin is outside the hem of a skirt or coat, for the part of the leg the hem is long enough to cover. */
const _tri = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()], _ray = new THREE.Ray(), _hit = new THREE.Vector3(), _inv = new THREE.Matrix4(), _q = new THREE.Vector3();
function outOfHem(P, hemMesh) {
  const pos = hemMesh.geometry.attributes.position, s = P.fit.s;
  let yMin = Infinity, w = 0;
  for (let i = 0; i < pos.count; i++) yMin = Math.min(yMin, pos.getY(i));
  _inv.copy(hemMesh.matrixWorld).invert();
  const deep = _q.setFromMatrixColumn(hemMesh.matrixWorld, 2).length();
  const at = [];                                                                // the vertices where the hem's bones put them (its own frame)
  for (let i = 0; i < pos.count; i++) at.push(hemMesh.isSkinnedMesh ? hemMesh.getVertexPosition(i, new THREE.Vector3()) : new THREE.Vector3().fromBufferAttribute(pos, i));
  const first = (dir) => {
    _ray.direction.set(0, 0, dir);
    let n = 0, d = Infinity;
    for (let i = 0; i < pos.count; i += 3) {
      if (_ray.intersectTriangle(at[i], at[i + 1], at[i + 2], false, _hit)) { n++; d = Math.min(d, _hit.distanceTo(_ray.origin)); }
    }
    return [n, d];
  };
  const hemLen = -yMin - 0.02 * s;                                              // (the hem starts 0.02 s under the hip joints)
  for (const sd of P.bodies.sides) for (const [b, r, d0] of [[sd.thigh, 0.065, 0], [sd.shin, 0.045, P.fit.leg]]) for (let f = 0.05; f <= 1; f += 0.05) for (const dir of [1, -1]) {
    if (d0 + f * P.fit.leg > hemLen - 0.02) continue;                             // further along the leg than the hem is long: a raised knee comes out
    _ray.origin.set(0, f * P.fit.leg, dir * r * s).applyMatrix4(b.matrixWorld).applyMatrix4(_inv);
    if (_ray.origin.y < yMin + 0.01 || _ray.origin.y > -0.14 * s) continue;      // below the hem, or inside the hips
    if (first(dir)[0] % 2 === 1) continue;                                     // inside
    const back = first(-dir)[1];
    if (back < Infinity) w = Math.max(w, back * deep);                           // (beside the hem: not judged)
  }
  return w;
}
const deg = (a) => (a * 180 / Math.PI).toFixed(0);
{
  const adults = people.filter((p) => p.h >= 1.5).slice(0, 40), kids = people.filter((p) => p.h < 1.5).slice(0, 12);
  let hems = 0, hemOut = 0;
  for (const [name, group] of [['adults 1.50-1.85 m', adults], ['children 1.00-1.35 m', kids]]) {
    const W = group.map((p) => walk(p)), mx = (f) => Math.max(...W.map(f)), mn = (f) => Math.min(...W.map(f));
    line(mx((o) => o.drift) <= 0.03, `no sliding, ${name.split(' ')[0]}`,
      `${name}: toes of a foot on the ground move at most ${(mx((o) => o.drift) * 100).toFixed(1)} cm (limit 3); on the ground ${(mn((o) => Math.min(...o.duty)) * 100).toFixed(0)}-${(mx((o) => Math.max(...o.duty)) * 100).toFixed(0)} % of the cycle, both feet in the air ${(mx((o) => o.air) * 100).toFixed(0)} %`);
    line(mx((o) => o.sink) <= 0.03 && mx((o) => o.float) <= 0.002, `soles, ${name.split(' ')[0]}`,
      `at most ${(mx((o) => o.sink) * 100).toFixed(1)} cm below the ground (the toes at toe-off), a standing foot at most ${(mx((o) => o.float) * 1000).toFixed(1)} mm above it`);
    const running = mx((o) => o.gait.run) > 0.5;                       // short legs run: the knee stays bent under the body
    const old = W.filter((o) => o.type === 'elder'), young = W.filter((o) => o.type !== 'elder');
    line(mn((o) => o.kneeSwing) > 0.6 && Math.max(...young.map((o) => o.kneeStand)) < (running ? 1.2 : 0.5) && Math.max(0, ...old.map((o) => o.kneeStand)) < 0.66, `knees, ${name.split(' ')[0]}`,
      `bend ${deg(mn((o) => o.kneeSwing))}-${deg(mx((o) => o.kneeSwing))} deg in the swing, ${deg(mn((o) => o.kneeStand))}-${deg(mx((o) => o.kneeStand))} deg under the body; elbows ${deg(mn((o) => o.elbow[0]))}-${deg(mx((o) => o.elbow[1]))} deg`);
    // as tall as KeiSim says to the top of the hair; a hat, helmet, hood, bun or headphones may add up to 6 cm
    const onTop = (o) => o.look.hat || o.look.hair === 'bun' || o.look.phones, bare = W.filter((o) => !onTop(o)), hat = W.filter(onTop);
    const dh = Math.max(0, ...bare.map((o) => Math.abs(o.tall - o.fit.h))), dHat = hat.map((o) => o.tall - o.fit.h);
    line(dh <= 0.02 && dHat.every((d) => d > -0.02 && d <= 0.06) && mx((o) => Math.abs(o.low)) <= 0.004 && mx((o) => o.feetApart) < 0.01, `standing, ${name.split(' ')[0]}`,
      `height off by at most ${(dh * 100).toFixed(1)} cm (${hat.length} with a hat, bun or headphones: ${hat.length ? (Math.min(...dHat) * 100).toFixed(1) + ' .. ' + (Math.max(...dHat) * 100).toFixed(1) : '-'} cm), soles ${(mx((o) => Math.abs(o.low)) * 1000).toFixed(1)} mm from the ground, feet side by side`);
    const heads = W.map((o) => o.fit.h / o.fit.headH), kid = name.startsWith('children');
    line(Math.min(...heads) >= (kid ? 5 : 7) - 0.01 && Math.max(...heads) <= (kid ? 6 : 7.5) + 0.01, `heads tall, ${name.split(' ')[0]}`,
      `${Math.min(...heads).toFixed(2)} .. ${Math.max(...heads).toFixed(2)} heads (wanted ${kid ? '5 .. 6' : '7 .. 7.5'})`);
    for (const o of W) if (o.look.top === 'coat' || o.look.bottom === 'skirt' || o.look.bottom === 'longskirt') { hems++; hemOut = Math.max(hemOut, o.hem); }
  }
  line(hems > 0 && hemOut <= 0.03, 'skirts and coats', `${hems} people with a hem: the legs come out of it by at most ${(hemOut * 100).toFixed(1)} cm (limit 3)`);
}
// ---- stage P2: every type's way of walking
{
  const r7 = rng(4242), by = new Map(PERSON_TYPES.map((t) => [t, []])), canes = [], bags = [];
  for (let id = 0; id < 6000 && ([...by.values()].some((a) => a.length < 6) || canes.length < 6 || bags.length < 14); id++) {
    const p = { id, ...sample(r7, id % 6 === 5) }, L = personLook(p.h, p.cols, p.id), kind = L.hand && L.hand.kind;
    if (by.get(L.type).length < 6) by.get(L.type).push(p);
    else if (kind === 'cane' && canes.length < 6) canes.push(p);
    else if (kind && kind !== 'cane' && bags.length < 14) bags.push(p);
  }
  const W = new Map([...by].map(([t, a]) => [t, a.map((p) => walk(p, 360))])), mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const M = new Map([...W].map(([t, a]) => [t, {
    drift: Math.max(...a.map((o) => o.drift)), bob: mean(a.map((o) => (o.hipY[1] - o.hipY[0]) / o.fit.h)), arm: mean(a.map((o) => o.armA[1] - o.armA[0]).filter(Number.isFinite)),
    lean: mean(a.map((o) => o.lean)), reach: mean(a.map((o) => o.over / o.fit.h)), roll: Math.max(...a.map((o) => Math.abs(o.roll))), knee: Math.max(...a.map((o) => o.kneeStand)) }]));
  const worst = Math.max(...[...M.values()].map((m) => m.drift));
  line(worst <= 0.03, '4 no sliding, by type', [...M].map(([t, m]) => `${t} ${(m.drift * 100).toFixed(1)}`).join(', ') + ' cm (limit 3), 6 people each');
  console.log('     habits as they show   type      hips up-down  free arm swing  trunk lean  foot past landing  shoulder drop  (factors: bob arm lean tw lift reach drop)');
  for (const [t, m] of M) {
    const b = HABITS[t];
    console.log(`                           ${t.padEnd(9)} ${(m.bob * 100).toFixed(2).padStart(6)} % h     ${deg(m.arm).padStart(6)} deg     ${(m.lean * 180 / Math.PI).toFixed(1).padStart(5)} deg   ${(m.reach * 100).toFixed(2).padStart(8)} % h      ${(m.roll * 180 / Math.PI).toFixed(1).padStart(5)} deg      (${b.bob} ${b.arm} ${b.lean} ${b.tw} ${b.lift} ${b.reach} ${b.drop})`);
  }
  const g = (t) => M.get(t), st = g('student');
  line(g('kid').arm > 1.15 * st.arm && g('elder').arm < 0.7 * st.arm && g('elder').bob < 0.8 * st.bob && g('elder').lean > st.lean + 0.06 && g('office').lean < st.lean - 0.01
    && g('office').reach > st.reach + 0.005 && g('shopper').roll > st.roll + 0.02, 'habits differ by type',
    'children swing their arms more than students, old people less, with less bob and a stoop; office workers walk upright and swing the foot out further; shoppers drop a shoulder');
  // each person keeps the habits the hash gives them, within 10 % of the type's
  let off = 0, n = 0; const spread = new Set();
  for (const [t, a] of by) for (const p of a) { const h1 = personHabit(p.h, p.cols, p.id), h2 = personHabit(p.h, p.cols, p.id), b = HABITS[t]; n++; spread.add(h1.arm.toFixed(4));
    if (JSON.stringify(h1) !== JSON.stringify(h2) || Math.abs(h1.arm / b.arm - 1) > 0.1001 || Math.abs(h1.bob / b.bob - 1) > 0.1001 || Math.abs(h1.tw / b.tw - 1) > 0.1001) off++; }
  line(off === 0 && spread.size > n / 2, 'habits of their own', `${n} people: the same twice, bob / arm / turn within 10 % of their type's; ${spread.size} different arm swings`);
  // walking sticks
  const C = canes.map((p) => walk(p, 720));
  line(C.length >= 4 && Math.max(...C.map((o) => o.caneDrift)) <= 0.03 && Math.max(...C.map((o) => o.caneGap)) <= 0.015 && Math.min(...C.map((o) => o.caneDown)) > 0.3 && Math.max(...C.map((o) => o.drift)) <= 0.03, 'walking sticks',
    `${C.length} people with a stick: its tip on the ground moves at most ${(Math.max(...C.map((o) => o.caneDrift)) * 100).toFixed(1)} cm (limit 3), on the ground ${(Math.min(...C.map((o) => o.caneDown)) * 100).toFixed(0)}-${(Math.max(...C.map((o) => o.caneDown)) * 100).toFixed(0)} % of the cycle; handle at most ${(Math.max(...C.map((o) => o.caneGap)) * 1000).toFixed(1)} mm from the hand's grip (limit 15)`);
  // bags in a hand
  const G = bags.map((p) => walk(p, 360)), kinds = [...new Set(G.map((o) => o.look.hand.kind))].sort();
  const tight = G.reduce((a, o) => (o.clear < a.clear ? o : a));
  line(G.length >= 10 && tight.clear >= -0.005, 'things in a hand', `${G.length} people (${kinds.join(', ')}): the thing comes no nearer than ${(tight.clear * 100).toFixed(1)} cm to the leg of its side (${tight.look.hand.kind}; limit -0.5: half a centimetre into it)`);
}
// ---- 5: variety. ids 0 .. 199 as KeiSim makes its pedestrians, then children as it makes the ones that run out
{
  const r5 = rng(2026), grown = [], kids = [];
  for (let id = 0; id < 200; id++) grown.push({ id, ...sample(r5) });
  for (let id = 200; id < 240; id++) kids.push({ id, ...sample(r5, true) });
  const look = (p) => personLook(p.h, p.cols, p.id);
  const count = new Map(PERSON_TYPES.map((t) => [t, 0])), keys = new Map();
  let offBand = 0;
  for (const p of grown) {
    const L = look(p);
    count.set(L.type, count.get(L.type) + 1); keys.set(L.key, (keys.get(L.key) || 0) + 1);
    const band = TYPE_BANDS.find((b) => b[0] === L.type);
    if (!band || p.h < band[1] - 1e-9 || p.h > band[2] + 1e-9) offBand++;
  }
  const kidTypes = new Set(kids.map((p) => look(p).type)), kidKeys = new Set(kids.map((p) => look(p).key));
  const missing = TYPE_BANDS.map((b) => b[0]).filter((t) => !count.get(t));
  const twins = [...keys.values()].filter((n) => n > 1).reduce((a, n) => a + n, 0);
  line(missing.length === 0 && offBand === 0 && count.get('kid') === 0, '5 types', `ids 0-199: ${[...count].filter(([, n]) => n).map(([t, n]) => `${t} ${n}`).join(', ')}${missing.length ? '; missing: ' + missing.join(', ') : ''}; ${offBand} outside their type's heights`);
  line(kidTypes.size === 1 && kidTypes.has('kid'), '5 children', `40 people of 1.00-1.35 m: all ${[...kidTypes].join(', ')} (below ${KID_BELOW} m), ${kidKeys.size} different looks`);
  line(twins <= 0.05 * grown.length, '5 look-alikes', `${twins} of 200 people share every part and colour with another (limit ${0.05 * grown.length})`);
  // the kinds of parts (4.4: hair and hats 8, tops 6, bottoms 4, shoes 3, things carried 8), over many people
  const kinds = { head: new Set(), top: new Set(), bottom: new Set(), shoes: new Set(), things: new Set() };
  for (let id = 0; id < 4000; id++) {
    const L = look({ id, ...sample(r5, id % 6 === 5) });
    kinds.head.add(L.headgear); kinds.top.add(L.top); kinds.bottom.add(L.bottom); kinds.shoes.add(L.shoes); for (const t of L.things) kinds.things.add(t);
  }
  const need = { head: 8, top: 6, bottom: 4, shoes: 3, things: 8 };
  line(Object.keys(need).every((k) => kinds[k].size >= need[k]), '5 parts', Object.keys(need).map((k) => `${k} ${kinds[k].size} (${[...kinds[k]].sort().join(' ')})`).join('; '));
}
// ---- standing settles in about 0.3 s when the phase stops, and only with a call before it
{
  const p = people.find((q) => q.h >= 1.5), P = makePerson(ctx, mat, p.h, p.cols, p.id);
  const shoe = meshes(P.g).find((m) => m.name === 'shoeL');
  const footZ = () => { P.g.updateMatrixWorld(true); return v.set(0, 0, 0).applyMatrix4(shoe.matrixWorld).z; };
  P.pose({ phase: 1.5 }); const z0 = footZ();
  const Q = makePerson(ctx, mat, p.h, p.cols, p.id); Q.pose({ phase: 1.5, stand: false }); Q.g.updateMatrixWorld(true);
  const zWalk = v.set(0, 0, 0).applyMatrix4(meshes(Q.g).find((m) => m.name === 'shoeL').matrixWorld).z;
  let calls = 0; const zStand = -0.012 * p.h;
  while (calls < 20 && Math.abs(footZ() - zStand) > 1e-4) { P.pose({ phase: 1.5 }); calls++; }       // same phase, 0.1 s a call
  line(Math.abs(z0 - zWalk) < 1e-9 && calls >= 2 && calls <= 4, 'standing still',
    `first call: the walking pose; the phase unchanged for ${calls} calls of 0.1 s: standing`);
}
// ---------------------------------------------------------------- stage P3: in the browser
if (!argv.includes('--no-browser')) {
  const { launch, serveTree } = await import('./golden.mjs');
  const { findRoad, street } = await import('./people_scene.mjs');
  const { PED_DETAIL_DIST } = await import(pathToFileURL(path.join(root, 'src/world/people.js')).href);
  const W = 640, H = 320, gl = argv.includes('--gl') ? argv[argv.indexOf('--gl') + 1] : 'hw';
  const browser = await launch(gl, W, H), server = serveTree(root);
  await new Promise((res) => server.listen(0, '127.0.0.1', res));
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/index.html?shot=1&ego=1&town=1000&w=${W}&h=${H}&q=medium&peddetail=1`, { waitUntil: 'load', timeout: 180000 });
    await page.waitForFunction('window.__ready === true', { timeout: 900000, polling: 100 });
    const road = await page.evaluate(findRoad), S = street(road, { near: 20, far: 4 });
    const all = S.ped.concat(S.far);
    const frame = async (ped, t, extra = {}) => {
      const o = await page.evaluate((q) => { const r = window.__ego(q); return { rgb: r.rgb, sem: r.sem, errors: window.__errors.length }; },
        { cam: S.cam, t, veh: [], ped, bike: [], labels: true, ...extra });
      return { rgb: Buffer.from(o.rgb, 'base64'), sem: Buffer.from(o.sem, 'base64'), errors: o.errors };
    };
    /** What is shown of every pedestrian, with its distance from the camera. */
    const shown = () => page.evaluate(() => {
      const A = window.__ctx.services.actors, c = new window.THREE.Vector3(); window.__ctx.camera.getWorldPosition(c);
      return [...A.peds].filter(([, s]) => s.g.visible).map(([id, s]) => ({ id, d: s.g.position.distanceTo(c), detail: !!s.detail && s.detail.g.visible, plain: s.plain ? s.plain.visible : true,
        fine: s.detail ? s.detail.fine.filter((m) => m.visible).length : 0, fineAll: s.detail ? s.detail.fine.length : 0,
        hems: (() => { let n = 0; if (s.detail) s.detail.g.traverse((o) => { if (o.name === 'hem') n++; }); return n; })(),
        skinned: (() => { let n = 0; if (s.detail) s.detail.g.traverse((o) => { if (o.isSkinnedMesh) n++; }); return n; })() }));
    });

    // --- the switch by distance, and the labels of a frame with every kind of person
    const A = await frame(all, 5, { reset: true }), vis = await shown();
    const nearOk = vis.filter((p) => p.d < PED_DETAIL_DIST).every((p) => p.detail && !p.plain), farOk = vis.filter((p) => p.d >= PED_DETAIL_DIST).every((p) => !p.detail && p.plain);
    const nNear = vis.filter((p) => p.d < PED_DETAIL_DIST).length, nFar = vis.length - nNear;
    line(vis.length === all.length && nNear >= 20 && nFar >= 4 && nearOk && farOk, 'switch by distance',
      `${nNear} people within ${PED_DETAIL_DIST} m: all detailed; ${nFar} beyond: all plain (${vis.filter((p) => p.hems).length} with a skirt or coat hem, ${vis.filter((p) => p.fineAll).length} with thin parts, ${vis.filter((p) => p.skinned).length} SkinnedMeshes)`);
    const B = await frame([], 5);
    let changed = 0, not12 = 0, both = 0, rgbChanged = 0;
    for (let i = 0; i < A.sem.length; i++) {
      const dRgb = A.rgb[3 * i] !== B.rgb[3 * i] || A.rgb[3 * i + 1] !== B.rgb[3 * i + 1] || A.rgb[3 * i + 2] !== B.rgb[3 * i + 2];
      if (dRgb) rgbChanged++;
      if (A.sem[i] !== B.sem[i]) { changed++; if (A.sem[i] !== 12) not12++; if (dRgb) both++; }
    }
    line(changed > 3000 && not12 === 0 && both >= 0.97 * changed && A.errors === 0, '3 labels',
      `${changed} pixels change when the people are drawn, ${not12} of them not class 12 (pedestrian); the picture changes at ${(100 * both / changed).toFixed(1)} % of them (${rgbChanged} pixels of the picture change: the people and their shadows)`);

    // --- view('model'): the plain people everywhere (the KeiPilot frame), and back
    await page.evaluate(() => window.__ctx.services.actors.view('model'));
    const C = await frame(all, 5), visM = await shown();
    await page.evaluate(() => window.__ctx.services.actors.view('main'));
    let n12 = 0, n12m = 0;
    for (let i = 0; i < A.sem.length; i++) { if (A.sem[i] === 12) n12++; if (C.sem[i] === 12) n12m++; }
    await frame(all, 5); const visBack = await shown();
    line(visM.every((p) => !p.detail && p.plain) && visBack.filter((p) => p.d < PED_DETAIL_DIST).every((p) => p.detail), "view('model')",
      `all ${visM.length} people plain in the model's frame (${n12m} pedestrian pixels; ${n12} with the detailed ones), detailed again after view('main')`);

    // --- one person at 10 m, plain and detailed: as tall in the picture (rows of class 12)
    const span = (sem) => { let lo = H, hi = -1; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (sem[y * W + x] === 12) { if (y < lo) lo = y; if (y > hi) hi = y; } return hi - lo + 1; };
    let worstPx = 0; const heights = [];
    for (const k of [0, 2, 4, 7, 9]) {                      // grown-ups and children
      const p = S.ped[k].slice(); p[1] = S.x0 + S.ux * 10; p[2] = S.y0 + S.uy * 10; p[14] = 0;
      const d = span((await frame([p], 6, { reset: true })).sem);
      await page.evaluate(() => window.__ctx.services.actors.view('model'));
      const m = span((await frame([p], 6)).sem);
      await page.evaluate(() => window.__ctx.services.actors.view('main'));
      heights.push(`${p[4].toFixed(2)} m: ${m} / ${d}`); worstPx = Math.max(worstPx, Math.abs(d - m));
    }
    line(worstPx <= 4, 'same size', `rows of pedestrian pixels at 10 m, plain / detailed: ${heights.join(', ')} (hats and buns reach higher; limit 4 rows)`);

    // --- frames with the same phase: standing after 0.3 s; a jump back in time or a long gap breaks nothing
    const finite = () => page.evaluate(() => { let bad = 0; for (const s of window.__ctx.services.actors.peds.values()) if (s.detail) s.detail.g.traverse((o) => { for (const v of o.matrixWorld.elements) if (!Number.isFinite(v)) bad++; }); return bad; });
    const footGap = () => page.evaluate(() => {
      const out = [], v = new window.THREE.Vector3();
      for (const s of window.__ctx.services.actors.peds.values()) {
        if (!s.detail || !s.detail.g.visible) continue;
        const z = ['shoeL', 'shoeR'].map((n) => { let m = null; s.detail.g.traverse((o) => { if (o.name === n) m = o; }); return s.detail.g.worldToLocal(v.setFromMatrixPosition(m.matrixWorld)).z; });
        out.push(Math.abs(z[0] - z[1]));
      }
      return Math.max(...out);
    });
    await frame(S.ped, 20, { reset: true }); const walking = await footGap();
    for (const t of [20.1, 20.2, 20.3, 20.4, 20.5]) await frame(S.ped, t);
    const standing = await footGap();
    await frame(S.ped, 3); await frame(S.ped, 900); await frame(S.ped, 900);
    const bad = await finite();
    line(walking > 0.15 && standing < 0.02 && bad === 0, 'standing in ego mode',
      `feet ${(walking * 100).toFixed(0)} cm apart along the walk in the first frame, ${(standing * 100).toFixed(1)} cm after 0.5 s of frames with the same phase; a jump back in time and a gap of minutes leave ${bad} broken numbers`);

    // --- dropping people leaves nothing behind (the hems' bone textures, geometries)
    const mem = () => page.evaluate(() => { const m = window.__ctx.renderer.info.memory; return [m.geometries, m.textures]; });
    const cyc = [];
    for (let k = 0; k < 5; k++) { await frame(all, 30 + k, { reset: true }); cyc.push(await mem()); }
    line(cyc[4][0] === cyc[1][0] && cyc[4][1] === cyc[1][1], 'dropped people',
      `geometries / textures after rebuilding everyone 2 and 5 times: ${cyc[1].join(' / ')} and ${cyc[4].join(' / ')}`);
  } catch (e) { line(false, 'browser', String(e && e.stack || e)); } finally { await browser.close(); server.close(); }
}
console.log('     6 speed: node tools/people_bench.mjs (measured, not judged here)');
console.log(failed ? 'RESULT: FAIL' : 'RESULT: OK');
process.exit(failed ? 1 : 0);
