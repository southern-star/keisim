// ?gallery=people — the pedestrians of the redesign (src/world/people.js): one column for each of the seven types,
// four people deep, walking on the spot at KeiSim's phase (4.2 rad per metre). Heights and colours are drawn as
// KeiSim draws them (a fixed seed; the children's as keisim/parked.py draws the ones that run out), and each column
// takes the first people whose hash gives its type. One bicycle rider (src/world/bikes.js) stands beside them to
// compare the builds. The front row stops for 2 s in every 6 s: the walk settles into the standing pose and starts again.
//   ?only=office,elder      only these types (kid student office shopper elder worker casual)
//   ?rows=6                 people per column          ?seed=3   other people
//   ?has=box,cap            only people with all of these: a thing carried (randoseru rucksack sling school brief tote
//                           plastic cane umbrella phone box headphones), hair or hat (short bob long pony bun twin cap brim
//                           tulip helmet hood), top (tee shirt jacket parka coat vest), bottom (long shorts skirt longskirt),
//                           shoes (sneaker leather boots), glasses, mask
//   ?walk=0 (or ?stand=1)   everybody stands           ?stop=0   nobody stops
//   ?sync=1                 everybody in step: the same phase and 1.3 m/s (to compare the types' ways of walking)
//   ?file=1                 the types one behind the other, 1.5 m apart (a camera at azimuth 90 sees every one from the side)
//   S                       stand / walk
import * as THREE from 'three';
import { makePerson, personLook, PERSON_TYPES, STEP } from './world/people.js';
import { makeBike } from './world/bikes.js';

function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
// keisim/traffic.py SHIRTS, PANTS, SKIN and keisim/parked.py KID_SHIRTS (BGR there, RGB here)
const SHIRTS = [[40, 60, 200], [200, 40, 40], [40, 160, 40], [230, 230, 230], [30, 30, 30], [230, 140, 0], [150, 50, 150], [220, 200, 60], [120, 120, 120], [80, 150, 200]];
const PANTS = [[25, 30, 40], [30, 60, 90], [30, 30, 30], [100, 110, 120], [110, 50, 60], [150, 150, 150]];
const SKIN = [[220, 180, 150], [205, 160, 120], [170, 125, 90], [130, 90, 60]];
const KID_SHIRTS = [[230, 60, 40], [30, 120, 230], [240, 200, 40], [200, 60, 200], [90, 200, 60], [240, 240, 240]];
const q8 = (v) => Math.round(Math.min(255, Math.max(0, v)) / 16) * 16;
const rgbColor = (c) => new THREE.Color().setRGB(q8(c[0]) / 255, q8(c[1]) / 255, q8(c[2]) / 255, THREE.SRGBColorSpace);   // as actors.js
const JA = { kid: '小学生', student: '中高生', office: '会社員', shopper: '買い物客', elder: '高齢者', worker: '作業員・配達員', casual: '普段着' };

const DX = 1.3, DZ = 1.8, WALK = 4, STOP = 2;

export async function setup({ ctx, scene, params }) {
  const mat = ctx.mat.toon('#ffffff', { vertexColors: true, paint: 0.02 });      // the riders' material (actors.js mBike)
  const r = rng(Number(params.get('seed') || 5));
  const types = params.get('only') ? params.get('only').split(',').filter((t) => PERSON_TYPES.includes(t)) : PERSON_TYPES;
  const ROWS = Number(params.get('rows') || 4), COLS = types.length, stops = params.get('stop') !== '0';
  let standAll = params.get('walk') === '0' || params.get('stand') === '1';
  const sync = params.get('sync') === '1', file = params.get('file') === '1';
  const root = new THREE.Group(); root.name = 'gallery people';
  scene.add(root);
  const has = params.get('has') ? params.get('has').split(',') : [];
  const tags = (L) => [L.headgear, L.hair, L.top, L.bottom, L.shoes, ...L.things, L.glasses && 'glasses', L.mask && 'mask'];
  // people as KeiSim makes them, until every column is full
  const want = new Map(types.map((t) => [t, []]));
  for (let id = 0; id < 60000 && [...want.values()].some((a) => a.length < ROWS); id++) {
    const kid = id % 5 === 4, k = 0.85 + 0.25 * r();
    const h = +(kid ? 1.0 + 0.35 * r() : 1.5 + 0.35 * r()).toFixed(2);
    const cols = kid ? [rgbColor(KID_SHIRTS[Math.floor(r() * KID_SHIRTS.length)]), rgbColor(PANTS[0]), rgbColor(SKIN[1])]
      : [rgbColor(SHIRTS[Math.floor(r() * SHIRTS.length)].map((v) => v * k)), rgbColor(PANTS[Math.floor(r() * PANTS.length)]), rgbColor(SKIN[Math.floor(r() * SKIN.length)])];
    const speed = kid ? 2.2 + 0.8 * r() : 1.0 + 0.6 * r(), phase0 = 2 * Math.PI * r();
    const L = personLook(h, cols, id), list = want.get(L.type);
    if (list && list.length < ROWS && has.every((t) => tags(L).includes(t))) list.push({ id, h, cols, speed, phase0 });
  }
  const people = [];
  types.forEach((type, col) => want.get(type).forEach((q, row) => {
    const P = makePerson(ctx, mat, q.h, q.cols, q.id);
    if (file) P.g.position.set(-row * DX, 0, ((COLS - 1) / 2 - col) * 1.5);       // in single file, the first type in front
    else if (COLS === 1) P.g.position.set((row - (ROWS - 1) / 2) * DX, 0, 0);     // one type: side by side
    else P.g.position.set((col - (COLS - 1) / 2) * DX, 0, -row * DZ);
    P.g.traverse((o) => { if (o.isMesh) o.userData.dynamic = true; });
    root.add(P.g);
    people.push({ ...P, id: q.id, h: q.h, row: COLS === 1 ? row % 2 : row, col, speed: sync ? 1.3 : q.speed, phase0: sync ? 0 : q.phase0, phase: 0 });
  }));
  // the rider, standing with the left foot on the ground, beside the front row
  const rider = makeBike(ctx, mat, 0, 1.7, [rgbColor([60, 110, 160]), rgbColor([230, 230, 230]), rgbColor([30, 60, 90]), rgbColor(SKIN[0])], 4);
  rider.g.position.set(((COLS === 1 ? ROWS : COLS) / 2 + 0.6) * DX, 0, 0.1);
  if (file) rider.g.position.set(0, 0, -((COLS - 1) / 2 + 1.2) * 1.5);          // in single file: the last
  rider.pose([0.05, 0, 0.9, 0, 1, 0, 0]);
  root.add(rider.g);

  /** Metres walked by time t: the front row stands for STOP s after every WALK s. */
  const walked = (p, t) => (stops && p.row === 0 ? p.speed * (WALK * Math.floor(t / (WALK + STOP)) + Math.min(t % (WALK + STOP), WALK)) : p.speed * t);
  const moving = (p, t) => !(stops && p.row === 0 && t % (WALK + STOP) >= WALK);
  function update(dt, t) {
    for (const p of people) {
      p.phase = p.phase0 + walked(p, Math.max(0, t)) / STEP;
      const mv = moving(p, Math.max(0, t));
      if (standAll) p.pose({ phase: p.phase, stand: true });
      else if (t < last) p.pose({ phase: p.phase, stand: !mv });       // time went back (stills): the pose of that moment at once
      else p.pose({ phase: p.phase, moving: mv, dt });
    }
    last = t;
  }
  let last = -Infinity;
  return {
    title: '歩行者（作り直し P2: 類型ごとの歩き方）', people, rider,
    caption: `列は左から ${types.map((t) => JA[t]).join(' / ')}。その場で歩きます（位相は KeiSim と同じ 1 m あたり 4.2 rad）。前の列は ${WALK} 秒歩いて ${STOP} 秒止まります。右端は比較用の自転車の乗り手。S: 全員の立ち姿 / 歩き。?sync=1 で全員の足並みをそろえます。`,
    orbit: COLS === 1 ? { az: 20, el: 8, dist: 4 + 0.9 * ROWS, y: 0.9, z: 0 } : { az: 28, el: 9, dist: 9 + 1.1 * COLS, y: 0.9, z: -(ROWS - 1) * DZ / 2 },
    update,
    key(code) { if (code === 'KeyS') { standAll = !standAll; if (!standAll) for (const p of people) p.pose({ phase: p.phase, stand: false }); } },
  };
}
