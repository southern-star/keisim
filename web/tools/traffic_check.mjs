// Headless check of the browser demo's surrounding traffic (src/traffic.js) and routes (src/nav.js), in node: no
// browser, GPU or driving model. The ego is a stand-in that follows its navigation route at up to 8 m/s and stops
// behind the cars in front of it (or `--ego 0`: no ego). Prints, every `--every` s: cars, contacts with the ego,
// contacts between the other cars, their lane changes and ramp passages (towns with 片側2車線 roads or ramps), and
// the ego's route position, next turn and lane changes; at the end a hash of every car's pose (the same code and
// options give the same hash).
//   node tools/traffic_check.mjs [--town 1000] [--tdir towns] [--sec 120] [--every 10] [--cars 24] [--seed 7]
//                                [--ego 0] [--rseed 12345] [--quiet]   (rseed: the ego's route)
//                                [--why]   (each new contact: where the two cars are on their routes)
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = {};
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : '1'; args[k] = v; } }
const town = args.town || '1000', tdir = (args.tdir || 'towns').replace(/[^\w.-]/g, '');
const sec = Number(args.sec || 120), every = Number(args.every || 10), useEgo = args.ego !== '0';
// the navigator draws its turns with Math.random: a fixed stream, so that runs repeat
let st = Number(args.rseed || 12345);
Math.random = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);

const imp = (p) => import(pathToFileURL(path.join(root, p)).href);
const { buildLanes, createNavigator } = await imp('src/nav.js');
const { createTraffic } = await imp('src/traffic.js');
const { linkMap } = await imp('src/lanes.js');
const data = JSON.parse(fs.readFileSync(path.join(root, tdir, `town_${town}.json`), 'utf8'));
const lanes = buildLanes(data);
const nav = createNavigator(data, lanes);
const roadKm = lanes.reduce((a, l) => a + l.len, 0) / 1000;
const count = Number(args.cars) || Math.min(40, Math.round(roadKm * 12));
const traffic = createTraffic(data, lanes, { count, seed: Number(args.seed || 7) });

const DT = 0.05;
const p0 = nav.start();
const ego = { x: p0.x, y: p0.y, yaw: p0.yaw, v: 0 };
let egoS = nav.s;
traffic.reset(useEgo ? ego : null);
const egoChanges = new Set();
console.log(`town ${town}: ${lanes.length} lanes, ${traffic.n} cars${useEgo ? ', stand-in ego' : ''}`);
console.log('    t   cars  ego/npc contacts  lane changes  ramps   ego: s  km/h  next turn');
for (let k = 1, t = 0; k <= Math.round(sec / DT); k++) {
  t += DT;
  if (useEgo) {
    // the stand-in ego: up to 8 m/s, stopping behind a car within its corridor ahead
    const pts = nav.slice(egoS, egoS + 0.5), a = pts[0], b = pts[pts.length - 1];
    const yaw = Math.atan2(b[1] - a[1], b[0] - a[0]);
    let free = Infinity;
    for (const c of traffic.cars) {
      const dx = c.xy[0] - ego.x, dy = c.xy[1] - ego.y;
      const along = dx * Math.cos(yaw) + dy * Math.sin(yaw), lat = -dx * Math.sin(yaw) + dy * Math.cos(yaw);
      if (along > 0 && Math.abs(lat) < 1.9) free = Math.min(free, along - (c.L + 4.5) / 2);
    }
    const vt = Math.max(0, Math.min(8, (free - 3) * 0.8));
    ego.v += Math.max(-6, Math.min(2, (vt - ego.v) / 0.5)) * DT;
    ego.v = Math.max(0, ego.v);
    egoS += ego.v * DT;
    const q = nav.slice(egoS, egoS + 0.5), q0 = q[0], q1 = q[q.length - 1];
    Object.assign(ego, { x: q0[0], y: q0[1], yaw: Math.atan2(q1[1] - q0[1], q1[0] - q0[0]) });
    nav.track(ego.x, ego.y);
    for (const ch of nav.changes ? nav.changes() : []) if (ch.s1 < egoS) egoChanges.add(`${ch.from}>${ch.to}@${ch.s0.toFixed(0)}`);
  }
  const before = [traffic.contacts, traffic.npcContacts];
  traffic.step(DT, t, useEgo ? ego : null, useEgo ? { s: nav.s, decision: nav.nextDecision() } : null);
  if (args.why && (traffic.contacts > before[0] || traffic.npcContacts > before[1])) explain(t);
  if (!args.quiet && k % Math.round(every / DT) === 0) {
    const inp = nav.inputs();
    console.log(`${t.toFixed(0).padStart(5)} ${String(traffic.n).padStart(6)} ${`${traffic.contacts}/${traffic.npcContacts}`.padStart(17)}` +
      ` ${String(traffic.laneChanges ?? '-').padStart(13)} ${String(traffic.rampPassages ?? '-').padStart(6)}` +
      `  ${useEgo ? `${nav.s.toFixed(0).padStart(7)} ${(ego.v * 3.6).toFixed(0).padStart(5)}  ${inp.turn || 'straight'}` : ''}`);
  }
}
/** Print the overlapping pairs: speed, and the segment each car is on (lane id / piece, connector turn, change). */
function explain(t) {
  const where = (c) => {
    const g = c.route.segs[c.seg], nx = c.route.segs[c.seg + 1];
    const d = g.kind === 'lane' ? `lane ${g.lane.id}${g.lane.kind === 'ramp' ? ' (ramp)' : ''} at ${(g.lo + c.s - g.s0).toFixed(1)}/${g.lane.len.toFixed(0)}`
      : g.kind === 'conn' ? `conn ${g.conn.turn} ${g.conn.from.id}->${g.conn.to.id} at ${(c.s - g.s0).toFixed(1)}`
        : `change ${g.from.id}->${g.to.id} ${((c.s - g.s0) / (g.s1 - g.s0) * 100).toFixed(0)}% (at ${linkMap(g.from, g.to, c.route.localOn(g, c.s)).toFixed(1)} on ${g.to.id})${c.lcp ? (c.lcp.go ? ' go' : ' NO-GO') : ''}`;
    return `car ${c.id} v=${c.v.toFixed(1)} L=${c.L.toFixed(1)} ${d}${nx && nx.kind === 'change' ? ` (then change ${nx.from.id}->${nx.to.id} at ${nx.a0.toFixed(1)})` : ''}`;
  };
  const box = (x, y, yaw, L, W) => [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => [x + Math.cos(yaw) * L / 2 * a - Math.sin(yaw) * W / 2 * b, y + Math.sin(yaw) * L / 2 * a + Math.cos(yaw) * W / 2 * b]);
  const overlap = (P, Q) => {
    for (const poly of [P, Q]) for (let i = 0; i < 4; i++) {
      const a = poly[i], b = poly[(i + 1) % 4], n = [b[1] - a[1], a[0] - b[0]];
      const pr = (R) => R.map((p) => p[0] * n[0] + p[1] * n[1]);
      const p = pr(P), q = pr(Q);
      if (Math.max(...p) < Math.min(...q) || Math.max(...q) < Math.min(...p)) return false;
    }
    return true;
  };
  const cs = traffic.cars, bx = cs.map((c) => box(c.xy[0], c.xy[1], c.yaw, c.L, c.W));
  if (useEgo) {
    const eb = box(ego.x, ego.y, ego.yaw, 4.5, 1.85);
    cs.forEach((c, i) => { if (overlap(eb, bx[i])) console.log(`t=${t.toFixed(2)} ego (v=${ego.v.toFixed(1)}) x ${where(c)}`); });
  }
  for (let i = 0; i < cs.length; i++) for (let j = i + 1; j < cs.length; j++) {
    if (Math.hypot(cs[i].xy[0] - cs[j].xy[0], cs[i].xy[1] - cs[j].xy[1]) < 8 && overlap(bx[i], bx[j])) console.log(`t=${t.toFixed(2)} ${where(cs[i])}  x  ${where(cs[j])}`);
  }
}

const h = createHash('sha256');
for (const c of traffic.cars) h.update(`${c.id}:${c.xy[0].toFixed(4)},${c.xy[1].toFixed(4)},${c.yaw.toFixed(4)},${c.v.toFixed(4)};`);
console.log(`contacts ${traffic.contacts} (ego) / ${traffic.npcContacts} (between cars)` +
  `${traffic.laneChanges !== undefined ? `, lane changes ${traffic.laneChanges}, ramp passages ${traffic.rampPassages}` : ''}` +
  `${useEgo ? `, ego lane changes ${egoChanges.size}` : ''}`);
console.log(`ego route: ${nav.changes().length} lane changes, ${nav.changes().filter((c) => c.ramp).length / 2} ramps planned`);
const stuck = traffic.cars.filter((c) => c.stuck > 60);
console.log(`standing for over 60 s now: ${stuck.length}${stuck.length ? ` (${stuck.map((c) => `car ${c.id} ${c.stuck.toFixed(0)} s`).join(', ')})` : ''}`);
console.log(`hash ${h.digest('hex').slice(0, 16)}`);
