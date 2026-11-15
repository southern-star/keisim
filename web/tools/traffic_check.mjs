// Headless check of the browser demo's surrounding traffic (src/traffic.js) and routes (src/nav.js), in node: no
// browser, GPU or driving model. The ego is a stand-in that follows its navigation route at up to 8 m/s and stops
// behind the cars in front of it (or `--ego 0`: no ego). Prints, every `--every` s: cars, contacts with the ego,
// contacts between the other cars, their lane changes and ramp passages (towns with 片側2車線 roads or ramps), and
// the ego's route position, next turn and lane changes; at the end a hash of every car's pose (the same code and
// options give the same hash).
//   node tools/traffic_check.mjs [--town 1000] [--tdir towns] [--sec 120] [--every 10] [--cars 24] [--seed 7]
//                                [--ego 0] [--rseed 12345] [--quiet]   (rseed: the ego's route)
//                                [--why]   (each new contact: where the two cars are on their routes)
//                                [--bikes 8]   (bicycles, riders per road km: src/bicycles.js; adds their numbers)
//                                [--watch ID --from T0 --to T1]   (rider ID and the cars near it at every step)
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
const nBikes = Number(args.bikes || 0);
const traffic = createTraffic(data, lanes, { count, seed: Number(args.seed || 7), bikes: nBikes });

const DT = 0.05;
const p0 = nav.start();
const ego = { x: p0.x, y: p0.y, yaw: p0.yaw, v: 0 };
let egoS = nav.s;
traffic.reset(useEgo ? ego : null);
const egoChanges = new Set();
console.log(`town ${town}: ${lanes.length} lanes, ${traffic.n} cars${nBikes ? `, ${traffic.bikes} bicycles` : ''}${useEgo ? ', stand-in ego' : ''}`);
console.log(`    t   cars  ego/npc contacts  lane changes  ramps   ego: s  km/h  next turn${nBikes ? '   bicycles: ego/car contacts  km/h  standing' : ''}`);
let bikeV = 0, bikeN = 0, shiftMax = 0, passes = 0;
const beside = new Set();
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
    if (traffic.riders.length) {                            // … or a bicycle on its way ahead: it does not pass them;
      const way = Array.from({ length: 31 }, (_, i) => nav.slice(egoS + i, egoS + i + 0.01)[0]);
      for (const r of traffic.riders) {
        if (Math.hypot(r.x - ego.x, r.y - ego.y) > 34) continue;
        // one crossing its way that has stopped for it counts only when it is right in the way (else both would wait)
        const near = Math.cos(r.yaw - ego.yaw) > 0.5 || r.v > 0.5 ? 2.3 : 1.5;
        for (let i = 0, d = 0; i < way.length; i++) {
          if (i) d += Math.hypot(way[i][0] - way[i - 1][0], way[i][1] - way[i - 1][1]);
          if (Math.hypot(r.x - way[i][0], r.y - way[i][1]) < near) { free = Math.min(free, d - (1.8 + 4.5) / 2); break; }
        }
      }
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
  const beforeB = [traffic.bikeContacts, traffic.carBikeContacts];
  traffic.step(DT, t, useEgo ? ego : null, useEgo ? { s: nav.s, decision: nav.nextDecision() } : null);
  if (nBikes) {
    for (const r of traffic.riders) { bikeV += r.v; bikeN++; }
    for (const c of traffic.cars) {                         // passes: a car that was behind a rider on its lane is ahead
      shiftMax = Math.max(shiftMax, c.shift);
      const g = c.route.segs[c.seg];
      if (g.kind !== 'lane') continue;
      for (const r of traffic.riders) {
        if (r.piece !== g.lane) continue;
        const key = `${c.id}:${r.id}`, ds = g.lo + c.s - g.s0 - r.laneS;
        if (ds < -3) beside.add(key); else if (ds > 3 && beside.delete(key)) passes++;
      }
    }
    if (args.why && (traffic.bikeContacts > beforeB[0] || traffic.carBikeContacts > beforeB[1])) explainBikes(t);
    if (args.watch !== undefined && t >= Number(args.from || 0) && t <= Number(args.to || sec)) watch(t, traffic.riders[Number(args.watch)]);
  }
  if (args.why && (traffic.contacts > before[0] || traffic.npcContacts > before[1])) explain(t);
  if (!args.quiet && k % Math.round(every / DT) === 0) {
    const inp = nav.inputs();
    console.log(`${t.toFixed(0).padStart(5)} ${String(traffic.n).padStart(6)} ${`${traffic.contacts}/${traffic.npcContacts}`.padStart(17)}` +
      ` ${String(traffic.laneChanges ?? '-').padStart(13)} ${String(traffic.rampPassages ?? '-').padStart(6)}` +
      `  ${useEgo ? `${nav.s.toFixed(0).padStart(7)} ${(ego.v * 3.6).toFixed(0).padStart(5)}  ${(inp.turn || 'straight').padEnd(9)}` : ''.padEnd(24)}` +
      (nBikes ? `   ${`${traffic.bikeContacts}/${traffic.carBikeContacts}`.padStart(17)} ${(traffic.riders.reduce((a, r) => a + r.v, 0) / Math.max(traffic.bikes, 1) * 3.6).toFixed(1).padStart(5)} ${String(traffic.riders.filter((r) => r.v < 0.1).length).padStart(9)}` : ''));
  }
}
/** The corners of a body, and whether two of them overlap. */
function box(x, y, yaw, L, W) {
  return [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => [x + Math.cos(yaw) * L / 2 * a - Math.sin(yaw) * W / 2 * b, y + Math.sin(yaw) * L / 2 * a + Math.cos(yaw) * W / 2 * b]);
}
function overlap(P, Q) {
  for (const poly of [P, Q]) for (let i = 0; i < 4; i++) {
    const a = poly[i], b = poly[(i + 1) % 4], n = [b[1] - a[1], a[0] - b[0]];
    const pr = (R) => R.map((p) => p[0] * n[0] + p[1] * n[1]);
    const p = pr(P), q = pr(Q);
    if (Math.max(...p) < Math.min(...q) || Math.max(...q) < Math.min(...p)) return false;
  }
  return true;
}

/** Print the overlapping pairs: speed, and the segment each car is on (lane id / piece, connector turn, change). */
function explain(t) {
  const where = (c) => {
    const g = c.route.segs[c.seg], nx = c.route.segs[c.seg + 1];
    const d = g.kind === 'lane' ? `lane ${g.lane.id}${g.lane.kind === 'ramp' ? ' (ramp)' : ''} at ${(g.lo + c.s - g.s0).toFixed(1)}/${g.lane.len.toFixed(0)}`
      : g.kind === 'conn' ? `conn ${g.conn.turn} ${g.conn.from.id}->${g.conn.to.id} at ${(c.s - g.s0).toFixed(1)}`
        : `change ${g.from.id}->${g.to.id} ${((c.s - g.s0) / (g.s1 - g.s0) * 100).toFixed(0)}% (at ${linkMap(g.from, g.to, c.route.localOn(g, c.s)).toFixed(1)} on ${g.to.id})${c.lcp ? (c.lcp.go ? ' go' : ' NO-GO') : ''}`;
    return `car ${c.id} v=${c.v.toFixed(1)} L=${c.L.toFixed(1)}${c.shift ? ` shift=${c.shift.toFixed(2)}` : ''} ${d}${nx && nx.kind === 'change' ? ` (then change ${nx.from.id}->${nx.to.id} at ${nx.a0.toFixed(1)})` : ''}`;
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

/** Print a rider's state and the cars within 7 m of it, in its frame (ahead, to its left). */
function watch(t, r) {
  if (!r) return;
  const f = (x, n = 1) => (Number.isFinite(x) ? x.toFixed(n) : String(x));
  const ux = Math.cos(r.yaw), uy = Math.sin(r.yaw);
  const rel = (x, y) => `${f((x - r.x) * ux + (y - r.y) * uy)},${f(-(x - r.x) * uy + (y - r.y) * ux)}`;
  const cs = traffic.cars.filter((c) => Math.hypot(c.xy[0] - r.x, c.xy[1] - r.y) < 7)
    .map((c) => `car ${c.id} (${rel(c.xy[0], c.xy[1])}) v=${f(c.v)} dyaw=${f(c.yaw - r.yaw, 2)} sh=${f(c.shift, 2)} ${c.route.segs[c.seg].kind}`);
  if (useEgo && Math.hypot(ego.x - r.x, ego.y - r.y) < 7) cs.push(`ego (${rel(ego.x, ego.y)}) v=${f(ego.v)}`);
  for (const o of traffic.riders) if (o !== r && Math.hypot(o.x - r.x, o.y - r.y) < 7) cs.push(`rider ${o.id} (${rel(o.x, o.y)}) v=${f(o.v)}`);
  console.log(`t=${t.toFixed(2)} rider ${r.id} ${r.piece.turn ? `conn ${r.piece.turn}` : `lane ${r.piece.id}`} ${f(r.laneS)}/${f(r.piece.len, 0)} v=${f(r.v, 2)} want=${f(r.vWant, 2)} gap=${f(r.gap)} ` +
    `side=${f(r.side, 2)} out=${f(r.lat, 2)}>${f(r.latT, 2)} wait=${f(r.wait)} stuck=${f(r.stuck, 0)} | ${cs.join(' | ')}`);
}

/** Print the riders whose body overlaps a car's or the ego's: speeds, and where each of them is. */
function explainBikes(t) {
  const whereB = (r) => `rider ${r.id} v=${r.v.toFixed(1)} ${r.piece.turn ? `conn ${r.piece.turn} ${r.piece.from.id}->${r.piece.to.id}` : `lane ${r.piece.id}`} at ${r.laneS.toFixed(1)}/${r.piece.len.toFixed(0)} side=${r.side.toFixed(2)} out=${r.lat.toFixed(2)}`;
  for (const r of traffic.riders) {
    const rb = box(r.x, r.y, r.yaw, 1.8, 0.6);
    for (const c of traffic.cars) {
      if (Math.hypot(c.xy[0] - r.x, c.xy[1] - r.y) > 8 || !overlap(rb, box(c.xy[0], c.xy[1], c.yaw, c.L, c.W))) continue;
      const g = c.route.segs[c.seg];
      console.log(`t=${t.toFixed(2)} ${whereB(r)}  x  car ${c.id} v=${c.v.toFixed(1)} L=${c.L.toFixed(1)} W=${c.W.toFixed(2)} shift=${c.shift.toFixed(2)} ` +
        (g.kind === 'conn' ? `conn ${g.conn.turn} ${g.conn.from.id}->${g.conn.to.id} at ${(c.s - g.s0).toFixed(1)}` : g.kind === 'lane' ? `lane ${g.lane.id} at ${(g.lo + c.s - g.s0).toFixed(1)}` : 'changing lanes'));
    }
    if (useEgo && overlap(rb, box(ego.x, ego.y, ego.yaw, 4.5, 1.85))) console.log(`t=${t.toFixed(2)} ${whereB(r)}  x  ego v=${ego.v.toFixed(1)}`);
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
if (nBikes) {
  const st = traffic.riders.filter((r) => r.stuck > 60);
  console.log(`bicycles ${traffic.bikes}: contacts ${traffic.bikeContacts} (ego) / ${traffic.carBikeContacts} (cars), mean ${(bikeV / Math.max(bikeN, 1) * 3.6).toFixed(1)} km/h, ` +
    `passed by a car ${passes} times, cars moved over by up to ${shiftMax.toFixed(2)} m, standing for over 60 s now: ${st.length}`);
  if (args.why) {
    for (const r of st) watch(sec, r);
    for (const c of stuck) {
      const g = c.route.segs[c.seg];
      console.log(`car ${c.id} L=${c.L.toFixed(1)} shift=${c.shift.toFixed(2)} ` + (g.kind === 'conn' ? `conn ${g.conn.turn} ${g.conn.from.id}->${g.conn.to.id} at ${(c.s - g.s0).toFixed(1)}/${g.conn.len.toFixed(0)}`
        : g.kind === 'lane' ? `lane ${g.lane.id} at ${(g.lo + c.s - g.s0).toFixed(1)}/${g.lane.len.toFixed(0)}` : 'changing lanes') + ` at (${c.xy[0].toFixed(1)}, ${c.xy[1].toFixed(1)})`);
    }
  }
}
console.log(`hash ${h.digest('hex').slice(0, 16)}`);
