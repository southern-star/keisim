// Surrounding traffic for the in-browser KeiPilot (index.html?pilot=1, src/pilot.js): KeiSim's NPC vehicles in a
// lighter form. Each car follows a random route through the lanes of src/nav.js, keeps its distance with the IDM
// (to the cars and the ego ahead on its path), stops at red and yellow like keisim/traffic.py, and does not enter a
// junction while a car of another signal phase is inside it or its exit is full. Classic towns have split signals
// (one approach at a time, so turns never cross a green stream). Two-phase towns (opposite approaches green
// together, Japan's usual signal) add keisim's right-turn rule: a turn that crosses or merges with a movement of
// higher rank (right < left < straight) waits at its connector's wait point while a car with priority, the ego
// included, would reach the crossing within YIELD_GAP s. Cars stuck for long where the ego camera cannot see them
// are moved elsewhere. All coordinates are KeiSim's (x east, y north, yaw counter-clockwise).
import { heading, pointOn, wrap } from './nav.js';

const TL = { RED: 0, YELLOW: 1, GREEN: 2 };
const IDM = { a: 2.0, b: 3.0, T: 1.2, s0: 2.5 };            // keisim TrafficConfig
const LAT_ACC = 2.0;
const LOOK = 45.0;                                          // m of path searched for a leader
const YIELD_GAP = 5.0;                                      // keisim/traffic.py YIELD_GAP
const YIELD_CLEAR = 2.4;                                    // keisim/roadnet.py YIELD_CLEAR
const RANK = { right: 0, left: 1, straight: 2 };            // the lower-ranked movement yields
const EGO = { L: 4.5, W: 1.85 };
// keisim/traffic.py CAR_COLORS (BGR there) as RGB
const COLORS = [[235, 235, 235], [238, 240, 240], [195, 190, 190], [32, 30, 30], [125, 120, 120], [25, 60, 140],
  [170, 40, 40], [190, 110, 30], [50, 70, 70], [90, 140, 180], [60, 130, 60]];

/** keisim Junction.signal_state */
export function signalState(J, phase, t) {
  if (!J.signalized) return TL.GREEN;
  const per = J.green + J.yellow + J.allred;
  const T = per * J.phases;
  const tau = (((t + J.offset) % T) + T) % T;
  const k = Math.floor(tau / per);
  if (k !== phase) return TL.RED;
  const r = tau - k * per;
  return r < J.green ? TL.GREEN : r < J.green + J.yellow ? TL.YELLOW : TL.RED;
}

/** keisim Traffic._reach: time to cover d from speed v, accelerating at a up to vmax. */
function reach(d, v, vmax, a) {
  const t1 = Math.max(0, (vmax - v) / a);
  const d1 = v * t1 + 0.5 * a * t1 * t1;
  if (d1 >= d) return (-v + Math.sqrt(v * v + 2 * a * d)) / a;
  return t1 + (d - d1) / Math.max(vmax, 0.1);
}

/** keisim Traffic._arrival: moving cars at 2.5 m/s^2, standing ones after a 1 s reaction at 2 m/s^2, the ego at
 *  once at 3.5 m/s^2 (the driving model may go at any moment). */
function arrival(d, v, vmax, ego = false) {
  if (d <= 0) return 0;
  const [a, t0] = ego ? [3.5, 0] : v >= 0.2 ? [2.5, 0] : [2.0, 1.0];
  return t0 + reach(d, v, vmax, a);
}

/** keisim Traffic._will_stop: whether a car d m before its red / yellow stop line (front bumper) stops there. */
function willStop(st, d, v, ego = false) {
  if (ego) return v < 2.0 && d > -0.5;
  if (d < -0.5) return false;
  if (st === TL.RED) return d + 0.5 >= v * v / 16;
  return d > v * v / 7 + 1;
}

/** Oriented boxes [x, y, yaw, half length, half width] overlap (separating axes). */
function boxesOverlap(A, B) {
  const corners = [A, B].map(([x, y, yaw, hl, hw]) => {
    const u = [Math.cos(yaw), Math.sin(yaw)], n = [-u[1], u[0]];
    return [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => [x + u[0] * hl * a + n[0] * hw * b, y + u[1] * hl * a + n[1] * hw * b]);
  });
  for (const [, , yaw] of [A, B]) {
    for (const ax of [[Math.cos(yaw), Math.sin(yaw)], [-Math.sin(yaw), Math.cos(yaw)]]) {
      const pr = corners.map((cs) => cs.map((p) => p[0] * ax[0] + p[1] * ax[1]));
      if (Math.max(...pr[0]) < Math.min(...pr[1]) || Math.max(...pr[1]) < Math.min(...pr[0])) return false;
    }
  }
  return true;
}

/** A car's route: a growing polyline of lanes and junction connectors, chosen at random at every junction. */
class Route {
  constructor(lane, rnd) {
    this.rnd = rnd;
    this.pts = []; this.cum = []; this.segs = [];
    this.last = lane; this.hint = 1;
    this.append(lane.pts);
    this.segs.push({ kind: 'lane', lane, s0: 0, s1: this.end });
  }

  get end() { return this.cum[this.cum.length - 1]; }

  append(points) {
    for (const p of points) {
      if (this.pts.length) {
        const q = this.pts[this.pts.length - 1], d = Math.hypot(p[0] - q[0], p[1] - q[1]);
        if (d < 1e-6) continue;
        this.cum.push(this.end + d);
      } else this.cum.push(0);
      this.pts.push(p);
    }
  }

  extend(until) {
    while (this.end < until) {
      const opts = this.last.next;
      if (!opts.length) return;
      const c = opts[Math.floor(this.rnd() * opts.length)];
      const a = this.end;
      this.append(c.conn.pts);                      // its first point is the lane's last: arc lengths match conn.cum
      this.segs.push({ kind: 'conn', junction: this.last.to, from: this.last, conn: c.conn, s0: a, s1: this.end });
      const b = this.end;
      this.append(c.lane.pts);
      this.segs.push({ kind: 'lane', lane: c.lane, s0: b, s1: this.end });
      this.last = c.lane;
    }
  }

  /** Drop what lies more than `keep` m behind arc length s; returns the shift to subtract from s. */
  trim(s, keep = 30) {
    let i = 1;
    while (i < this.cum.length - 1 && this.cum[i] < s - keep) i++;
    if (i < 50) return 0;
    const base = this.cum[i - 1];
    this.pts = this.pts.slice(i - 1);
    this.cum = this.cum.slice(i - 1).map((c) => c - base);
    this.segs = this.segs.filter((g) => g.s1 > base).map((g) => ({ ...g, s0: g.s0 - base, s1: g.s1 - base }));
    this.hint = 1;
    return base;
  }

  locate(s) {
    let i = Math.max(1, Math.min(this.hint, this.cum.length - 1));
    while (i > 1 && this.cum[i - 1] > s) i--;
    while (i < this.cum.length - 1 && this.cum[i] < s) i++;
    this.hint = i;
    return i;
  }

  pointAt(s) {
    const i = this.locate(s), a = this.pts[i - 1], b = this.pts[i];
    const f = Math.max(0, Math.min(1, (s - this.cum[i - 1]) / ((this.cum[i] - this.cum[i - 1]) || 1e-9)));
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
  }

  headingAt(s) {
    const i = this.locate(s);
    return heading(this.pts[i - 1], this.pts[i]);
  }

  segAt(s) {
    for (let k = 0; k < this.segs.length; k++) if (s < this.segs[k].s1 || k === this.segs.length - 1) return k;
    return this.segs.length - 1;
  }
}

/**
 * opts: {count, seed}. Returns {step(dt, t, ego, egoNav), drawList(cx, cy, r), reset(ego), contacts, npcContacts,
 * n, twoPhase}. ego: {x, y, yaw, v} (centre), 4.5 x 1.85 m. egoNav (optional): {s, decision} from src/nav.js (the
 * ego's route position and next signalised junction), so the other cars know which lane or connector the ego is on.
 */
export function createTraffic(data, lanes, { count = 24, seed = 1 } = {}) {
  let st = seed >>> 0 || 1;
  const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  const junctions = new Map(data.junctions.map((j) => [j.id, j]));
  const STOP_SETBACK = data.cfg.stop_line_setback ?? 5.0;     // keisim TownConfig.stop_line_setback
  const VMAX = 1.05 * (data.cfg.speed_limit ?? 10.0);
  // the signal phase of every lane that ends at a signalised junction: the head facing it
  const phaseOf = new Map();
  for (const L of lanes) {
    const J = junctions.get(L.to);
    if (!J || !J.signalized) continue;
    let best = null, bd = 0.7;
    for (const h of data.signals) {
      if (h.junction !== J.id) continue;
      const d = Math.abs(wrap(h.yaw - (L.h1 + Math.PI)));
      if (d < bd) { bd = d; best = h; }
    }
    if (best) phaseOf.set(L.id, best.phase);
  }
  const twoPhase = data.cfg.signal_mode === 'two_phase';
  if (twoPhase) makeYields();
  const roadLanes = lanes.filter((L) => L.len > 30);
  const totalLen = roadLanes.reduce((a, L) => a + L.len, 0);
  const cars = [];
  let contacts = 0, lastContact = -1e9, npcContacts = 0;
  const touching = new Set();

  /** keisim Town._make_yields: inside every signalised junction, a connector yields to the connectors of the other
   *  approaches green at the same time whose movement ranks higher and whose centre line comes within 2.2 m of its
   *  own (a crossing or a merge): yields [{b, here, there}] (closest points), and wait: the furthest front-bumper
   *  position whose front circle (0.9 m behind the bumper) stays YIELD_CLEAR from all of those paths. */
  function makeYields() {
    const byJunction = new Map();
    for (const L of lanes) {
      const J = junctions.get(L.to);
      if (!J || !J.signalized) continue;
      if (!byJunction.has(J.id)) byJunction.set(J.id, []);
      for (const o of L.next) byJunction.get(J.id).push(o.conn);
    }
    for (const conns of byJunction.values()) {
      for (const a of conns) {
        const ga = phaseOf.get(a.from.id);
        for (const b of conns) {
          const gb = phaseOf.get(b.from.id);
          if (b.from === a.from || ga === undefined || ga !== gb || RANK[a.turn] >= RANK[b.turn]) continue;
          let best = Infinity, here = 0, there = 0;
          for (let i = 0; i < a.pts.length; i++) {
            for (let j = 0; j < b.pts.length; j++) {
              const d = Math.hypot(a.pts[i][0] - b.pts[j][0], a.pts[i][1] - b.pts[j][1]);
              if (d < best) { best = d; here = a.cum[i]; there = b.cum[j]; }
            }
          }
          if (best < 2.2) a.yields.push({ b, here, there });
        }
      }
      for (const a of conns) {
        if (!a.yields.length) continue;
        a.yields.sort((p, q) => p.here - q.here);
        const others = a.yields.flatMap((y) => y.b.pts);
        const h0 = heading(a.pts[0], a.pts[1]);
        a.wait = 0;
        for (let sb = a.yields[0].here; sb >= -0.01; sb -= 0.25) {
          const sc = sb - 0.9;
          const c = sc >= 0 ? pointOn(a, sc) : [a.pts[0][0] + sc * Math.cos(h0), a.pts[0][1] + sc * Math.sin(h0)];
          if (others.every((p) => Math.hypot(p[0] - c[0], p[1] - c[1]) >= YIELD_CLEAR)) { a.wait = sb; break; }
        }
      }
    }
  }

  function sampleDims() {
    const r = rnd();
    if (r < 0.62) return [4.0 + 0.8 * rnd(), 1.68 + 0.17 * rnd(), 1.4 + 0.2 * rnd(), 0];
    if (r < 0.78) return [3.35 + 0.1 * rnd(), 1.48, 1.6 + 0.2 * rnd(), 0];          // kei car
    if (r < 0.92) return [4.6 + 0.4 * rnd(), 1.8 + 0.15 * rnd(), 1.75 + 0.25 * rnd(), 0];
    return [6.0 + 1.6 * rnd(), 2.0 + 0.25 * rnd(), 2.6 + 0.4 * rnd(), 1];             // small truck
  }

  /** A free spot on a random lane, away from the ego and the other cars, never right before a stop line. */
  function place(car, ego, avoidEgo) {
    for (let tries = 0; tries < 60; tries++) {
      let x = rnd() * totalLen, L = roadLanes[0];
      for (const l of roadLanes) { if (x < l.len) { L = l; break; } x -= l.len; }
      const hi = phaseOf.has(L.id) ? L.len - STOP_SETBACK - 20 : L.len - 4;
      if (hi < 6) continue;
      const s = 4 + rnd() * (hi - 4);
      const route = new Route(L, rnd);
      const p = route.pointAt(s);
      if (ego && Math.hypot(p[0] - ego.x, p[1] - ego.y) < avoidEgo) continue;
      if (cars.some((c) => c !== car && c.xy && Math.hypot(c.xy[0] - p[0], c.xy[1] - p[1]) < 12)) continue;
      route.extend(s + 80);
      Object.assign(car, { route, s, seg: route.segAt(s), xy: p, yaw: route.headingAt(s), stuck: 0 });
      return true;
    }
    return false;
  }

  function spawn(ego) {
    cars.length = 0;
    for (let i = 0; i < count; i++) {
      const [L, W, H, kind] = sampleDims();
      const vf = 0.8 + 0.25 * rnd();
      const c = COLORS[Math.floor(rnd() * COLORS.length)].map((x) => Math.max(0, Math.min(255, x * (0.9 + 0.18 * rnd()))));
      const car = { id: i, L, W, H, kind, rgb: c, vf, v: 0, brake: false };
      if (place(car, ego, 30)) {
        car.v = (0.3 + 0.5 * rnd()) * 10 * vf;
        cars.push(car);
      }
    }
  }

  const push = (m, k, e) => { if (!m.has(k)) m.set(k, []); m.get(k).push(e); };

  /** Who is where: cars on every lane (front first, with the connector they take next), on every connector, and the
   *  signal phases of the approaches of the cars inside every signalised junction (keisim box_occupants: a car past
   *  its stop line that is still moving, i.e. went in at the end of yellow, counts too). The ego is placed by its
   *  route (egoNav); inside a signalised junction without one, it blocks every approach. */
  function occupancy(ego, en) {
    const onLane = new Map(), onConn = new Map(), box = new Map();
    for (const c of cars) {
      const g = c.route.segs[c.seg];
      if (g.kind === 'lane') {
        const nx = c.route.segs[c.seg + 1];
        push(onLane, g.lane.id, { s: c.s - g.s0, v: c.v, L: c.L, next: nx && nx.kind === 'conn' ? nx.conn : null });
        if (phaseOf.has(g.lane.id) && c.s - g.s0 + c.L / 2 > g.lane.len - STOP_SETBACK + 0.5 && c.v > 1.0) {
          push(box, g.lane.to, phaseOf.get(g.lane.id));
        }
      } else {
        push(onConn, g.conn, { s: c.s - g.s0, v: c.v, L: c.L });
        if (phaseOf.has(g.from.id)) push(box, g.junction, phaseOf.get(g.from.id));
      }
    }
    if (ego) {
      const d = en && en.decision;
      let inside = false;
      if (d && d.conn) {
        if (en.s < d.sIn && en.s > d.sIn - d.lane.len) {
          push(onLane, d.lane.id, { s: d.lane.len - (d.sIn - en.s), v: ego.v, L: EGO.L, next: d.conn, ego: true });
          if (phaseOf.has(d.lane.id) && en.s + EGO.L / 2 > d.sIn - STOP_SETBACK + 0.5 && ego.v > 1.0) {
            push(box, d.junction, phaseOf.get(d.lane.id)); inside = true;
          }
        } else if (en.s >= d.sIn && en.s < d.sOut) {
          push(onConn, d.conn, { s: en.s - d.sIn, v: ego.v, L: EGO.L, ego: true });
          if (phaseOf.has(d.lane.id)) { push(box, d.junction, phaseOf.get(d.lane.id)); inside = true; }
        }
      }
      if (!inside) {
        for (const J of junctions.values()) {
          if (J.signalized && Math.hypot(ego.x - J.pos[0], ego.y - J.pos[1]) < J.radius && !(d && d.junction === J.id)) {
            push(box, J.id, -1);                     // the ego (approach unknown): blocks everyone
          }
        }
      }
    }
    for (const l of onLane.values()) l.sort((a, b) => b.s - a.s);
    return { onLane, onConn, box };
  }

  /** keisim Traffic.min_arrival: earliest time a car with priority over connector `conn` can be in its crossing
   *  (0: already in it). Cars that will stop at their red / yellow line do not count, nor do cars queued behind one
   *  that stands still to go elsewhere (e.g. an oncoming car waiting to turn right itself). */
  function minArrival(conn, t, occ) {
    let best = Infinity;
    for (const { b, there: s2 } of conn.yields) {
      for (const o of occ.onConn.get(b) || []) {
        if (o.s - o.L / 2 > s2 + 1.5) continue;              // its rear is past the crossing
        const d = s2 - (o.s + o.L / 2);
        best = Math.min(best, d < 3 ? 0 : arrival(d, o.v, VMAX, o.ego));
      }
      const pred = b.from;
      // a car standing at the entry of one of pred's other connectors blocks the whole approach lane
      if (pred.next.some((x) => x.conn !== b && (occ.onConn.get(x.conn) || []).some((e) => e.v < 0.5 && e.s < 4))) continue;
      const J = junctions.get(pred.to), ph = phaseOf.get(pred.id);
      const sig = ph === undefined ? TL.GREEN : signalState(J, ph, t);
      for (const o of occ.onLane.get(pred.id) || []) {     // front first
        if (o.next !== b) {
          if (o.v < 0.5) break;                               // standing to go elsewhere: the rest queue behind
          continue;
        }
        const front = o.s + o.L / 2;
        if (!(sig !== TL.GREEN && willStop(sig, pred.len - STOP_SETBACK - front, o.v, o.ego))) {
          best = Math.min(best, arrival(pred.len - front + s2, o.v, VMAX, o.ego));
        }
        break;                                                // the cars behind it arrive later
      }
    }
    return best;
  }

  function leader(c, ego) {
    let gap = Infinity, vl = 0;
    const others = cars.filter((o) => o !== c && Math.abs(o.xy[0] - c.xy[0]) < LOOK + 8 && Math.abs(o.xy[1] - c.xy[1]) < LOOK + 8)
      .map((o) => ({ xy: o.xy, yaw: o.yaw, v: o.v, L: o.L }));
    if (ego) others.push({ xy: [ego.x, ego.y], yaw: ego.yaw, v: ego.v, L: EGO.L });
    if (!others.length) return { gap, vl };
    for (let a = 1.0; a <= LOOK; a += 1.5) {
      const p = c.route.pointAt(c.s + a);
      for (const o of others) {
        if (Math.abs(o.xy[0] - p[0]) > 2.6 || Math.abs(o.xy[1] - p[1]) > 2.6) continue;
        if (Math.hypot(o.xy[0] - p[0], o.xy[1] - p[1]) < 1.9) {
          const g = a - c.L / 2 - o.L / 2;
          if (g < gap) {
            gap = g;
            vl = o.v * Math.cos(wrap(o.yaw - c.route.headingAt(c.s + a)));
          }
        }
      }
      if (gap < Infinity && a > gap + 8) break;
    }
    return { gap, vl };
  }

  function step(dt, t, ego, egoNav) {
    const occ = occupancy(ego, egoNav);
    for (const c of cars) {
      const k = c.seg, g = c.route.segs[k];
      const v0 = 10 * c.vf;
      // curve speed envelope over the next 30 m (keisim: sqrt(vc^2 + 4 d))
      let vmax = v0;
      for (let a = 2; a <= 30; a += 4) {
        const dh = Math.abs(wrap(c.route.headingAt(c.s + a + 2) - c.route.headingAt(c.s + a - 2)));
        if (dh > 1e-3) vmax = Math.min(vmax, Math.sqrt(LAT_ACC / (dh / 4) + 4 * a));
      }
      let { gap, vl } = leader(c, ego);
      // stop line: red / yellow (if it can stop), or the box rule / a full exit at green
      if (g.kind === 'lane' && phaseOf.has(g.lane.id)) {
        const d = g.s0 + g.lane.len - STOP_SETBACK - (c.s + c.L / 2);
        if (d > -0.5 && d < LOOK) {
          const J = junctions.get(g.lane.to);
          const ph = phaseOf.get(g.lane.id);
          const sig = signalState(J, ph, t);
          let stop = sig === TL.RED || (sig === TL.YELLOW && d > c.v * c.v / 7 + 1);
          if (!stop && d < 12) {
            const inBox = occ.box.get(J.id);
            if (inBox && inBox.some((p) => p !== ph)) stop = true;          // a car of another phase is inside
            const exit = c.route.segs[k + 2];
            if (!stop && exit && exit.kind === 'lane') {
              const q = occ.onLane.get(exit.lane.id) || [];
              if (q.some((o) => o.s - o.L / 2 < c.L + 3 && o.v < 2)) stop = true;
            }
          }
          if (stop && d - 0.6 < gap) { gap = d - 0.6; vl = 0; }
        }
      }
      // right turns that give way (two-phase): wait at the wait point while priority traffic would arrive within
      // YIELD_GAP s; past the wait point, or too close to stop gently, the turn goes on (keisim Traffic._yield_gaps)
      if (twoPhase) {
        let yc = null, dW = 0;
        if (g.kind === 'conn') {
          if (g.conn.yields.length) { yc = g.conn; dW = g.conn.wait - (c.s - g.s0 + c.L / 2); }
        } else {
          const nx = c.route.segs[k + 1];
          if (nx && nx.kind === 'conn' && nx.conn.yields.length) { yc = nx.conn; dW = g.s1 - (c.s + c.L / 2) + nx.conn.wait; }
        }
        if (yc && dW >= -0.5 && dW <= 40 && !(c.v > 1 && dW < c.v * c.v / 8) && minArrival(yc, t, occ) < YIELD_GAP) {
          const gw = Math.max(dW, 0) + IDM.s0 - 0.3;          // the IDM stands still s0 behind the gap: at the point
          if (gw < gap) { gap = gw; vl = 0; }
        }
      }
      let acc = IDM.a * (1 - (c.v / Math.max(vmax, 0.5)) ** 4);
      if (gap < Infinity) {
        const ss = IDM.s0 + Math.max(0, c.v * IDM.T + c.v * (c.v - vl) / (2 * Math.sqrt(IDM.a * IDM.b)));
        acc = Math.min(acc, IDM.a * (1 - (c.v / Math.max(vmax, 0.5)) ** 4 - (ss / Math.max(gap, 0.05)) ** 2));
      }
      acc = Math.max(-8, Math.min(IDM.a, acc));
      c.brake = acc < -0.4 || c.v < 0.2;
      c.v = Math.max(0, c.v + acc * dt);
      c.s += c.v * dt;
      c.stuck = c.v < 0.1 ? c.stuck + dt : 0;
    }
    for (const c of cars) {
      c.route.extend(c.s + 80);
      c.s -= c.route.trim(c.s);
      c.seg = c.route.segAt(c.s);
      c.xy = c.route.pointAt(c.s);
      c.yaw = c.route.headingAt(c.s);
      // stuck for long where the ego camera cannot see it (behind, beside or far): move it elsewhere
      if (c.stuck > 40 && ego) {
        const dx = c.xy[0] - ego.x, dy = c.xy[1] - ego.y, dist = Math.hypot(dx, dy);
        const off = Math.abs(wrap(Math.atan2(dy, dx) - ego.yaw));
        if (dist > 110 || (dist > 8 && off > 1.05)) place(c, ego, 50);
      }
    }
    if (ego) {                                              // contacts with the ego (bodies overlapping)
      for (const c of cars) {
        if (Math.hypot(c.xy[0] - ego.x, c.xy[1] - ego.y) > 6) continue;
        if (boxesOverlap([c.xy[0], c.xy[1], c.yaw, c.L / 2, c.W / 2], [ego.x, ego.y, ego.yaw, EGO.L / 2, EGO.W / 2]) &&
            t - lastContact > 2) { contacts++; lastContact = t; }
      }
    }
    // contacts between the other cars (a check on the rules above; the HUD does not show it)
    for (let i = 0; i < cars.length; i++) {
      for (let j = i + 1; j < cars.length; j++) {
        const a = cars[i], b = cars[j], key = a.id * 1000 + b.id;
        const near = Math.abs(a.xy[0] - b.xy[0]) < 8 && Math.abs(a.xy[1] - b.xy[1]) < 8 &&
          boxesOverlap([a.xy[0], a.xy[1], a.yaw, a.L / 2, a.W / 2], [b.xy[0], b.xy[1], b.yaw, b.L / 2, b.W / 2]);
        if (near && !touching.has(key)) { touching.add(key); npcContacts++; } else if (!near) touching.delete(key);
      }
    }
  }

  /** actors.js format: [id, x, y, yaw, L, W, H, r, g, b, kind, brake] for the cars within r of (cx, cy). */
  function drawList(cx, cy, r = 140) {
    const out = [];
    for (const c of cars) {
      if (Math.abs(c.xy[0] - cx) > r || Math.abs(c.xy[1] - cy) > r) continue;
      out.push([c.id, c.xy[0], c.xy[1], c.yaw, c.L, c.W, c.H, c.rgb[0], c.rgb[1], c.rgb[2], c.kind, c.brake ? 1 : 0]);
    }
    return out;
  }

  function reset(ego) {
    spawn(ego);
    contacts = 0; npcContacts = 0; touching.clear();
  }

  return {
    step, drawList, reset, cars, twoPhase,
    get contacts() { return contacts; }, get npcContacts() { return npcContacts; }, get n() { return cars.length; },
  };
}
