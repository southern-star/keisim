// Surrounding traffic for the in-browser KeiPilot (index.html?pilot=1, src/pilot.js): KeiSim's NPC vehicles in a
// lighter form. Each car follows a random route through the lanes of src/nav.js, keeps its distance with the IDM
// (to the cars and the ego ahead on its path), stops at red and yellow like keisim/traffic.py, and does not enter a
// junction while a car of another signal phase is inside it or its exit is full. Classic towns have split signals
// (one approach at a time, so turns never cross a green stream). Two-phase towns (opposite approaches green
// together, Japan's usual signal) add keisim's right-turn rule: a turn that crosses or merges with a movement of
// higher rank (right < left < straight) waits at its connector's wait point while a car with priority, the ego
// included, would reach the crossing within YIELD_GAP s; of two equal movements that cross (the right turns at a
// skewed junction), one gives way (equalGivesWay). Real towns (keisim/realworld.py, e.g. town osm_sapporo) add
// keisim's priority crossings: at an unsignalised junction of 3+ arms the minor road's cars give way to every
// movement of the priority road (priorityGroups), a right turn also waits at its entry for the oncoming left turn
// around the same corner (mirrored_turn_yield), the minor road's cars stop once at its 止まれ line before that, and
// the NPC fleet is capped at max_vehicle_dims. Cars stuck for long where the ego camera cannot see them are moved
// elsewhere. Towns exported with KeiSim's lane graph (片側2車線 main roads and ramps, src/lanes.js) add keisim's lane
// changes: a car's route picks the movement at the next junction from any lane of the road and changes lanes for it
// (route.lane_moves), checks for room in the target lane from LC.CHECK m before the start (route.change_room, the
// ego included), moves the start on while there is none and waits at the window's end; the cars behind in the
// target lane follow a car once it has begun to change, and leave a gap to one that waits (keisim
// Traffic._lane_change_gaps). Such towns also bring KeiSim's give-way table (narrow bends taken one at a time), their
// roads' speed limits and the lanes trucks may use. All coordinates are KeiSim's (x east, y north, yaw
// counter-clockwise).
import { heading, pointOn, wrap } from './nav.js';
import { LC, Path, changeRoom, changeWindow, changedInto, laneHeading, laneMoves, linkMap, pickMove, projectOn } from './lanes.js';

const TL = { RED: 0, YELLOW: 1, GREEN: 2 };
const IDM = { a: 2.0, b: 3.0, T: 1.2, s0: 2.5 };            // keisim TrafficConfig
const LAT_ACC = 2.0;
const LOOK = 45.0;                                          // m of path searched for a leader
const YIELD_GAP = 5.0;                                      // keisim/traffic.py YIELD_GAP
const YIELD_CLEAR = 2.4;                                    // keisim/roadnet.py YIELD_CLEAR
const RANK = { right: 0, left: 1, straight: 2 };            // the lower-ranked movement yields
const MIRROR_PASS = 3.6;                                    // keisim/roadnet.py MIRROR_PASS
const KIND_RANK = { alley: 0, residential: 1, street: 2, main: 3 };   // road types: the minor road gives way
const STOP_SIGN_SETBACK = 2.0;                              // keisim/roadnet.py: 止まれ line before the minor approach's end
const STOP_SIGN_NPC_AT = 1.0, STOP_SIGN_NPC_ZONE = 3.0;     // keisim/traffic.py: stop 1 m before it, once (standing within 3 m)
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

/** A car's route: a growing path of lanes, junction connectors and lane changes (src/lanes.js), its movements drawn at
 *  random: at every junction (single lanes), or with keisim's lane moves on multi-lane roads and before ramps. truck:
 *  the lanes trucks may use (or null). */
class Route extends Path {
  constructor(lane, rnd, truck = null) {
    super();
    this.rnd = rnd; this.truck = truck;
    this.addLane(lane);
  }

  /** Grow the path to `until`; s: the car's position (a change on its own lane starts LC.START_MARGIN after it). */
  extend(until, s = null) {
    while (this.end < until) {
      const P = this.last, L = P.lane;
      if (!L.next.length) return;
      const sLo = Math.max(s !== null && s >= P.s0 && s < P.s1 ? P.lo + s - P.s0 + LC.START_MARGIN : P.lo, afterChange(this, P));
      let mv = null;
      if (L.links.size && !P.stay && !changedInto(this, P)) {
        const moves = laneMoves(L, sLo, this.truck);
        if (moves.some((m) => m.src !== L || m.ramp)) mv = pickMove(this.rnd, moves, L);
      }
      if (!mv) {
        let opts = L.next;
        if (this.truck) { const f = opts.filter((o) => this.truck.has(o.lane)); if (f.length) opts = f; }
        const c = opts[Math.floor(this.rnd() * opts.length)];
        mv = { turn: c.turn, src: L, pre: null, conn: c.conn };
      }
      this.addMove(mv, sLo);
    }
  }

  /** The change after lane piece P is dropped: the route goes on along P's lane to its end (and on from there). */
  stay(P) {
    this.dropAfter(P);
    this.setLaneEnd(P, P.lane.len);
    P.stay = true;
  }
}

/** Lane piece P right after a lane change (a ramp's merge): a change out of it starts LC.START_MARGIN in, at least. */
const afterChange = (path, P) => {
  const prev = path.segs[path.segs.indexOf(P) - 1];
  return prev && prev.kind === 'change' ? P.lo + LC.START_MARGIN : 0;
};

/** The lanes trucks may use in a lane graph town: as keisim Town.truck_lanes (wide lanes from which wide lanes go on),
 *  but wide is TRUCK_LANE, not 3.0 m: at the tight bends between streets and 3.0 m residential roads a 7 m truck,
 *  drawn as a box on its path, swings over the centre line onto the car waiting there. null: every lane. */
const TRUCK_LANE = 3.5;
function wideLanes(lanes) {
  const ok = new Set(lanes.filter((L) => L.kind === 'road' && L.width >= TRUCK_LANE - 1e-6));
  if (ok.size === lanes.filter((L) => L.kind === 'road').length) return null;
  for (let changed = true; changed;) {
    changed = false;
    for (const L of ok) if (!L.next.some((o) => ok.has(o.lane))) { ok.delete(L); changed = true; }
  }
  return ok.size ? ok : null;
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
  const graph = lanes.graph || null;                          // KeiSim's lane graph (src/lanes.js)
  const truckLanes = graph ? wideLanes(lanes) : null;
  // the signal phase of every lane that ends at a signalised junction: the head facing it (lane graph: KeiSim's)
  const phaseOf = new Map();
  for (const L of lanes) {
    if (graph) { if (L.signal !== null) phaseOf.set(L.id, L.signal); continue; }
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
  // real towns: unsignalised junctions of 3+ arms are priority crossings (keisim Town.from_graph)
  const priority = data.junctions.some((J) => !J.signalized && J.arms >= 3);
  const mirrored = !!data.mirrored_turn_yield;
  const stopSigns = !!(data.stop_signs && data.stop_signs.length);   // real towns exported with their 止まれ signs
  // lane graph: KeiSim's give-way table comes with the connectors (also the narrow bends of road-type towns)
  const yielding = graph ? graph.conns.some((c) => c.yields.length) : twoPhase || priority;
  if (yielding && !graph) makeYields();
  const roadLanes = lanes.filter((L) => L.len > 30 && L.kind !== 'ramp');
  const totalLen = roadLanes.reduce((a, L) => a + L.len, 0);
  const truckRoadLanes = truckLanes ? roadLanes.filter((L) => truckLanes.has(L)) : [];
  const truckLen = truckRoadLanes.reduce((a, L) => a + L.len, 0);
  const cars = [];
  let contacts = 0, lastContact = -1e9, npcContacts = 0, stopSignStops = 0, laneChanges = 0, rampPassages = 0;
  const touching = new Set();

  /** keisim Town._make_yields: inside every signalised junction, a connector yields to the connectors of the other
   *  approaches green at the same time whose movement ranks higher and whose centre line comes within 2.2 m of its
   *  own (a crossing or a merge): yields [{b, here, there}] (closest points), and wait: the furthest front-bumper
   *  position whose front circle (0.9 m behind the bumper) stays YIELD_CLEAR from all of those paths. Of two equal
   *  movements that cross (the right turns at a skewed junction), one gives way (equalGivesWay).
   *  Priority crossings (unsignalised, 3+ arms; real towns): the minor road's connectors give way to every movement
   *  of the priority road they would touch, whatever the turns; movements of the same road rank as above. With
   *  mirrored_turn_yield a right turn also waits at its entry for the oncoming left turn around the same corner.
   *  Towns with 止まれ signs: every approach of the minor road gets its stop line (lane.stopSign, arc length). */
  function makeYields() {
    const byJunction = new Map();
    for (const L of lanes) {
      const J = junctions.get(L.to);
      if (!J || !(J.signalized || J.arms >= 3)) continue;
      if (!byJunction.has(J.id)) byJunction.set(J.id, []);
      for (const o of L.next) byJunction.get(J.id).push(o.conn);
    }
    for (const [jid, conns] of byJunction) {
      const J = junctions.get(jid);
      const prio = !J.signalized;
      const major = prio ? priorityGroups(J, conns) : null;
      if (prio && stopSigns) {
        for (const [id, grp] of major) if (grp === 1) lanes[id].stopSign = Math.max(0.5, lanes[id].len - STOP_SIGN_SETBACK);
      }
      const equals = [];
      for (const a of conns) {
        const ga = phaseOf.get(a.from.id);
        for (const b of conns) {
          const gb = phaseOf.get(b.from.id);
          if (b.from === a.from) continue;
          // priority crossing, a and b from different roads: the minor one gives way whatever the turns
          const minor = prio && major.get(a.from.id) !== major.get(b.from.id);
          if (minor && major.get(a.from.id) === 0) continue;
          if (!prio && (ga === undefined || ga !== gb)) continue;
          const equal = RANK[a.turn] === RANK[b.turn];      // equal pairs once each, in keisim's order
          if (!minor && (RANK[a.turn] > RANK[b.turn] || (equal && (a.from.id > b.from.id || (a.from.id === b.from.id && a.to.id > b.to.id))))) continue;
          let best = Infinity, here = 0, there = 0;
          for (let i = 0; i < a.pts.length; i++) {
            for (let j = 0; j < b.pts.length; j++) {
              const d = Math.hypot(a.pts[i][0] - b.pts[j][0], a.pts[i][1] - b.pts[j][1]);
              if (d < best) { best = d; here = a.cum[i]; there = b.cum[j]; }
            }
          }
          if (best >= 2.2) continue;
          if (equal && !minor) equals.push({ a, b, here, there });
          else a.yields.push({ b, here, there });
        }
      }
      // approaches each connector gives way to by rank (taken before the equal pairs: order-independent)
      const ranked = new Map(conns.map((c) => [c, new Set(c.yields.map((y) => y.b.from.id))]));
      for (const { a, b, here, there } of equals) {
        if (equalGivesWay(J, a, b, ranked)) a.yields.push({ b, here, there });
        else b.yields.push({ b: a, here: there, there: here });
      }
      if (mirrored && J.arms >= 3) {
        // 道路交通法: a right turn does not get in the way of the oncoming left turn around the same corner; on 3 m
        // lanes their centre lines run within MIRROR_PASS, so it waits at its entry until that turn has cleared
        for (const a of conns) {
          if (a.turn !== 'right') continue;
          for (const b of conns) {
            if (b.turn !== 'left' || b.from === a.from || a.yields.some((y) => y.b === b)) continue;
            if (b.from.road !== a.to.road || b.to.road !== a.from.road) continue;   // not the same corner the other way round
            let best = Infinity;
            for (const p of a.pts) for (const q of b.pts) best = Math.min(best, Math.hypot(p[0] - q[0], p[1] - q[1]));
            if (best < MIRROR_PASS) a.yields.push({ b, here: 0, there: b.len });
          }
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

  /** keisim Town._priority_groups: approach lane id -> 0 on the priority road, 1 on the minor road(s). The priority
   *  road is a pair of nearly opposite arms (the through road of a T-junction); with a choice, the pair of the higher
   *  road kind, then the longer one. Every arm counts for the pair, also one that only leaves the junction (the
   *  start of a one-way road); only the approaches are in the map. */
  function priorityGroups(J, conns) {
    const arms = [];
    for (const c of conns) {
      if (arms.some((a) => a.L === c.from)) continue;
      const L = c.from, p = L.pts[L.pts.length - 1], r = data.roads[L.road] || {};
      arms.push({ L, ang: Math.atan2(p[1] - J.pos[1], p[0] - J.pos[0]), rank: KIND_RANK[r.kind || 'street'], len: r.length || L.len });
    }
    for (const L of lanes) {
      if (L.from !== J.id || arms.some((a) => a.L.road === L.road)) continue;
      const p = L.pts[0], r = data.roads[L.road] || {};
      arms.push({ L, out: true, ang: Math.atan2(p[1] - J.pos[1], p[0] - J.pos[0]), rank: KIND_RANK[r.kind || 'street'], len: r.length || L.len });
    }
    const greater = (u, v) => { for (let n = 0; n < u.length; n++) if (u[n] !== v[n]) return u[n] > v[n]; return false; };
    let best = null, pair = [0, 1];
    for (let i = 0; i < arms.length; i++) {
      for (let k = i + 1; k < arms.length; k++) {
        const opp = Math.abs(wrap(arms[i].ang - arms[k].ang));
        const key = [opp > Math.PI * 2 / 3 ? 1 : 0, Math.min(arms[i].rank, arms[k].rank), arms[i].rank + arms[k].rank, arms[i].len + arms[k].len, opp];
        if (!best || greater(key, best)) { best = key; pair = [i, k]; }
      }
    }
    return new Map(arms.map((a, n) => [a, pair.includes(n) ? 0 : 1]).filter(([a]) => !a.out).map(([a, g]) => [a.L.id, g]));
  }

  /** keisim Town._equal_gives_way: does connector a give way to b? If one of them already gives way to the other's
   *  approach, that one (the cars queue in one lane: the other order deadlocks); else the one from the minor road
   *  (road types); else the one with the other approach on its left (左方優先); else the lower approach lane. */
  function equalGivesWay(J, a, b, ranked) {
    const ra = ranked.get(a), rb = ranked.get(b);
    if (rb.has(a.from.id) !== ra.has(b.from.id)) return ra.has(b.from.id);
    const kindOf = (L) => KIND_RANK[(data.roads[L.road] && data.roads[L.road].kind) || 'street'];
    const ka = kindOf(a.from), kb = kindOf(b.from);
    if (ka !== kb) return ka < kb;
    const h = a.from.h1, vx = b.pts[0][0] - J.pos[0], vy = b.pts[0][1] - J.pos[1];
    let side = (Math.cos(h) * vy - Math.sin(h) * vx) / Math.max(Math.hypot(vx, vy), 1e-9);
    if (data.cfg.left_hand_traffic === false) side = -side;
    if (Math.abs(side) > 0.2) return side > 0;
    return a.from.id < b.from.id;
  }

  function sampleDims() {
    const r = rnd();
    if (r < 0.62) return [4.0 + 0.8 * rnd(), 1.68 + 0.17 * rnd(), 1.4 + 0.2 * rnd(), 0];
    if (r < 0.78) return [3.35 + 0.1 * rnd(), 1.48, 1.6 + 0.2 * rnd(), 0];          // kei car
    if (r < 0.92) return [4.6 + 0.4 * rnd(), 1.8 + 0.15 * rnd(), 1.75 + 0.25 * rnd(), 0];
    const d = [6.0 + 1.6 * rnd(), 2.0 + 0.25 * rnd(), 2.6 + 0.4 * rnd(), 1];           // small truck
    const cap = data.max_vehicle_dims;                                                // real towns: 6 m roads
    if (cap) { d[0] = Math.min(d[0], cap[0]); d[1] = Math.min(d[1], cap[1]); }
    return d;
  }

  /** A free spot on a random lane, away from the ego and the other cars, never right before a stop line. */
  function place(car, ego, avoidEgo) {
    const pool = car.kind === 1 && truckRoadLanes.length ? truckRoadLanes : roadLanes;     // trucks: on their lanes
    const total = pool === roadLanes ? totalLen : truckLen;
    for (let tries = 0; tries < 60; tries++) {
      let x = rnd() * total, L = pool[0];
      for (const l of pool) { if (x < l.len) { L = l; break; } x -= l.len; }
      const hi = phaseOf.has(L.id) ? L.len - STOP_SETBACK - 20 : L.len - 4;
      if (hi < 6) continue;
      const s = 4 + rnd() * (hi - 4);
      const route = new Route(L, rnd, car.kind === 1 ? truckLanes : null);
      const p = route.pointAt(s);
      if (ego && Math.hypot(p[0] - ego.x, p[1] - ego.y) < avoidEgo) continue;
      if (cars.some((c) => c !== car && c.xy && Math.hypot(c.xy[0] - p[0], c.xy[1] - p[1]) < 12)) continue;
      route.extend(s + 80, s);
      Object.assign(car, { route, s, seg: route.segAt(s), xy: p, yaw: route.headingAt(s), stuck: 0, ssDone: null, lcp: null });
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
        const nx = c.route.segs[c.seg + 1], s = g.lo + (c.s - g.s0);
        push(onLane, g.lane.id, { s, v: c.v, L: c.L, next: nx && nx.kind === 'conn' ? nx.conn : null });
        if (phaseOf.has(g.lane.id) && s + c.L / 2 > g.lane.len - STOP_SETBACK + 0.5 && c.v > 1.0) {
          push(box, g.lane.to, phaseOf.get(g.lane.id));
        }
      } else if (g.kind === 'change') {                // on both lanes while it changes (mid-lane: no box)
        const a = c.route.localOn(g, c.s), nx = c.route.segs[c.seg + 2];
        push(onLane, g.from.id, { s: a, v: c.v, L: c.L, next: null });
        push(onLane, g.to.id, { s: linkMap(g.from, g.to, a), v: c.v, L: c.L, next: nx && nx.kind === 'conn' ? nx.conn : null });
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
      const vmax = b.speed ? 1.05 * b.speed : VMAX;           // lane graph: the connector's speed limit
      for (const o of occ.onConn.get(b) || []) {
        if (o.s - o.L / 2 > s2 + 1.5) continue;              // its rear is past the crossing
        const d = s2 - (o.s + o.L / 2);
        best = Math.min(best, d < 3 ? 0 : arrival(d, o.v, vmax, o.ego));
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
          best = Math.min(best, arrival(pred.len - front + s2, o.v, vmax, o.ego));
        }
        break;                                                // the cars behind it arrive later
      }
    }
    return best;
  }

  /** The nearest car (or the ego) ahead on the car's path: its centre within 1.9 m of a path point. Lane graph towns
   *  also test the circles at the front and back of every body (half a width in from each end, like keisim's
   *  vehicle_circles): at their big junctions a right turn waits deeper inside, where the left turn from the same
   *  lane passes its rear without coming near its centre. */
  function leader(c, ego) {
    let gap = Infinity, vl = 0;
    const others = cars.filter((o) => o !== c && Math.abs(o.xy[0] - c.xy[0]) < LOOK + 8 && Math.abs(o.xy[1] - c.xy[1]) < LOOK + 8)
      .map((o) => ({ xy: o.xy, yaw: o.yaw, v: o.v, L: o.L, W: o.W }));
    if (ego) others.push({ xy: [ego.x, ego.y], yaw: ego.yaw, v: ego.v, L: EGO.L, W: EGO.W });
    if (!others.length) return { gap, vl };
    if (graph) {
      for (const o of others) {
        const e = o.L / 2 - o.W / 2, ux = Math.cos(o.yaw) * e, uy = Math.sin(o.yaw) * e;
        o.ends = [[o.xy[0] - ux, o.xy[1] - uy], [o.xy[0] + ux, o.xy[1] + uy]];
      }
    }
    for (let a = 1.0; a <= LOOK; a += 1.5) {
      const p = c.route.pointAt(c.s + a);
      for (const o of others) {
        if (Math.abs(o.xy[0] - p[0]) > 2.6 + (o.ends ? o.L / 2 : 0) || Math.abs(o.xy[1] - p[1]) > 2.6 + (o.ends ? o.L / 2 : 0)) continue;
        let g = Infinity;
        if (Math.hypot(o.xy[0] - p[0], o.xy[1] - p[1]) < 1.9) g = a - c.L / 2 - o.L / 2;
        else if (o.ends && o.ends.some((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1.9)) g = a - c.L / 2 - o.W / 2;
        if (g < gap) {
          gap = g;
          vl = o.v * Math.cos(wrap(o.yaw - c.route.headingAt(c.s + a)));
        }
      }
      if (gap < Infinity && a > gap + 8) break;
    }
    return { gap, vl };
  }

  /** Lane graph: the cars on every lane, {c, s (lane-local centre), v, L}: on a piece of it, changing into or out of
   *  it, or on a connector into it (s < 0: just out of the junction behind). */
  function byLane() {
    const m = new Map();
    for (const c of cars) {
      const g = c.route.segs[c.seg];
      if (g.kind === 'lane') push(m, g.lane.id, { c, s: g.lo + c.s - g.s0, v: c.v, L: c.L });
      else if (g.kind === 'change') {
        const a = c.route.localOn(g, c.s);
        push(m, g.from.id, { c, s: a, v: c.v, L: c.L });
        push(m, g.to.id, { c, s: linkMap(g.from, g.to, a), v: c.v, L: c.L });
      } else push(m, g.conn.to.id, { c, s: c.s - g.s0 - g.conn.len, v: c.v, L: c.L });
    }
    return m;
  }

  /** keisim Traffic._lane_change_gaps: a car about to change lanes needs room in the target lane (changeRoom; the ego
   *  counts too), checked from max(LC.CHECK, its stopping distance) before the start. Without room the start moves on
   *  (LC.PUSH, while the window lasts) and the car falls in behind the car in the way ahead of it; at the window's end
   *  it stops 1 m short of the start and waits. The cars in the target lane leave a gap to a car waiting there (ファス
   *  ナー合流) and follow a car once it has begun to change into their lane. A car that has waited LC.GIVE_UP s, or comes
   *  to its start too fast without room, stays in its lane (not on a ramp, which ends: it waits where it is).
   *  Returns Map(car -> {gap, vl}): virtual leaders for the IDM. */
  function laneChangeGaps(dt, ego) {
    const out = new Map(), waiting = [];
    let idx = null;
    const index = () => idx || (idx = byLane());
    const setGap = (c, g, v) => { const o = out.get(c); if (!o || g < o.gap) out.set(c, { gap: g, vl: v }); };
    const laneCars = (B, exclude) => {
      const list = (index().get(B.id) || []).filter((o) => o.c !== exclude);
      if (ego) {
        const pr = projectOn(B, [ego.x, ego.y]);
        if (Math.abs(pr.lat) < B.width / 2 + 0.3 && pr.s < B.len && Math.cos(ego.yaw - laneHeading(B, pr.s)) > 0.5) {
          list.push({ c: null, s: pr.s, v: ego.v, L: EGO.L });
        }
      }
      return list;
    };
    for (const c of cars) {
      const R = c.route, k = c.seg, g = R.segs[k];
      let P = null, ch = null;
      if (g.kind === 'lane' && R.segs[k + 1] && R.segs[k + 1].kind === 'change') { P = g; ch = R.segs[k + 1]; }
      else if (g.kind === 'change' && R.segs[k - 1] && R.segs[k - 1].kind === 'lane' &&
               !(c.lcp && c.lcp.go && c.lcp.from === g.from && c.lcp.to === g.to)) {
        P = R.segs[k - 1]; ch = g;                       // in a change it has not started (too fast, or no plan yet)
      }
      if (!ch) { if (g.kind !== 'change') c.lcp = null; continue; }
      const A = P.lane, B = ch.to, local = ch === g ? R.localOn(g, c.s) : P.lo + c.s - P.s0;
      let p = c.lcp;
      if (!p || p.from !== A || p.to !== B) {           // the plan, made when first needed (keisim Traffic._lc_plan)
        let w = changeWindow(A, local + 2, B);
        if (!w && !A.next.length) w = { aMin: local + 2, aMax: local + 2, lc: Math.max(5, Math.min(LC.MIN, A.len - local - 2)) };
        if (!w) { R.stay(P); c.lcp = null; c.seg = R.segAt(c.s); continue; }     // too late: on along its lane
                                                                                  // (a ramp ends: it merges where it is)
        p = c.lcp = { from: A, to: B, aMin: w.aMin, aMax: w.aMax, lc: w.lc, go: false, wait: 0 };
        ch = R.retime(ch, w.aMin, w.lc, w.aMin, w.aMax);
      }
      if (p.go) continue;                                // under way
      const d = ch.a0 - local;
      const deadEnd = !A.next.length;                    // a ramp: it has to merge
      if (d <= 0.3 || (p.wait > LC.GIVE_UP && !deadEnd)) {
        if (ch.a0 < p.aMax && d <= 0.3) R.retime(ch, Math.min(p.aMax, local + LC.PUSH_TO), p.lc, p.aMin, p.aMax);
        else if (deadEnd) { p.aMax = local + 0.5; R.retime(ch, p.aMax, p.lc, p.aMin, p.aMax); }
        else { R.stay(P); c.lcp = null; }
        c.seg = R.segAt(c.s);
        continue;
      }
      if (d > Math.max(LC.CHECK, c.v * c.v / 6 + 3)) continue;   // not yet (checked from where it could stop)
      const sE = linkMap(A, B, local);
      const bad = laneCars(B, c).filter((o) => !changeRoom(o.s - sE, o.v, o.L, c.v, c.L));
      if (!bad.length) {
        if (d < 1.5) { p.go = true; if (A.kind === 'ramp') rampPassages++; else if (B.kind !== 'ramp') laneChanges++; }
        continue;
      }
      if (d < LC.PUSH && ch.a0 < p.aMax) ch = R.retime(ch, Math.min(p.aMax, local + LC.PUSH_TO), p.lc, p.aMin, p.aMax);
      for (const o of bad) {
        const dx = o.s - sE;
        if (dx > 0) setGap(c, Math.max(dx - (c.L + o.L) / 2, 0.05), o.v);
      }
      if (ch.a0 >= p.aMax - 1e-6) {
        setGap(c, ch.a0 - 1 - local + IDM.s0, 0);        // the IDM stands still s0 behind the gap: 1 m before the start
        if (ch.a0 - local < 4 && c.v < 1) { waiting.push({ c, B, s: sE, L: c.L }); p.wait += dt; }
      }
    }
    for (const w of waiting) {                           // a gap for the car waiting to change in
      for (const o of index().get(w.B.id) || []) {
        const g = w.s - o.s - (o.L + w.L) / 2;
        if (o.c !== w.c && g > 1.0) setGap(o.c, g, 0);  // fully behind it (the ones beside it go on)
      }
    }
    for (const c of cars) {                              // followers of a car that has begun to change into their lane
      const g = c.route.segs[c.seg];
      if (g.kind !== 'change') continue;
      const sB = linkMap(g.from, g.to, c.route.localOn(g, c.s));
      for (const o of index().get(g.to.id) || []) {
        if (o.c !== c && o.s < sB) setGap(o.c, Math.max(sB - o.s - (o.L + c.L) / 2, 0.05), c.v);
      }
    }
    return out;
  }

  /** keisim Traffic._speed_limits: the speed limit where the car is, eased down at 1.5 m/s^2 toward lower limits up
   *  to 40 m ahead (front bumper), so that it enters a slower road at that road's limit (lane graph towns). */
  function speedLimit(c) {
    const segs = c.route.segs, sp = (g) => (g.kind === 'lane' ? g.lane.speed : g.kind === 'conn' ? g.conn.speed : g.from.speed);
    let v = sp(segs[c.seg]), d = segs[c.seg].s1 - c.s - c.L / 2;
    for (let j = c.seg + 1; j < segs.length && d <= 40; j++) {
      const g = segs[j];
      if (g.kind !== 'change' && sp(g) < v) v = Math.min(v, Math.sqrt(sp(g) ** 2 + 3 * Math.max(d, 0)));
      d += g.s1 - g.s0;
    }
    return v;
  }

  function step(dt, t, ego, egoNav) {
    const occ = occupancy(ego, egoNav);
    const lcg = graph ? laneChangeGaps(dt, ego) : null;
    for (const c of cars) {
      const k = c.seg, g = c.route.segs[k], segs = c.route.segs;
      const v0 = graph ? c.vf * speedLimit(c) : 10 * c.vf;
      // curve speed envelope over the next 30 m (keisim: sqrt(vc^2 + 4 d))
      let vmax = v0;
      for (let a = 2; a <= 30; a += 4) {
        const dh = Math.abs(wrap(c.route.headingAt(c.s + a + 2) - c.route.headingAt(c.s + a - 2)));
        if (dh > 1e-3) vmax = Math.min(vmax, Math.sqrt(LAT_ACC / (dh / 4) + 4 * a));
      }
      let { gap, vl } = leader(c, ego);
      if (lcg && lcg.has(c)) { const x = lcg.get(c); if (x.gap < gap) { gap = x.gap; vl = x.vl; } }
      // the approach: the lane piece before the next junction, the car's own or (lane changes) one ahead of it
      let apK = g.kind === 'lane' ? k : g.kind === 'change' ? k + 1 : -1;
      while (apK >= 0 && segs[apK + 1] && segs[apK + 1].kind === 'change') apK += 2;
      const ap = apK >= 0 && segs[apK] && segs[apK].kind === 'lane' && segs[apK].hi >= segs[apK].lane.len - 1e-6 ? segs[apK] : null;
      // stop line: red / yellow (if it can stop), or the box rule / a full exit at green
      if (ap && phaseOf.has(ap.lane.id)) {
        const d = ap.s0 + (ap.lane.len - ap.lo) - STOP_SETBACK - (c.s + c.L / 2);
        if (d > -0.5 && d < LOOK) {
          const J = junctions.get(ap.lane.to);
          const ph = phaseOf.get(ap.lane.id);
          const sig = signalState(J, ph, t);
          let stop = sig === TL.RED || (sig === TL.YELLOW && d > c.v * c.v / 7 + 1);
          if (!stop && d < 12) {
            const inBox = occ.box.get(J.id);
            if (inBox && inBox.some((p) => p !== ph)) stop = true;          // a car of another phase is inside
            const exit = segs[apK + 2];
            if (!stop && exit && exit.kind === 'lane') {
              const q = occ.onLane.get(exit.lane.id) || [];
              if (q.some((o) => o.s - o.L / 2 < c.L + 3 && o.v < 2)) stop = true;
            }
          }
          if (stop && d - 0.6 < gap) { gap = d - 0.6; vl = 0; }
        }
      }
      // 止まれ: on the minor road's approach (or the lane before it), stop once with the front bumper STOP_SIGN_NPC_AT
      // before the line, then on to the connector's wait point (keisim Traffic._stop_sign_gaps)
      if (stopSigns) {
        for (const sg of [g, segs[k + 1]]) {
          if (!sg) break;
          if (sg.kind !== 'lane' || sg.lane.stopSign == null) continue;
          const d = sg.s0 + (sg.lane.stopSign - sg.lo) - (c.s + c.L / 2);
          if (c.ssDone === sg.lane.id || d < -0.5 || d > LOOK) break;
          if (c.v < 0.05 && d <= STOP_SIGN_NPC_ZONE) { c.ssDone = sg.lane.id; stopSignStops++; break; }
          const gs = d + IDM.s0 - STOP_SIGN_NPC_AT;          // the IDM stands still s0 behind the gap
          if (gs < gap) { gap = gs; vl = 0; }
          break;
        }
      }
      // turns that give way (two-phase signals, priority crossings): wait at the wait point while priority traffic
      // would arrive within YIELD_GAP s; past the wait point, or too close to stop gently, the movement goes on
      // (keisim Traffic._yield_gaps)
      if (yielding) {
        let yc = null, dW = 0;
        if (g.kind === 'conn') {
          if (g.conn.yields.length) { yc = g.conn; dW = g.conn.wait - (c.s - g.s0 + c.L / 2); }
        } else if (ap) {
          const nx = segs[apK + 1];
          if (nx && nx.kind === 'conn' && nx.conn.yields.length) { yc = nx.conn; dW = ap.s1 - (c.s + c.L / 2) + nx.conn.wait; }
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
      c.route.extend(c.s + 80, c.s);
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
    contacts = 0; npcContacts = 0; stopSignStops = 0; laneChanges = 0; rampPassages = 0; touching.clear();
  }

  return {
    step, drawList, reset, cars, twoPhase, priority,
    get contacts() { return contacts; }, get npcContacts() { return npcContacts; }, get n() { return cars.length; },
    get stopSignStops() { return stopSignStops; },             // full stops at 止まれ lines (tests)
    // lane graph towns (tests): lane changes begun between the lanes of a road, and merges out of a ramp
    get laneChanges() { return graph ? laneChanges : undefined; }, get rampPassages() { return graph ? rampPassages : undefined; },
  };
}
