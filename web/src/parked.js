// Parked cars (路上駐車) for the in-browser demo (index.html?pilot=…&parked=N): keisim/parked.py's placement in
// JavaScript. Cars stand still at the left kerb of two-way roads with one lane each way (not main roads), away from
// the ends of the lane, from junctions of 3+ arms (through bends, but tight bends whose connectors give way to each
// other count as junctions), from curves, from the other kerb's parked cars, and from the ego's start, only where a
// wide car can get by. Cars of one kerb less than GROUP_GAP apart (rows of two) are passed in one go (groups). The
// surrounding cars pass them (src/traffic.js) and the riders move out round them (src/bicycles.js). There are no
// pedestrians in the browser demo, so no child darts out from in front of them (keisim's dart_out). Off by default:
// without them nothing here runs and no other random number changes. All coordinates are KeiSim's (x east, y north,
// yaw counter-clockwise).
import { laneHeading, lanePoint, projectOn } from './lanes.js';

// keisim/parked.py PK_*
export const PK = {
  EDGE: [0.2, 0.45], FROM: [30, 30], FROM_JUNCTION: 60, EXIT: 20, CURVE: 1 / 60, CURVE_REACH: 15, SPACING: 45,
  ROW: 0.3, ROW_GAP: [0.8, 2.5], GROUP_GAP: 6, OPPOSITE: 70, EGO: 30, CLEAR: 0.5, PASSABLE: 0.3, WIDE_HW: 0.95,
  MARGIN: 0.15, BIKE_CLEAR: 0.4, KEEP: 0.15, IN: [16, 5], OUT: [1, 8], DECIDE: 22, WAIT: 5, ONCOMING_T: 2.5,
  YIELD_B: 2.5, YIELD_D: 4, ZONE: [18, 10], PASS_V: 20 / 3.6, SWING_V: 3, RATE: 1.2, ACC: 2, BACK: 0.25, CONVOY: 3, CONVOY_GAP: 12,
  CONVOY_ROOM: 2, GIVE_ROOM: 6,
};
export const PARKED_ID = 100000;              // drawList ids of the parked cars: this + their index
// keisim/traffic.py CAR_COLORS as RGB (as src/traffic.js)
const COLORS = [[235, 235, 235], [238, 240, 240], [195, 190, 190], [32, 30, 30], [125, 120, 120], [25, 60, 140],
  [170, 40, 40], [190, 110, 30], [50, 70, 70], [90, 140, 180], [60, 130, 60]];

/**
 * opts: {perKm: parked cars per km of the lanes they may stand beside, seed, max}. Returns {cars, groups, laneGroups,
 * oppGroups, n, place(egoXY), width(L), circles(out, margin), drawList(cx, cy, r)}. cars: [{x, y, yaw, L, W, H, kind,
 * rgb, lane, s, side}] (side: the centre's offset from the lane's centre line, + to the right); groups: [{id, lane,
 * cars, sLo, sHi, q (how far right of the lane's centre line their side towards the traffic is), opp (the other
 * direction's lane or null), oppZone: [lo, hi] of the passing stretch on it}].
 */
export function createParked(data, lanes, { perKm = 0, seed = 1, max = 60 } = {}) {
  let st = (seed * 2246822519) >>> 0 || 1;
  const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  const uni = (a, b) => a + (b - a) * rnd();
  const cfg = data.cfg;
  const graph = lanes.graph || null;
  const junctions = new Map(data.junctions.map((j) => [j.id, j]));
  const roadOf = (L) => data.roads[L.road] || {};
  const laneW = (L) => L.width ?? roadOf(L).lane_width ?? cfg.lane_width ?? 3.5;
  // the connectors of every junction (a tight bend: one of them gives way to another)
  const connsAt = new Map();
  for (const L of lanes) for (const o of L.next) {
    if (!connsAt.has(o.conn.junction)) connsAt.set(o.conn.junction, []);
    connsAt.get(o.conn.junction).push(o.conn);
  }
  // the other direction's lane of every two-way road with one lane each way
  const byRoad = new Map();
  for (const L of lanes) {
    if (L.kind !== 'road') continue;
    if (!byRoad.has(L.road)) byRoad.set(L.road, []);
    byRoad.get(L.road).push(L);
  }
  const opposite = (L) => {
    const o = (byRoad.get(L.road) || []).filter((M) => M !== L && M.from === L.to && M.to === L.from);
    return o.length === 1 ? o[0] : null;
  };

  /** keisim parked._to_junction: metres from L's end (ahead) or start (behind) to the next junction of 3+ arms that
   *  way, through bends, but for bends whose connectors give way to each other; limit or more: limit. */
  function toJunction(L, ahead, limit = PK.FROM_JUNCTION) {
    let d = 0;
    for (let k = 0; k < 8; k++) {
      const J = junctions.get(ahead ? L.to : L.from);
      if (!J) return limit;
      if (J.arms !== 2) return J.arms >= 3 ? d : limit;
      if ((connsAt.get(J.id) || []).some((c) => c.yields && c.yields.length)) return d;
      const C = ahead ? (L.next.length === 1 ? L.next[0].conn : null) : (L.pred.length === 1 ? L.pred[0] : null);
      if (!C) return d;
      L = ahead ? C.to : C.from;
      d += C.len + L.len;
      if (d >= limit) return limit;
    }
    return d;
  }

  /** keisim parked.parked_span: [from, to] on L where parked cars may stand. */
  const span = (L) => {
    const a = Math.max(PK.FROM[0], PK.FROM_JUNCTION + PK.ZONE[0] - toJunction(L, false, PK.FROM_JUNCTION + PK.ZONE[0]));
    const b = Math.max(PK.FROM[1], PK.FROM_JUNCTION - toJunction(L, true));
    return [a, L.len - b];
  };

  /** keisim parked.parked_lanes: road lanes of two-way roads with one lane each way, not main roads, long enough. */
  const candidates = cfg.left_hand_traffic === false ? [] : lanes.filter((L) => {
    if (L.kind !== 'road') return false;
    const r = roadOf(L);
    if (r.oneway || r.kind === 'main' || (L.n ?? 1) !== 1 || (r.median ?? 0) > 0 || !opposite(L)) return false;
    const [a, b] = span(L);
    return b - a >= 8;
  });
  const total = candidates.reduce((a, L) => a + L.len, 0);
  const count = Math.min(max, Math.round(perKm * total / 1000));
  const trucks = graph && graph.truck ? graph.truck : null;

  const cars = [], groups = [], laneGroups = new Map(), oppGroups = new Map();

  function sampleDims(truckOk) {
    const r = rnd();
    if (r < 0.42) return [uni(3.35, 3.45), 1.48, uni(1.55, 1.8), 0];                    // kei car
    if (r < 0.78) return [uni(4.0, 4.8), uni(1.68, 1.82), uni(1.4, 1.6), 0];
    if (r < 0.92 || !truckOk) return [uni(4.6, 5.0), uni(1.75, 1.9), uni(1.75, 2.0), 0];
    return [uni(4.7, 6.0), uni(1.85, 2.0), uni(2.2, 2.6), 1];                          // delivery truck
  }

  const curvAt = (L, s) => {                       // |heading change| per metre over 2 m
    const d = laneHeading(L, s + 1) - laneHeading(L, s - 1);
    return Math.abs(Math.atan2(Math.sin(d), Math.cos(d))) / 2;
  };

  function tryAdd(L, s, dims, row, avoid) {
    const [Lc, W, H, kind] = dims;
    for (let x = Math.max(1, s - Lc / 2 - PK.CURVE_REACH); x <= Math.min(L.len - 1, s + Lc / 2 + PK.CURVE_REACH); x += 1) {
      if (curvAt(L, x) > PK.CURVE) return false;
    }
    for (const c of cars) {                          // this kerb: SPACING apart (a row: touching)
      if (c.lane !== L) continue;
      if (Math.abs(c.s - s) - (Lc + c.L) / 2 < (row ? 0.5 : PK.SPACING)) return false;
    }
    const r = roadOf(L), lw = laneW(L);
    const half = r.half_width ?? cfg.road_half_width ?? lw + (r.shoulder ?? cfg.shoulder ?? 0.25);
    const offC = half - uni(PK.EDGE[0], PK.EDGE[1]) - W / 2;       // the car's centre from the road's centre line
    const side = -(offC - lw / 2);                                    // from the lane's centre, + to the right
    if (side + W / 2 + PK.PASSABLE + 2 * PK.WIDE_HW + 0.1 > lw) return false;   // a wide car could not get by
    const p = lanePoint(L, s), h = laneHeading(L, s);
    const x = p[0] + Math.sin(h) * side, y = p[1] - Math.cos(h) * side;
    const O = opposite(L);
    if (O) {                                         // the other kerb: not across the road from it
      const so = projectOn(O, [x, y]).s;
      if (cars.some((c) => c.lane === O && Math.abs(c.s - so) < PK.OPPOSITE)) return false;
    }
    if (avoid && Math.hypot(x - avoid[0], y - avoid[1]) < PK.EGO) return false;
    const base = COLORS[Math.floor(rnd() * COLORS.length)], k = uni(0.9, 1.08);
    const rgb = base.map((c) => Math.max(0, Math.min(255, c * k + uni(-8, 8))));
    cars.push({ id: cars.length, x, y, yaw: h, L: Lc, W, H, kind, rgb, lane: L, s, side, opp: O });
    return true;
  }

  /** keisim ParkedCars._index: groups, and the passing stretch on the other direction's lane. */
  function index() {
    groups.length = 0; laneGroups.clear(); oppGroups.clear();
    const byLane = new Map();
    for (const c of [...cars].sort((a, b) => a.s - b.s)) {
      if (!byLane.has(c.lane)) byLane.set(c.lane, []);
      byLane.get(c.lane).push(c);
    }
    for (const [L, cs] of byLane) {
      const gs = [];
      for (const c of cs) {
        const lo = c.s - c.L / 2, hi = c.s + c.L / 2, q = c.side + c.W / 2;
        const G = gs.length ? groups[gs[gs.length - 1]] : null;
        if (G && lo - G.sHi < PK.GROUP_GAP) {
          G.cars.push(c); G.sHi = Math.max(G.sHi, hi); G.q = Math.max(G.q, q);
        } else {
          gs.push(groups.length);
          groups.push({ id: groups.length, lane: L, cars: [c], sLo: lo, sHi: hi, q, opp: c.opp, oppZone: null });
        }
        c.group = gs[gs.length - 1];
      }
      laneGroups.set(L.id, gs);
    }
    for (const G of groups) {
      if (!G.opp) continue;
      const L = G.lane, O = G.opp;
      const z0 = Math.max(G.sLo - PK.ZONE[0], 0), z1 = Math.min(G.sHi + PK.ZONE[1], L.len);
      const a = projectOn(O, lanePoint(L, z1)).s, b = projectOn(O, lanePoint(L, z0)).s;
      G.oppZone = [Math.min(a, b), Math.max(a, b)];
      if (!oppGroups.has(O.id)) oppGroups.set(O.id, []);
      oppGroups.get(O.id).push(G.id);
    }
  }

  /** keisim ParkedCars.__init__ (without the route's weighting): place `count` cars, away from egoXY. */
  function place(egoXY = null) {
    cars.length = 0;
    st = (seed * 2246822519) >>> 0 || 1;           // the same cars every time for one seed
    if (!candidates.length || count <= 0) { index(); return; }
    let tries = 0;
    while (cars.length < count && tries < count * 30) {
      tries++;
      let x = rnd() * total, L = candidates[0];
      for (const l of candidates) { if (x < l.len) { L = l; break; } x -= l.len; }
      const dims = sampleDims(!trucks || trucks.has(L));
      const hl = dims[0] / 2, [a, b] = span(L);
      if (b - a < 2 * hl) continue;
      const s = uni(a + hl, b - hl);
      if (!tryAdd(L, s, dims, false, egoXY)) continue;
      if (rnd() < PK.ROW && cars.length < count) {     // another one right behind it
        const d2 = sampleDims(!trucks || trucks.has(L));
        const s2 = s - hl - uni(PK.ROW_GAP[0], PK.ROW_GAP[1]) - d2[0] / 2;
        if (s2 - d2[0] / 2 >= a) tryAdd(L, s2, d2, true, egoXY);
      }
    }
    index();
  }

  /** Corridor circles of every parked car within r of (cx, cy) (src/bicycles.js carCircles' shape: three along it
   *  and a small one in each corner), into `out`. */
  function circles(out, margin, cx = null, cy = null, r = Infinity) {
    for (const c of cars) {
      if (cx !== null && (Math.abs(c.x - cx) > r || Math.abs(c.y - cy) > r)) continue;
      const e = c.L / 2 - c.W / 2, co = Math.cos(c.yaw), si = Math.sin(c.yaw);
      for (let k = -1; k <= 1; k++) out.push({ x: c.x + co * e * k, y: c.y + si * e * k, r: c.W / 2 * 1.05, v: 0, yaw: c.yaw, own: c, car: true, parked: true, m: margin });
      const a = c.L / 2 - 0.15, b = c.W / 2 - 0.15;
      for (let k = -1; k <= 1; k += 2) for (let j = -1; j <= 1; j += 2)
        out.push({ x: c.x + co * a * k - si * b * j, y: c.y + si * a * k + co * b * j, r: 0.15, v: 0, yaw: c.yaw, own: c, car: true, parked: true, m: margin });
    }
    return out;
  }

  /** actors.js rows [id, x, y, yaw, L, W, H, r, g, b, kind, brake (off)] for the parked cars within r of (cx, cy). */
  function drawList(cx, cy, r = 140) {
    const out = [];
    for (const c of cars) {
      if (Math.abs(c.x - cx) > r || Math.abs(c.y - cy) > r) continue;
      out.push([PARKED_ID + c.id, c.x, c.y, c.yaw, c.L, c.W, c.H, c.rgb[0], c.rgb[1], c.rgb[2], c.kind, 0]);
    }
    return out;
  }

  index();
  return { cars, groups, laneGroups, oppGroups, place, circles, drawList, width: laneW, candidates,
    get n() { return cars.length; } };
}
