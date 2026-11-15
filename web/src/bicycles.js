// Bicycles for the in-browser demo (index.html?pilot=…&bikes=N): KeiSim's riders (keisim/bicycles.py) in JavaScript,
// stepped with the surrounding cars of src/traffic.js and drawn by src/world/bikes.js. Each rider keeps to the left
// edge of the road (BIKE_EDGE from it), goes straight on or left at junctions, stops at red / yellow, at 止まれ lines
// and where its turn gives way, follows whoever is ahead on its line (cars, the ego, other riders), filters slowly
// past cars standing close beside it and moves out round a slower rider after a look over the shoulder. The speed
// comes from forces (what the legs give, rolling and air drag, the brakes), with gears, coasting, a foot on the
// ground at a stop, the lean in bends and the wobble at low speed, as in KeiSim. There are no pedestrians in the
// browser demo, so the riders' rules for zebras and people are left out. Off by default: without riders nothing here
// runs and the cars' random numbers are untouched. All coordinates are KeiSim's (x east, y north, yaw
// counter-clockwise).
import { wrap } from './nav.js';

export const BIKE_LEN = 1.8, BIKE_W = 0.6, BIKE_R = 0.3;
const NCB = 3;                    // circles along a bike
const BIKE_EDGE = 0.5;            // m from the road's edge to the rider's line
const G = 9.81, RHO = 1.2, WHEELBASE = 1.08;
const B_MAX = 4.5, TAU_V = 1.1;
const OUT = 0.85, OUT_RATE = 0.7, OUT_BLEND = 6.0;
const MARGIN_CAR = 0.2, MARGIN_BIKE = 0.25, MARGIN_TURN = 0.15;
const FILTER_SIDE = 0.9, FILTER_DV = 2.5;
const CORNER_R = 0.15;
const MARGIN_SINGLE = 1.5;                  // to a car the rider stays behind (same right turn, a fold in the lane)
const EGO_WAIT = 3;                         // a rider and the standing ego beside it: the rider goes after this (s), for 6 s
export const SINGLE_AHEAD = 0.3;            // the car goes first when its centre is this far ahead of the rider's (m)
const S0_BIKE = 0.8, S0_CAR = 1.4, T_HEAD = 0.9, LOOK_BACK = 0.6, STUCK_S = 45.0;
const SIG = 0, SIGN = 1, YIELD = 2;
const PEDAL = 0, COAST = 1, BRAKE = 2;
const P = 48;                     // points of a rider's lookahead: 24 every 0.5 m, then 24 every 1 m
const REL = Array.from({ length: P }, (_, i) => (i < 24 ? 0.5 * i : 12 + (i - 24)));
// keisim/bicycles.py TYPES: city bike, sport bike, slow rider
const TYPES = [
  { p: 0.62, cruise: [3.3, 4.7], a0: [0.9, 1.3], pmax: [160, 240], mass: 88, b: 1.4, cda: 0.60, crr: 0.008,
    gears: [3.6, 4.8, 6.2], cad: [0.75, 1.15], wheel: 0.33, alat: [1.2, 1.7] },
  { p: 0.22, cruise: [5.0, 6.6], a0: [1.2, 1.7], pmax: [300, 420], mass: 80, b: 1.8, cda: 0.42, crr: 0.005,
    gears: [2.9, 3.7, 4.6, 5.7, 7.0], cad: [1.15, 1.6], wheel: 0.34, alat: [1.7, 2.4] },
  { p: 0.16, cruise: [2.5, 3.3], a0: [0.6, 0.9], pmax: [90, 140], mass: 85, b: 1.1, cda: 0.62, crr: 0.009,
    gears: [3.6, 4.8], cad: [0.6, 1.0], wheel: 0.33, alat: [0.9, 1.3] },
];
// colours as R, G, B (keisim's are B, G, R)
const FRAMES = [[205, 200, 200], [38, 35, 35], [235, 235, 235], [190, 40, 40], [30, 60, 110], [120, 200, 190],
  [220, 190, 40], [50, 110, 60], [170, 90, 150], [230, 140, 90]];
const SPORTS = [[38, 35, 35], [200, 40, 40], [235, 235, 235], [30, 120, 200], [230, 200, 60], [60, 170, 70]];
const SHIRTS = [[40, 60, 200], [200, 40, 40], [40, 160, 40], [230, 230, 230], [30, 30, 30], [230, 140, 0],
  [150, 50, 150], [220, 200, 60], [120, 120, 120], [80, 150, 200]];
const PANTS = [[25, 30, 40], [30, 60, 90], [30, 30, 30], [100, 110, 120], [110, 50, 60], [150, 150, 150]];
const SKIN = [[220, 180, 150], [205, 160, 120], [170, 125, 90], [130, 90, 60]];

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

/**
 * Circles for the corridor checks: a car's three (keisim vehicle_circles) and a small one in each corner of its box,
 * which the three leave out (0.3 m of a car that stands at an angle to the rider's line), into `out` as
 * {x, y, r, v, yaw, own}.
 */
export function carCircles(out, x, y, yaw, L, W, v, own, margin) {
  const e = L / 2 - W / 2, c = Math.cos(yaw), s = Math.sin(yaw);
  for (let k = -1; k <= 1; k++) out.push({ x: x + c * e * k, y: y + s * e * k, r: W / 2 * 1.05, v, yaw, own, car: true, m: margin });
  const a = L / 2 - CORNER_R, b = W / 2 - CORNER_R;
  for (let k = -1; k <= 1; k += 2) for (let j = -1; j <= 1; j += 2)
    out.push({ x: x + c * a * k - s * b * j, y: y + s * a * k + c * b * j, r: CORNER_R, v, yaw, own, car: true, m: margin });
}

/**
 * keisim corridor_gaps for one path: the nearest obstacle circle whose distance to the path's points is under
 * halfW + its radius + its margin. px, py: the path; rel: distance along it; n: its valid points; (ox, oy), heading:
 * the vehicle's centre and heading (circles more than 0.5 m behind it do not count); obs: [{x, y, r, v, yaw, own}];
 * margin(o): the room wanted beside o (-Infinity: o does not count). Returns {gap, v, o} (gap Infinity: nobody).
 */
export function corridorGap(px, py, rel, n, ox, oy, heading, halfLen, halfW, obs, margin) {
  const ux = Math.cos(heading), uy = Math.sin(heading);
  const res = { gap: Infinity, v: 0, o: null };
  for (const o of obs) {
    const mg = margin(o);
    if (mg === -Infinity) continue;
    const lon = (o.x - ox) * ux + (o.y - oy) * uy;
    if (lon <= -0.5) continue;
    let best = Infinity, bp = 0;
    for (let p = 0; p < n; p++) {
      const dx = px[p] - o.x, dy = py[p] - o.y, d2 = dx * dx + dy * dy;
      if (d2 < best) { best = d2; bp = p; }
    }
    const lim = halfW + o.r + mg;
    if (lim <= 0 || best >= lim * lim) continue;
    const g = Math.max(rel[bp], lon) - halfLen - o.r;
    if (g < res.gap) { res.gap = g; res.v = o.v * Math.cos(wrap(o.yaw - heading)); res.o = o; }
  }
  return res;
}

/** Time to cover d from speed v, accelerating at a up to vmax (keisim Traffic._reach). */
function reach(d, v, vmax, a) {
  const t1 = Math.max(0, (vmax - v) / a);
  const d1 = v * t1 + 0.5 * a * t1 * t1;
  if (d1 >= d) return (-v + Math.sqrt(v * v + 2 * a * d)) / a;
  return t1 + (d - d1) / Math.max(vmax, 0.1);
}

function interp(x, xs, ys) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 1, hi = n - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (xs[m] < x) lo = m + 1; else hi = m; }
  const f = (x - xs[lo - 1]) / ((xs[lo] - xs[lo - 1]) || 1e-9);
  return ys[lo - 1] + f * (ys[lo] - ys[lo - 1]);
}

/**
 * env (from src/traffic.js): {cars, stopLine(lane) -> arc length of its signal's stop line or null, light(lane, t),
 * mustYield(conn, t)}. opts: {perKm: riders per km of road, seed, max}.
 */
export function createBicycles(data, lanes, env, { perKm = 0, seed = 1, max = 60 } = {}) {
  let st = (seed * 2654435761) >>> 0 || 1;
  const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  const uni = (a, b) => a + (b - a) * rnd();
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  const cfg = data.cfg;
  const graph = lanes.graph || null;
  const roadOf = (L) => data.roads[L.road] || {};
  const shoulderOf = (L) => roadOf(L).shoulder ?? cfg.shoulder ?? 0.25;
  /** Half the lane's width: from its centre line to the line at its left edge (towns without a lane graph: one lane
   *  each way beside the centre line, or a one-way road's single lane on it). */
  const halfWidth = (L) => {
    if (L.width != null) return L.width / 2;
    const r = roadOf(L), lw = r.lane_width ?? cfg.lane_width ?? 3.5;
    return (r.half_width ?? cfg.road_half_width ?? lw + shoulderOf(L)) - (r.oneway ? 0 : lw / 2) - shoulderOf(L);
  };
  const speedOf = (L) => L.speed ?? roadOf(L).speed_limit ?? cfg.speed_limit ?? 10;
  /** keisim edge_offset: from the lane's centre line to the rider's line on its left. */
  const edgeOffset = (L) => halfWidth(L) + shoulderOf(L) - BIKE_EDGE;
  const isKerb = (L) => L.kind === 'road' && (L.index ?? 0) === 0;
  const bikeLanes = lanes.filter((L) => isKerb(L) && L.len > 12);
  const roadKm = (graph ? bikeLanes.reduce((a, L) => a + L.len, 0) / 2 : data.roads.reduce((a, r) => a + r.length, 0)) / 1000;
  const count = Math.min(max, Math.round(perKm * roadKm));
  const riders = [];

  // ------------------------------------------------------------------ the rider's line along a lane or connector
  const tracks = new Map();
  /** keisim bike_track: {pts, cum, laneS (the lane's arc length at each point), head (unwrapped), curv, off, len}. */
  function track(piece) {
    let tr = tracks.get(piece);
    if (tr) return tr;
    // the centre line, at most a metre between points
    const base = [], laneS = [];
    for (let i = 0; i < piece.pts.length; i++) {
      if (i) {
        const a = piece.pts[i - 1], b = piece.pts[i], d = piece.cum[i] - piece.cum[i - 1], m = Math.ceil(d / 1.0);
        for (let k = 1; k < m; k++) { base.push([a[0] + (b[0] - a[0]) * k / m, a[1] + (b[1] - a[1]) * k / m]); laneS.push(piece.cum[i - 1] + d * k / m); }
      }
      base.push(piece.pts[i]); laneS.push(piece.cum[i]);
    }
    const n = base.length;
    const hd = headingsOf(base), kLane = curvatureOf(base, hd);
    let off;
    if (piece.kind === 'road' || piece.kind === 'ramp') off = new Array(n).fill(edgeOffset(piece));
    else {                              // a connector: from the entry road's offset to the exit road's
      const d0 = edgeOffset(piece.from), d1 = edgeOffset(piece.to);
      off = laneS.map((s) => { const x = s / Math.max(piece.len, 1e-6); return d0 + (d1 - d0) * (x * x * (3 - 2 * x)); });
      const lim = kLane.map((k) => 0.6 / Math.max(k, 1e-6));     // left bends: well outside the centre of the arc
      if (lim.some((l, i) => l < off[i])) {
        off = off.map((o, i) => Math.min(o, lim[i]));
        const m = Math.min(9, n), h = m >> 1, sm = [];
        for (let i = 0; i < n; i++) {
          let a = 0;
          for (let k = -h; k < m - h; k++) a += off[clamp(i + k, 0, n - 1)];
          sm.push(a / m);
        }
        off = off.map((o, i) => { const keep = Math.min(1, Math.min(laneS[i], piece.len - laneS[i]) / 2); return Math.min(o, sm[i] * keep + o * (1 - keep)); });
      }
    }
    const pts = base.map((p, i) => [p[0] - Math.sin(hd[i]) * off[i], p[1] + Math.cos(hd[i]) * off[i]]);
    const cum = [0];
    for (let i = 1; i < n; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const head = headingsOf(pts);
    let curv = curvatureOf(pts, head);
    if (n > 7) curv = curv.map((_, i) => { let a = 0; for (let k = -3; k <= 3; k++) a += curv[clamp(i + k, 0, n - 1)]; return a / 7; });
    tr = { pts, cum, laneS, head, curv, off, len: cum[n - 1] };
    tracks.set(piece, tr);
    return tr;
  }

  /** Headings at the points of a polyline (unwrapped; the mean direction of the two segments at a point). */
  function headingsOf(pts) {
    const n = pts.length, seg = [];
    for (let i = 0; i < n - 1; i++) seg.push(Math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0]));
    for (let i = 1; i < seg.length; i++) seg[i] = seg[i - 1] + wrap(seg[i] - seg[i - 1]);
    const out = [];
    for (let i = 0; i < n; i++) out.push(i === 0 ? seg[0] : i === n - 1 ? seg[n - 2] : (seg[i - 1] + seg[i]) / 2);
    return out;
  }

  /** Curvature at the points (1/m, + = a left bend). */
  function curvatureOf(pts, hd) {
    const n = pts.length, out = new Array(n).fill(0);
    for (let i = 1; i < n - 1; i++) {
      const d = Math.hypot(pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]);
      const h0 = Math.atan2(pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0]);
      const h1 = Math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0]);
      out[i] = d > 1e-6 ? 2 * wrap(h1 - h0) / d : 0;
    }
    if (n > 2) { out[0] = out[1]; out[n - 1] = out[n - 2]; }
    return out;
  }

  const trackAt = (tr, laneS) => interp(laneS, tr.laneS, tr.cum);       // the track's arc length at the lane's
  const laneAt = (tr, u) => interp(u, tr.cum, tr.laneS);

  // ------------------------------------------------------------------ routes
  /** keisim Bicycles._extend: at least three pieces beyond the current one: straight on or left at each junction, on
   *  to a kerb lane (a right turn only where there is nothing else). */
  function extend(r) {
    while (r.route.length - r.ri < 4) {
      const last = r.route[r.route.length - 1];
      if (last.kind !== 'road') { r.route.push(last.to); continue; }       // after a connector: its lane
      const ok = last.next.filter((o) => isKerb(o.lane));
      if (!ok.length) break;
      const nr = ok.filter((o) => o.turn !== 'right');
      const cand = nr.length ? nr : ok;
      let tot = 0;
      for (const o of cand) tot += o.turn === 'straight' ? 1.5 : 1.0;
      let x = rnd() * tot, c = cand[cand.length - 1];
      for (const o of cand) { x -= o.turn === 'straight' ? 1.5 : 1.0; if (x < 0) { c = o; break; } }
      r.route.push(c.conn);
    }
    if (r.ri > 2) { r.route.splice(0, r.ri - 1); r.ri = 1; }
  }

  /** keisim Bicycles._cat_of: the tracks of the rider's next pieces in one: {pts, cum, head, curv, stops, first}. */
  function catOf(r) {
    if (r.route.length - r.ri < 4) extend(r);
    const key = r.route.slice(r.ri, r.ri + 4);
    const old = r.cat;
    if (old && old.key.length === key.length && old.key.every((p, i) => p === key[i])) return old;
    const cat = { key, pts: [], cum: [], head: [], curv: [], stops: [], first: 0 };
    let base = 0, hOff = 0;
    key.forEach((piece, j) => {
      const tr = track(piece);
      if (j) hOff += 2 * Math.PI * Math.round((cat.head[cat.head.length - 1] - (tr.head[0] + hOff)) / (2 * Math.PI));
      else cat.first = tr.len;
      for (let i = j ? 1 : 0; i < tr.pts.length; i++) {
        cat.pts.push(tr.pts[i]); cat.cum.push(base + tr.cum[i]); cat.head.push(tr.head[i] + hOff); cat.curv.push(tr.curv[i]);
      }
      if (piece.kind === 'road') {
        const sl = env.stopLine(piece);
        if (sl !== null) cat.stops.push({ u: base + trackAt(tr, sl), piece, kind: SIG });
        if (piece.stopSign != null) cat.stops.push({ u: base + trackAt(tr, piece.stopSign), piece, kind: SIGN });
      } else if (piece.yields && piece.yields.length && piece.wait != null) {
        cat.stops.push({ u: base + trackAt(tr, piece.wait), piece, kind: YIELD });
      }
      base += tr.len;
    });
    cat.stops.sort((a, b) => a.u - b.u);
    r.cat = cat;
    return cat;
  }

  /** keisim Bicycles._at: point, heading and curvature of the track at arc length u -> [x, y, h, k]. */
  function at(cat, u) {
    const cum = cat.cum, n = cum.length;
    let lo = 0, hi = n - 1;                         // last j with cum[j] <= u
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (cum[m] <= u) lo = m; else hi = m - 1; }
    const j = Math.min(lo, n - 2);
    const f = clamp((u - cum[j]) / Math.max(cum[j + 1] - cum[j], 1e-9), 0, 1);
    const a = cat.pts[j], b = cat.pts[j + 1];
    return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), cat.head[j] + f * (cat.head[j + 1] - cat.head[j]),
      cat.curv[j] + f * (cat.curv[j + 1] - cat.curv[j])];
  }

  // ------------------------------------------------------------------ spawning
  function free(p, ego, avoid, skip) {
    if (ego && Math.hypot(p[0] - ego.x, p[1] - ego.y) < avoid) return false;
    for (const c of env.cars) if (Math.hypot(c.xy[0] - p[0], c.xy[1] - p[1]) < c.L / 2 + BIKE_R + 0.45 + 0.5) return false;
    for (const o of riders) if (o !== skip && Math.hypot(o.x - p[0], o.y - p[1]) < 9) return false;
    return true;
  }

  /** A random spot on the kerb lanes (weighted by length) -> {L, tr, u, p, h} or null. */
  function spot(ego, avoid, skip) {
    const total = bikeLanes.reduce((a, L) => a + L.len, 0);
    let x = rnd() * total, L = bikeLanes[0];
    for (const l of bikeLanes) { if (x < l.len) { L = l; break; } x -= l.len; }
    const tr = track(L);
    const u = uni(3, tr.len - 3);
    const p = [interp(u, tr.cum, tr.pts.map((q) => q[0])), interp(u, tr.cum, tr.pts.map((q) => q[1]))];
    if (!free(p, ego, avoid, skip)) return null;
    return { L, tr, u, p, h: interp(u, tr.cum, tr.head) };
  }

  function put(r, s) {
    Object.assign(r, { route: [s.L], ri: 0, cat: null, u: s.u, x: s.p[0], y: s.p[1], yaw: s.h, lat: 0, latT: 0,
      piece: s.L, laneS: laneAt(s.tr, s.u), side: interp(laneAt(s.tr, s.u), s.tr.laneS, s.tr.off), stuck: 0, ssDone: null });
  }

  function spawn(ego) {
    riders.length = 0;
    if (!bikeLanes.length) return;
    for (let tries = 0; riders.length < count && tries < count * 30; tries++) {
      const s = spot(ego, 10, null);
      if (!s) continue;
      const ty = (() => { const x = rnd(); return x < TYPES[0].p ? 0 : x < TYPES[0].p + TYPES[1].p ? 1 : 2; })();
      const T = TYPES[ty];
      const cruise = uni(...T.cruise);
      const k1 = uni(0.9, 1.05), k2 = uni(0.85, 1.1);
      const rgb = [pick(ty === 1 ? SPORTS : FRAMES).map((c) => clamp(c * k1, 0, 255)), pick(SHIRTS).map((c) => clamp(c * k2, 0, 255)),
        pick(PANTS), pick(SKIN)].flat();
      const r = { id: riders.length, type: ty, T, cruise, rgb, v: cruise * uni(0.6, 1.0), acc: 0, out: OUT * uni(0.9, 1.15),
        dwell: 1, crank: uni(0, 2 * Math.PI), wheel: uni(0, 2 * Math.PI), roll: 0, steer: 0, foot: 0, look: 0, lookT: 0,
        effort: 0.3, wob: uni(0, 2 * Math.PI), wobF: uni(0.55, 0.9), a0: uni(...T.a0), pmax: uni(...T.pmax),
        alat: uni(...T.alat), react: uni(0.3, 1.1), wait: 0, stood: 0, gap: Infinity, vWant: 0, yawrate: 0,
        height: uni(1.5, 1.82), shiftT: 0, mode: PEDAL, gear: T.gears.length >> 1 };
      put(r, s);
      riders.push(r);
    }
  }

  /** A rider standing for STUCK_S (jammed, or at the end of a road with no way on) goes elsewhere, out of sight. */
  function release(r, ego) {
    if (ego && Math.hypot(r.x - ego.x, r.y - ego.y) < 60) return;
    for (let k = 0; k < 20; k++) {
      const s = spot(ego, 60, r);
      if (!s) continue;
      put(r, s);
      Object.assign(r, { v: 0, acc: 0, lookT: 0, roll: 0, steer: 0, yawrate: 0, wait: 0, stood: 0, mode: PEDAL, dwell: 1 });
      return;
    }
  }

  // ------------------------------------------------------------------ one step
  const pHx = new Float64Array(P), pHy = new Float64Array(P), pOx = new Float64Array(P), pOy = new Float64Array(P);

  function circlesOf(out, r) {
    const c = Math.cos(r.yaw), s = Math.sin(r.yaw), e = BIKE_LEN / 2 - BIKE_R;
    for (let k = -1; k <= 1; k++) out.push({ x: r.x + c * e * k, y: r.y + s * e * k, r: BIKE_R, v: r.v, yaw: r.yaw, own: r, bike: true, m: MARGIN_BIKE });
  }

  /** keisim Bicycles._rear_clear: no car beside the rider or coming up from behind in the time it takes to move out
   *  and pass, and no other rider there. */
  function rearClear(r, ego) {
    const c = Math.cos(r.yaw), s = Math.sin(r.yaw);
    const car = (x, y, v) => {
      const dx = x - r.x, dy = y - r.y, lon = dx * c + dy * s, right = dx * s - dy * c;
      return lon < 3 && lon > -(6 + 3.5 * Math.max(v - r.v, 0) + 0.5 * v) && right > -1 && right < 3.6;
    };
    for (const o of env.cars) if (car(o.xy[0], o.xy[1], o.v)) return false;
    if (ego && car(ego.x, ego.y, ego.v)) return false;
    for (const o of riders) {
      if (o === r) continue;
      const dx = o.x - r.x, dy = o.y - r.y, lon = dx * c + dy * s, right = dx * s - dy * c;
      if (lon < 1.9 && lon > -(2 + 2.5 * Math.max(o.v - r.v, 0)) && right > -0.3 && right < 2.0) return false;
    }
    return true;
  }

  /** Pass 1 for rider i (everybody still where the last step left them): what is in the way on its home line and on
   *  the line moved out, the cars close beside it, the bends ahead -> r.g. */
  function sense(i, all) {
    const r = riders[i], cat = catOf(r), total = cat.cum[cat.cum.length - 1];
    const look = 12 + 3 * r.v;
    let n = 0, vBend = Infinity;
    for (let p = 0; p < P; p++) {
      const d = REL[p];
      if (p && (d > look || r.u + d > total)) break;
      const q = at(cat, Math.min(r.u + d, total));
      if (p && Math.cos(q[2] - r.yaw) < -0.3) break;          // an imported lane that folds back: its line ends here
      const x = clamp(d / OUT_BLEND, 0, 1), nx = Math.sin(q[2]), ny = -Math.cos(q[2]);
      const a = r.lat * (1 - x), b = r.lat + (r.out - r.lat) * x;
      pHx[n] = q[0] + nx * a; pHy[n] = q[1] + ny * a; pOx[n] = q[0] + nx * b; pOy[n] = q[1] + ny * b;
      vBend = Math.min(vBend, Math.sqrt(r.alat / Math.max(Math.abs(q[3]), 1e-3) + 2 * 0.9 * d));   // braking gently into bends
      n++;
    }
    // the other riders going the same way: who does not count on this rider's paths (see keisim: a rider a length
    // behind; of two side by side the one further out gives way), who is between it and its home line
    const ux = Math.cos(r.yaw), uy = Math.sin(r.yaw);
    const ign = new Set(), lefts = [];
    let beside = false;
    for (let j = 0; j < riders.length; j++) {
      if (j === i) continue;
      const o = riders[j], dx = o.x - r.x, dy = o.y - r.y;
      if (Math.abs(dx) > 60 || Math.abs(dy) > 60 || Math.cos(o.yaw - r.yaw) <= 0.5) continue;
      const du = dx * ux + dy * uy, rgt = dx * uy - dy * ux, dlat = o.lat - r.lat;
      const lower = Math.abs(dlat) > 0.25 ? dlat > 0 : Math.abs(du) > 0.15 ? du < 0 : j > i;
      if (du <= -BIKE_LEN || (Math.abs(du) < BIKE_LEN && lower)) ign.add(o);
      if (rgt < 0.3 && rgt > -(r.lat + 0.7)) {
        if (du > -(BIKE_LEN + 0.5 + Math.max(o.v - r.v, 0)) && du < BIKE_LEN + 0.4) beside = true;
        if (du > -BIKE_LEN && du < 8) lefts.push({ o, du });
      }
    }
    const lim = look + 6;
    const near = all.filter((o) => o.own !== r && Math.abs(o.x - r.x) < lim && Math.abs(o.y - r.y) < lim);
    // a turning car's future does not count for a rider it comes up behind, only for those it is beside or ahead of
    // one behind the other with a car through the same right turn or where the lane folds (env.single): behind the
    // car when it is ahead, and no waiting for one that waits for the rider
    const one = new Map();
    const single = (o) => {
      if (!o.own.route) return null;
      let l = one.get(o.own);
      if (l === undefined) { l = env.single(o.own, r); one.set(o.own, l); }
      return l;
    };
    const mAll = (o) => {
      if (o.car) { const l = single(o); if (l !== null) return l > SINGLE_AHEAD ? MARGIN_SINGLE : Math.min(o.m, 0); }
      // the ego standing beside or behind a rider that has stood for a while: it is waiting for the rider, who goes on
      if (o.car && !o.own.route && o.v < 0.3 && r.egoGo > 0 && (o.own.x - r.x) * ux + (o.own.y - r.y) * uy < 1) return Math.min(o.m, 0);
      return o.cx === undefined ? o.m : (o.cx - r.x) * ux + (o.cy - r.y) * uy > -1.5 ? o.m : -Infinity;
    };
    const mIgn = (o) => (o.bike && ign.has(o.own) ? -Infinity : mAll(o));
    const mSide = (o) => {                      // filtering past standing cars, but not past one it stays behind
      if (!o.car || o.cx !== undefined) return -Infinity;
      const l = single(o);
      return l !== null && l > SINGLE_AHEAD ? MARGIN_SINGLE : FILTER_SIDE;
    };
    const hl = BIKE_LEN / 2, hw = BIKE_W / 2;
    r.g = {
      Hf: corridorGap(pHx, pHy, REL, n, r.x, r.y, r.yaw, hl, hw, near, mAll).gap,
      H: corridorGap(pHx, pHy, REL, n, r.x, r.y, r.yaw, hl, hw, near, mIgn),
      O: corridorGap(pOx, pOy, REL, n, r.x, r.y, r.yaw, hl, hw, near, mIgn),
      F: corridorGap(pHx, pHy, REL, n, r.x, r.y, r.yaw, hl, hw, near, mSide),
      beside, lefts, vBend,
    };
  }

  /** Pass 2 for rider r: what it decides and how it moves in dt (keisim Bicycles.step, the loop over the riders). */
  function move(r, dt, t, ego) {
    const T = r.T, g = r.g, TL = env.TL, v = r.v, b = T.b;
    let cat = r.cat;
    const cur = r.route[r.ri], onRoad = cur.kind === 'road';
    const front = r.u + BIKE_LEN / 2;
    // -- moving out round a slower rider (after a look back) / back in
    if (r.latT < 0.01) {
      const leadSlow = g.H.gap < Infinity && g.H.v < r.cruise - 0.8 && !!g.H.o.bike;
      const want = leadSlow && g.H.gap < 5 + 2.5 * v && g.O.gap > g.H.gap + 8 && onRoad && cat.first - r.u > 14 && r.u > 3;
      if (want && rearClear(r, ego)) {
        r.lookT += dt;
        if (r.lookT >= LOOK_BACK) { r.latT = r.out; r.lookT = 0; }
      } else r.lookT = Math.max(0, r.lookT - 2 * dt);
    } else {
      r.lookT = 0;
      if ((g.Hf > Math.max(9, 2.5 * v) || (g.Hf >= g.O.gap && g.O.gap < 6)) && !g.beside) r.latT = 0;
    }
    const lead = r.latT > 0.01 ? g.O : g.H;
    // -- the speed the rider wants
    let vT = Math.min(r.cruise, speedOf(cur), g.vBend);
    let whyStop = false, aReq = 0;              // aReq: the braking it takes to stop where the rider has to (<= 0)
    let stopD = Infinity;
    for (const s of cat.stops) {                // lights, 止まれ, giving way
      const d = s.u - front;
      if (d > 40) break;
      if (d < -0.5) continue;
      if (s.kind === SIG) {
        const st = env.light(s.piece, t);
        if (st === TL.RED || (st === TL.YELLOW && d > v * v / (2 * 2.5) + 0.5)) stopD = Math.min(stopD, d - 0.3);
      } else if (s.kind === SIGN) {
        if (r.ssDone === s.piece) continue;
        if (v < 0.05 && d <= 2) { r.ssDone = s.piece; continue; }
        stopD = Math.min(stopD, d - 0.4);
      } else {
        if (v > 1 && d < v * v / (2 * 3)) continue;
        if (env.mustYield(s.piece, t)) stopD = Math.min(stopD, Math.max(d - 0.2, 0));
      }
    }
    if (stopD < Infinity) {
      vT = Math.min(vT, Math.sqrt(2 * b * Math.max(stopD, 0)));
      whyStop = stopD < v * v / (2 * 0.35) + 3;
      aReq = -v * v / (2 * Math.max(stopD, 0.05));
    }
    if (lead.gap < Infinity) {                  // whoever is in the way
      const s0 = lead.o.car ? S0_CAR : S0_BIKE, room = Math.max(lead.gap - s0, 0), vl = Math.max(lead.v, 0);
      let vObs = Math.min(Math.sqrt(vl * vl + 2 * b * room), vl + room / T_HEAD);
      if (lead.gap < s0) vObs = Math.max(0, vl + (lead.gap - s0) / T_HEAD);   // too close (beside a car it stays behind): drop back
      if (lead.v < 0) vObs = Math.min(vObs, Math.sqrt(2 * b * Math.max(room + 1.5 * lead.v, 0)));   // coming towards the rider
      vT = Math.min(vT, vObs);
      whyStop = whyStop || (vl < 0.5 && lead.gap < v * v / (2 * 0.35) + 3);
      if (v > vl) aReq = Math.min(aReq, -(v * v - vl * vl) / (2 * Math.max(room, 0.05)));
    }
    // moved out with a junction close: no riding into it side by side - fall in behind whoever is on the left
    if (r.lat > 0.25 && (!onRoad || cat.first - r.u < 14))
      for (const { o, du } of g.lefts) vT = Math.min(vT, Math.max(0, o.v + (du - BIKE_LEN - S0_BIKE) / T_HEAD));
    // filtering past cars close beside the line
    if (g.F.gap < Infinity) vT = Math.min(vT, Math.sqrt((Math.max(g.F.v, 0) + FILTER_DV) ** 2 + 2 * Math.max(g.F.gap, 0)));
    // -- reaction before setting off
    if (v < 0.05) {
      r.stood += dt;
      if (vT > 0.4) {
        r.wait += dt;
        if (r.wait < (r.stood > 1 ? r.react : 0.15)) vT = 0;
      } else r.wait = 0;
    } else {
      r.wait = 0;
      if (v > 0.5) r.stood = 0;
    }
    r.gap = lead.gap; r.vWant = vT;
    // -- forces: what the legs give, rolling and air drag, the brakes
    const m = T.mass;
    const aCoast = v > 0.02 ? -(T.crr * G + 0.5 * RHO * T.cda * v * v / m) : 0;
    let aCmd = (vT - v) / TAU_V;
    // riders brake late: once stopping in time takes most of their comfortable braking, as hard as it takes
    if (aReq < -0.7 * b || (r.mode === BRAKE && aReq < -0.4 * b)) aCmd = Math.min(aCmd, aReq);
    aCmd = Math.max(aCmd, -B_MAX);
    const aPed = Math.min(r.a0, r.pmax / (m * Math.max(v, 0.6))) + aCoast;      // the most pedalling gives
    let mode = r.mode;
    r.dwell += dt;
    const halt = vT <= 0 && v < 1.2;            // the last metre of a stop: on the brakes
    const easing = (whyStop && v > 1.6) || vT <= 0;
    if (mode === PEDAL) {
      if (aCmd < aCoast - 0.03 || easing) mode = aCmd > aCoast - 0.45 && !halt ? COAST : BRAKE;
    } else if (mode === COAST) {
      if (aCmd <= aCoast - 0.45 || halt) mode = BRAKE;
      else if (aCmd > aCoast + 0.2 && r.dwell > 0.5 && !easing) mode = PEDAL;
    } else if (aCmd > aCoast - 0.2 && r.dwell > 0.3 && !halt) mode = aCmd < aCoast + 0.2 || easing ? COAST : PEDAL;
    if (mode !== r.mode) { r.mode = mode; r.dwell = 0; }
    let a, eff = 0;
    if (mode === PEDAL) { a = Math.min(aCmd, aPed); eff = Math.max(0, (a - aCoast) * m * Math.max(v, 0.6)) / r.pmax; }
    else if (mode === COAST) a = aCoast;
    else a = Math.min(aCmd, aCoast, halt ? -0.9 : 0);
    r.effort += (Math.min(eff, 1) - r.effort) * Math.min(1, dt / 0.3);
    let vNew = Math.max(0, v + a * dt);
    if (vNew < 0.06 && a <= 0) vNew = 0;
    r.acc = a;
    // -- along the track
    let u = r.u + vNew * dt;
    while (u > cat.first) {
      if (r.ri + 1 >= r.route.length) { u = cat.first; vNew = 0; break; }
      u -= cat.first;
      r.ri++;
      cat = catOf(r);
    }
    r.v = vNew; r.u = u;
    const piece = r.route[r.ri], trk = track(piece);
    r.piece = piece;
    r.laneS = laneAt(trk, u);
    // -- sideways: the move out and back, and the wobble (larger when slow, none when standing)
    const rate = Math.min(OUT_RATE, 0.22 * vNew);
    const dl = clamp(r.latT - r.lat, -rate * dt, rate * dt);
    r.lat += dl;
    const wf = 2 * Math.PI * r.wobF * Math.min(1, vNew);
    r.wob += wf * dt;
    const amp = (0.012 + 0.06 * Math.exp(-vNew / 1.1)) * Math.min(1, vNew / 0.5);
    const wob = amp * Math.sin(r.wob), dwob = amp * Math.cos(r.wob) * wf;
    const q = at(cat, u);
    r.x = q[0] + Math.sin(q[2]) * (r.lat + wob);
    r.y = q[1] - Math.cos(q[2]) * (r.lat + wob);
    r.side = interp(r.laneS, trk.laneS, trk.off) - r.lat - wob;
    const yaw = q[2] - Math.atan2(dl / dt + dwob, Math.max(vNew, 1.5));
    // -- what the renderer shows: lean, steering, wheels, gears, cranks, the foot on the ground, the look back
    const w = vNew > 0.05 ? wrap(yaw - r.yaw) / dt : 0;
    r.yaw = wrap(yaw);
    r.yawrate += (w - r.yawrate) * Math.min(1, dt / 0.25);
    r.roll += (clamp(Math.atan(vNew * r.yawrate / G), -0.55, 0.55) - r.roll) * Math.min(1, dt / 0.2);
    const steer = vNew > 0.05 ? Math.atan(WHEELBASE * w / Math.max(vNew, 1.2)) : r.steer * 0.9;
    r.steer += (clamp(steer, -0.6, 0.6) - r.steer) * Math.min(1, dt / 0.12);
    r.wheel = (r.wheel + vNew * dt / T.wheel) % (2 * Math.PI);
    let gi = Math.min(r.gear, T.gears.length - 1);
    const cad = vNew / T.gears[gi];
    r.shiftT += dt;
    if (r.shiftT > 1.5) {
      if (cad > T.cad[1] && gi + 1 < T.gears.length) { gi++; r.shiftT = 0; }
      else if (cad < T.cad[0] && gi > 0) { gi--; r.shiftT = 0; }
    }
    r.gear = gi;
    const stopping = vNew < 0.9 && (a <= 0 || vT <= 0);
    let ph = r.crank;
    if (mode === PEDAL && vNew > 0.05) ph += 2 * Math.PI * vNew / T.gears[gi] * dt;
    else {            // cranks to level while coasting; the right pedal up and forward for the start at a stop
      const goal = stopping || vNew < 0.05 ? Math.PI / 3 : Math.cos(ph - Math.PI / 2) > 0 ? Math.PI / 2 : 3 * Math.PI / 2;
      ph += clamp(wrap(goal - ph), -3 * dt, 3 * dt);
    }
    r.crank = ((ph % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    r.foot += clamp((vNew < 0.05 || stopping ? 1 : 0) - r.foot, -dt / 0.45, dt / 0.3);
    r.look += clamp((r.lookT > 0 ? -1 : 0) - r.look, -dt / 0.2, dt / 0.25);
    r.stuck = vNew < 0.1 ? r.stuck + dt : 0;
    r.egoGo = r.stuck > EGO_WAIT ? 6 : Math.max(0, (r.egoGo || 0) - dt);
  }

  /**
   * One step of dt at town time t, after the cars' (their poses and speeds are this step's). ego: {x, y, yaw, v} or
   * null; egoPred: [[x, y]] where the ego will be shortly when it is about to turn (or nothing).
   */
  function step(dt, t, ego, egoPred = null) {
    if (!riders.length) return;
    const all = [];
    for (const c of env.cars) {
      carCircles(all, c.xy[0], c.xy[1], c.yaw, c.L, c.W, c.v, c, MARGIN_CAR);
      if (c.v > 0.5 && env.turning(c)) for (const tau of [0.6, 1.2, 1.8]) {       // where a turning car will be
        const p = env.ahead(c, c.v * tau);
        if (!p) break;
        all.push({ x: p[0], y: p[1], r: c.W / 2, v: c.v, yaw: c.yaw, own: c, car: true, m: MARGIN_TURN, cx: c.xy[0], cy: c.xy[1] });
      }
    }
    if (ego) {
      carCircles(all, ego.x, ego.y, ego.yaw, env.ego.L, env.ego.W, ego.v, ego, MARGIN_CAR);
      for (const p of egoPred || []) all.push({ x: p[0], y: p[1], r: 0.95, v: ego.v, yaw: ego.yaw, own: ego, car: true, m: MARGIN_TURN, cx: ego.x, cy: ego.y });
    }
    for (const r of riders) circlesOf(all, r);
    for (let i = 0; i < riders.length; i++) sense(i, all);
    for (const r of riders) move(r, dt, t, ego);
    for (const r of riders) if (r.stuck > STUCK_S) release(r, ego);
  }

  // ------------------------------------------------------------------ what the cars need to know
  /** The riders' circles, and (predict) where the moving ones will be in 0.8, 1.6 and 2.4 s (one circle each). */
  function circles(predict = true) {
    const out = [];
    for (const r of riders) {
      circlesOf(out, r);
      if (!predict || r.v <= 0.5 || !r.cat) continue;
      const total = r.cat.cum[r.cat.cum.length - 1];
      for (const tau of [0.8, 1.6, 2.4]) {
        const uu = r.u + r.v * tau;
        if (uu > total) break;
        const q = at(r.cat, uu);
        out.push({ x: q[0] + Math.sin(q[2]) * r.lat, y: q[1] - Math.cos(q[2]) * r.lat, r: BIKE_R, v: r.v, yaw: r.yaw, own: r, bike: true, m: 0 });
      }
    }
    return out;
  }

  /** keisim Traffic._bike_arrival: when the first rider gets to arc length s2 of connector b (Infinity: none will):
   *  the riders on b, and those on its entry lane who take b and are not held by their light. */
  function arrival(b, s2, t) {
    let best = Infinity;
    const TL = env.TL;
    for (const r of riders) {
      const front = r.laneS + BIKE_LEN / 2;
      if (r.piece === b) {
        if (r.laneS - BIKE_LEN / 2 > s2 + 2.5) continue;
        const d = s2 - front;
        best = Math.min(best, d < 3 ? 0 : reach(d, r.v, r.cruise, 1.0));
      } else if (r.piece === b.from && r.route[r.ri + 1] === b) {
        const sl = env.stopLine(b.from);
        if (sl !== null) {
          const st = env.light(b.from, t), d = sl - front;
          if (st !== TL.GREEN && d > -0.5 && (st === TL.RED || d > r.v * r.v / 5 + 0.5)) continue;
        }
        best = Math.min(best, (r.v >= 0.2 ? 0 : 0.8) + reach(Math.max(b.from.len - front + s2, 0), r.v, r.cruise, 1.0));
      }
    }
    return best;
  }

  /** Rows for src/world/actors.js update(…, …, bike): [id, x, y, yaw, kind, height, 12 colours (frame, shirt, pants,
   *  skin), roll, steer, crank, wheel, foot, look, effort] of the riders within r of (cx, cy). */
  function drawList(cx, cy, rad = 140) {
    const out = [];
    for (const r of riders) {
      if (Math.abs(r.x - cx) > rad || Math.abs(r.y - cy) > rad) continue;
      out.push([r.id, r.x, r.y, r.yaw, r.type, r.height, ...r.rgb, r.roll + (r.type === 1 ? 0.12 : 0.07) * r.foot, r.steer, r.crank, r.wheel,
        r.foot, r.look, r.effort]);
    }
    return out;
  }

  return { riders, step, reset: spawn, circles, arrival, drawList, halfWidth, nextPiece: (r) => r.route[r.ri + 1] || null,
    get n() { return riders.length; } };
}
