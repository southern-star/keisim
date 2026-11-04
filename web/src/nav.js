// Route and navigation inputs for the in-browser KeiPilot (src/pilot.js), mirroring keisim/roadnet.py and
// keisim/route.py: one lane per road and direction, junction connectors as arc-like Béziers, and the model's
// two route inputs, the turn at the next junction with a choice (command) and the point 4 m past that junction's
// exit (target point). Towns exported with KeiSim's lane graph (片側2車線 roads, ramps: src/lanes.js) use its lanes
// instead; their routes change lanes where the next movement needs it, and take a ramp for the left turn at the
// junctions that have one. All coordinates are KeiSim's (x east, y north, yaw counter-clockwise).
import { LC, Path, buildGraph, changedInto, laneMoves, measure, pickMove } from './lanes.js';

export const CMD = { left: 0, straight: 1, right: 2 };
const RAMP_LEAD = 120.0;                            // m: start(ramp) starts this far before the ramp's diverge

export const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
export const heading = (a, b) => Math.atan2(b[1] - a[1], b[0] - a[0]);

/** Offset a polyline to the left by d (negative: right), using the averaged tangent at each vertex. */
function offsetPolyline(pts, d) {
  return pts.map((p, i) => {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
    const h = heading(a, b);
    return [p[0] - Math.sin(h) * d, p[1] + Math.cos(h) * d];
  });
}

/** keisim.geometry.arc_bezier: smooth, near-circular connector from pose (p0, h0) to pose (p1, h1). */
export function arcBezier(p0, h0, p1, h1, n = 24) {
  const chord = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
  const delta = Math.abs(wrap(h1 - h0));
  let d = chord / 3;
  if (delta >= 1e-3) d = Math.min(chord, (4 / 3) * Math.tan(delta / 4) * chord / (2 * Math.sin(delta / 2)));
  const c1 = [p0[0] + d * Math.cos(h0), p0[1] + d * Math.sin(h0)], c2 = [p1[0] - d * Math.cos(h1), p1[1] - d * Math.sin(h1)];
  const out = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, u = 1 - t;
    const w = [u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t];
    out.push([w[0] * p0[0] + w[1] * c1[0] + w[2] * c2[0] + w[3] * p1[0], w[0] * p0[1] + w[1] * c1[1] + w[2] * c2[1] + w[3] * p1[1]]);
  }
  return out;
}

/** Polyline with arc lengths: {pts, cum, len}. */
export function polyline(pts) {
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  return { pts, cum, len: cum[cum.length - 1] };
}

/** Point at arc length s of a polyline from polyline() (clamped to its ends). */
export function pointOn(pl, s) {
  const { pts, cum } = pl;
  let i = 1;
  while (i < cum.length - 1 && cum[i] < s) i++;
  const t = Math.max(0, Math.min(1, (s - cum[i - 1]) / ((cum[i] - cum[i - 1]) || 1e-9)));
  return [pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t];
}

/** One lane per road and direction (left-hand traffic: left of the centre line; one-way roads of real towns: one
 *  lane on the centre line), with the ways out of the junction at its end (no U-turns): {id, road, from, to, pts,
 *  len, h0, h1, next: [{lane, turn, conn}]}.
 *  conn is the junction connector (keisim/roadnet.py): {from, to, junction, turn, pts, cum, len, yields, wait},
 *  shared by every route that takes it; yields / wait are filled in for two-phase signals (src/traffic.js). */
export function buildLanes(data) {
  if (data.lane_graph) return buildGraph(data);        // KeiSim's own lanes (src/lanes.js)
  const cfg = data.cfg;
  const lanes = [];
  for (const r of data.roads) {
    if (r.oneway) {                                   // real towns: a single lane on the centre line, a -> b
      lanes.push({ id: lanes.length, road: r.id, from: r.a, to: r.b, pts: r.center.map((p) => [p[0], p[1]]) });
      continue;
    }
    const off = (cfg.left_hand_traffic === false ? -1 : 1) * (r.lane_width || cfg.lane_width || 3.5) / 2;   // road types: per road
    lanes.push({ id: lanes.length, road: r.id, from: r.a, to: r.b, pts: offsetPolyline(r.center, off) });
    lanes.push({ id: lanes.length, road: r.id, from: r.b, to: r.a, pts: offsetPolyline(r.center, -off).reverse() });
  }
  for (const L of lanes) Object.assign(measure(L), { kind: 'road', links: new Map(), siblings: [], pred: [], signal: null });
  // junction connectors: from every incoming lane to every outgoing lane of another road (no U-turns)
  for (const L of lanes) {
    L.next = lanes.filter((o) => o.from === L.to && o.road !== L.road).map((o) => {
      const dh = wrap(o.h0 - L.h1);
      const turn = dh > 0.5 ? 'left' : dh < -0.5 ? 'right' : 'straight';
      const conn = { from: L, to: o, junction: L.to, turn, yields: [], wait: null,
        ...polyline(arcBezier(L.pts[L.pts.length - 1], L.h1, o.pts[0], o.h0)) };
      o.pred.push(conn);
      return { lane: o, turn, conn };
    });
  }
  return lanes;
}

export function createNavigator(data, lanes = buildLanes(data)) {
  const junctions = new Map(data.junctions.map((j) => [j.id, j]));

  // ---------------------------------------------------------------- the route: a growing path of lanes (src/lanes.js)
  let path, decisions, seg;
  const want = { turn: null };                      // the driver's pick for the next decision junction

  /** The movements from lane piece P at the end of its lane, made from lane-local sLo on: with siblings or ramps,
   *  keisim route.lane_moves (no change straight back after a change); else the junction's connectors. */
  function movesFrom(P, sLo) {
    const L = P.lane;
    if (L.links.size && !changedInto(path, P)) {
      const moves = laneMoves(L, sLo);
      if (moves.some((m) => m.src !== L || m.ramp)) return { moves, multi: true };
    }
    return { moves: L.next.map((o) => ({ turn: o.turn, src: L, pre: null, conn: o.conn })), multi: false };
  }

  /** From where on lane piece P a change may start: after the car (and its margin) if it is on P. */
  const lowOn = (P) => {
    const prev = path.segs[path.segs.indexOf(P) - 1];       // right after a change (a ramp's merge): LC.START_MARGIN in
    return Math.max(seg.s >= P.s0 && seg.s < P.s1 ? P.lo + seg.s - P.s0 + LC.START_MARGIN : P.lo,
      prev && prev.kind === 'change' ? P.lo + LC.START_MARGIN : 0);
  };

  /** Way out at the end of the last lane piece: the driver's pick at the first junction with a choice that the car
   *  has not entered yet, otherwise a random one (the only one at a bend); on multi-lane roads the movement first,
   *  then the lane (pickMove). */
  function pick(P) {
    const L = P.lane;
    if (!L.next.length) return null;
    const { moves, multi } = movesFrom(P, lowOn(P));
    const J = junctions.get(L.to);
    if (J && J.arms >= 3 && want.turn && !decisions.some((d) => d.sIn > seg.s + 1)) {
      const ok = moves.filter((m) => m.turn === want.turn);
      want.turn = null;
      if (ok.length) return ok.find((m) => m.src === L) || ok[0];
    }
    if (multi) return pickMove(Math.random, moves, L);
    return moves[Math.floor(Math.random() * moves.length)];
  }

  /** Append movement mv; a junction with a choice (or a ramp) becomes a decision: {junction, turn, sIn, sOut, lane
   *  (the lane it is made from), conn (null for a ramp), ramp}. */
  function take(mv, sLo) {
    const J = junctions.get(mv.src.to);
    const r = path.addMove(mv, sLo);
    if (r.conn) {
      if (J && J.arms >= 3) decisions.push({ junction: J.id, turn: mv.turn, sIn: r.conn.s0, sOut: r.conn.s1, lane: mv.conn.from, conn: mv.conn });
    } else {
      const { ramp } = r;
      decisions.push({ junction: ramp.junction, turn: ramp.turn, sIn: ramp.c1.s0, sOut: ramp.c2.s1, lane: ramp.from, conn: null, ramp: ramp.lane });
    }
  }

  function extend(until) {
    while (path.end < until) {
      const P = path.last, mv = pick(P);
      if (!mv) break;
      take(mv, lowOn(P));
    }
  }

  /** Start on lane `L` at arc length `s0` (first(): the first movements, taken before the random ones); returns the
   *  pose there. */
  function reset(L, s0, first = null) {
    path = new Path(); decisions = [];
    seg = { i: 0, s: 0 };
    path.addLane(L);
    seg.s = Math.min(s0, L.len - 1);
    if (first) first();
    extend(seg.s + 250);
    const p = path.pointAt(seg.s), q = path.pointAt(seg.s + 0.5);
    return { x: p[0], y: p[1], yaw: heading(p, q) };
  }

  function locate(s) {
    const { cum } = path;
    s = Math.max(0, Math.min(s, cum[cum.length - 1]));
    let i = Math.max(1, Math.min(seg.i, cum.length - 1));
    while (i > 1 && cum[i - 1] > s) i--;
    while (i < cum.length - 1 && cum[i] < s) i++;
    return i;
  }
  function pointAt(s) {
    const { pts, cum } = path, i = locate(s), a = pts[i - 1], b = pts[i];
    const t = Math.max(0, Math.min(1, (s - cum[i - 1]) / ((cum[i] - cum[i - 1]) || 1e-9)));
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }

  /** Follow the car along the route (closest point within a window ahead); returns its lateral offset [m]. */
  function track(x, y) {
    const { pts, cum } = path;
    let best = Infinity, bs = seg.s, bi = seg.i;
    const i0 = locate(seg.s - 3), i1 = locate(seg.s + 25);
    for (let i = i0; i <= i1; i++) {
      const a = pts[i - 1], b = pts[i], dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy || 1e-9;
      const t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / l2));
      const d = Math.hypot(a[0] + dx * t - x, a[1] + dy * t - y);
      if (d < best) { best = d; bs = cum[i - 1] + t * Math.sqrt(l2); bi = i; }
    }
    seg.s = Math.max(seg.s, bs); seg.i = bi;
    extend(seg.s + 250);
    return best;
  }

  /** Next junction with a real choice (or ramp) that has not been left yet (keisim Route.next_decision). */
  const nextDecision = () => decisions.find((d) => d.sOut > seg.s + 0.5) || null;

  /** Command and target point (world) for the model, as keisim Route.command / target_point_world. */
  function inputs() {
    const j = nextDecision();
    const end = path.end;
    if (!j) return { cmd: CMD.straight, tp: pointAt(end), dist: null };
    return { cmd: CMD[j.turn], tp: pointAt(Math.min(j.sOut + 4, end)), dist: j.sIn - seg.s, turn: j.turn, ramp: !!j.ramp };
  }

  /** The driver asks for `turn` at the next decision junction not yet entered. Returns false if impossible. On
   *  multi-lane roads the route may change lanes for it (from where the car is, if it is on that road already). */
  function request(turn) {
    const k = decisions.findIndex((d) => d.sIn > seg.s + 1);
    if (k < 0) { want.turn = turn; return true; }
    const d = decisions[k];
    if (d.turn === turn) return true;
    // the road before that junction: its segments from the last connector (or the route's start) on
    const segs = path.segs;
    let jd = segs.findIndex((g) => g.s0 >= d.sIn - 1e-6 && g.kind !== 'lane');
    let iF = jd - 1;
    while (iF > 0 && segs[iF - 1].kind !== 'conn') iF--;
    const ci = path.segAt(seg.s);
    let base = segs[iF];
    if (ci > iF) base = segs[ci].kind === 'lane' ? segs[ci] : segs[Math.min(ci + 1, jd - 1)];
    if (base.kind !== 'lane') return false;
    const sLo = lowOn(base);
    const opts = movesFrom(base, sLo).moves.filter((m) => m.turn === turn);
    if (!opts.length) return false;
    const mv = opts.find((m) => m.src === base.lane) || opts[0];
    path.dropAfter(base);
    path.setLaneEnd(base, base.lane.len);
    decisions.length = k;
    take(mv, sLo);
    extend(seg.s + 250);
    return true;
  }

  /** Route points from s0 to s1 (for drawing). */
  function slice(s0, s1) {
    const out = [pointAt(s0)];
    for (let i = locate(s0); i < path.cum.length && path.cum[i] < s1; i++) out.push(path.pts[i]);
    out.push(pointAt(s1));
    return out;
  }

  /** The route's lane changes: [{from, to (lane ids), s0, s1 (route arc lengths)}]. */
  const changes = () => path.segs.filter((g) => g.kind === 'change').map((g) => ({ from: g.from.id, to: g.to.id, s0: g.s0, s1: g.s1, ramp: g.from.kind === 'ramp' || g.to.kind === 'ramp' }));

  const startLane = lanes.filter((L) => L.kind !== 'ramp').reduce((a, b) => (b.len > a.len ? b : a));
  // ramps (lane graph): the approach lanes they are entered from (keisim Town.ramps 'from')
  // (first the ones whose approach lane is long enough to start on: no junction before the diverge)
  const hasRamp = (L) => [...L.links.values()].some((k) => k.lane.kind === 'ramp');
  const ramps = lanes.filter((L) => L.kind === 'ramp').map((R) => ({ R, A: lanes.find((L) => L.kind === 'road' && L.links.has(R.id)) }))
    .map((rp) => ({ ...rp, a0: rp.A.links.get(rp.R.id).lo }))
    .sort((p, q) => (q.a0 >= RAMP_LEAD) - (p.a0 >= RAMP_LEAD));
  /** Start where the ego always starts (the town's longest lane, a quarter in), or with `ramp` (index into the
   *  town's ramps) about RAMP_LEAD m before the ramp's diverge, taking it (keisim route.ramp_route): on its approach
   *  lane, or on the road before it (through the junction before; from a road without a ramp of its own if it can, so
   *  the car does not pass another ramp's diverge first, then straight on if it can). */
  function start(ramp = null) {
    const rp = ramp === null || !ramps.length ? null : ramps[ramp % ramps.length];
    if (!rp) return reset(startLane, startLane.len * 0.25);
    want.turn = null;
    const takeRamp = () => {                                // the ramp, from the approach lane (the route's last piece)
      const P = path.last, mv = laneMoves(P.lane, lowOn(P)).find((m) => m.ramp === rp.R && !m.pre);
      if (mv) take(mv, lowOn(P));
    };
    const a0 = rp.a0;                                       // where the deceleration lane begins
    const into = rp.A.pred.filter((c) => c.from.kind === 'road' && c.from.len > 30)
      .sort((p, q) => hasRamp(p.from) - hasRamp(q.from) || (q.turn === 'straight') - (p.turn === 'straight'));
    if (a0 >= RAMP_LEAD || !into.length) return reset(rp.A, Math.max(3.0, Math.min(a0 - RAMP_LEAD, a0 - 30)), takeRamp);
    const c = into[0];
    return reset(c.from, Math.max(3.0, c.from.len - (RAMP_LEAD - a0 - c.len)), () => {
      take({ turn: c.turn, src: c.from, pre: null, conn: c }, lowOn(path.last));
      takeRamp();
    });
  }
  return {
    lanes, reset, track, inputs, request, slice, nextDecision, changes, start,
    get s() { return seg.s; },
    ramps: ramps.length,
  };
}
