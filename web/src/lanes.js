// KeiSim's lane graph in the browser demo (towns exported with lane_graph by scripts/export_town.py: 片側2車線 main
// roads and at-grade left-turn ramps), lane changes, and the growing route paths of src/nav.js (the ego) and
// src/traffic.js (the other cars). Mirrors keisim/route.py: a lane change leaves lane A at A-local a0 and blends into
// lane B over lc metres with a quintic lateral profile, inside A's window (siblings: LC.ENTRY after the junction
// to LC.EXIT before the lane's end; ramps: where the two lanes run side by side, Lane.links); a ramp is a change from
// the approach lane into the ramp lane, the ramp's curve, and a change into the lane of the road it turns into.
// Routes pick the movement at the next junction first (from any lane of the road), then the lane (route.lane_moves,
// pick_move). All coordinates are KeiSim's (x east, y north).

export const LC = {
  ENTRY: 10.0, EXIT: 25.0, LEN: 40.0, MIN: 25.0, START_MARGIN: 8.0, FREE_PROB: 0.25, DS: 0.5,   // keisim/route.py
  CHECK: 12.0, PUSH: 10.0, PUSH_TO: 15.0, GAP0: 2.0, T_REAR: 0.8, T_CLOSE: 1.5,
  GIVE_UP: 8.0,                                                                                 // keisim/traffic.py
};

/** keisim route.blend_weight: lateral share of the target lane at u in [0, 1] along a change. */
export const blendWeight = (u) => u * u * u * (10.0 + u * (-15.0 + 6.0 * u));

/** keisim route.change_room: does a car in the target lane at dx (along it, + ahead) leave room for a change? */
export function changeRoom(dx, vOther, lenOther, v, len) {
  const room = (len + lenOther) / 2 + LC.GAP0;
  return !(-(room + vOther * LC.T_REAR + Math.max(0, vOther - v) * LC.T_CLOSE) < dx &&
    dx < room + Math.max(0, v - vOther) * LC.T_CLOSE);
}

/** keisim route.link_map: A-local arc length s -> B's, for a change from A into B. */
export function linkMap(A, B, s) {
  const lk = A.links.get(B.id);
  if (!lk || lk.scale === null) return s * B.len / A.len;      // siblings: the whole lane, scaled
  return lk.blo + (s - lk.lo) * lk.scale;
}

/** keisim route.change_window: {aMin, aMax, lc}, where on A a change into B can start and its length, or null. */
export function changeWindow(A, sLo = 0, B = null) {
  const lk = B ? A.links.get(B.id) : null;
  let lo, hi;
  if (!lk || lk.scale === null) { lo = Math.max(LC.ENTRY, sLo); hi = A.len - LC.EXIT; }
  else { lo = Math.max(lk.lo, sLo); hi = lk.hi; }
  if (hi - lo < LC.MIN) return null;
  const lc = Math.min(LC.LEN, hi - lo);
  return { aMin: lo, aMax: hi - lc, lc };
}

// ---------------------------------------------------------------- lane polylines
/** Arc lengths, length and end headings of a lane (or connector) {pts}. */
export function measure(L) {
  const p = L.pts, cum = [0];
  for (let i = 1; i < p.length; i++) cum.push(cum[i - 1] + Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]));
  L.cum = cum; L.len = cum[cum.length - 1];
  L.h0 = Math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]);
  L.h1 = Math.atan2(p[p.length - 1][1] - p[p.length - 2][1], p[p.length - 1][0] - p[p.length - 2][0]);
  return L;
}

function segIndex(cum, s) {            // first i >= 1 with cum[i] >= s (clamped)
  let lo = 1, hi = cum.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < s) lo = m + 1; else hi = m; }
  return lo;
}

/** Point at arc length s of a lane (clamped to its ends; linear beyond them for ramps' extrapolated maps). */
export function lanePoint(L, s) {
  const { pts, cum } = L;
  s = Math.max(0, Math.min(L.len, s));
  const i = segIndex(cum, s), a = pts[i - 1], b = pts[i];
  const t = Math.max(0, Math.min(1, (s - cum[i - 1]) / ((cum[i] - cum[i - 1]) || 1e-9)));
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

/** The lane's own polyline from arc length lo to hi (end points interpolated). */
export function lanePiece(L, lo, hi) {
  const out = [lanePoint(L, lo)];
  for (let i = segIndex(L.cum, lo); i < L.pts.length && L.cum[i] < hi - 1e-9; i++) if (L.cum[i] > lo + 1e-9) out.push(L.pts[i]);
  out.push(lanePoint(L, hi));
  return out;
}

/** Closest point of lane L to p: {s, lat} (lat > 0: left of the lane). */
export function projectOn(L, p) {
  let best = Infinity, bs = 0, lat = 0;
  for (let i = 1; i < L.pts.length; i++) {
    const a = L.pts[i - 1], b = L.pts[i], dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy || 1e-9;
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2));
    const qx = a[0] + dx * t, qy = a[1] + dy * t, d = Math.hypot(p[0] - qx, p[1] - qy);
    if (d < best) { best = d; bs = L.cum[i - 1] + t * Math.sqrt(l2); lat = (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) >= 0 ? d : -d; }
  }
  return { s: bs, lat };
}

/** Lane heading at arc length s. */
export function laneHeading(L, s) {
  const i = segIndex(L.cum, Math.max(0, Math.min(L.len, s)));
  return Math.atan2(L.pts[i][1] - L.pts[i - 1][1], L.pts[i][0] - L.pts[i - 1][0]);
}

// ---------------------------------------------------------------- the exported lane graph
/** Lanes of a town exported with lane_graph, in the shape src/nav.js buildLanes gives: road lanes (and the ramps'
 *  lanes, kind 'ramp', no junctions at their ends) {id, kind, road, from, to, index (0 = kerb lane), n, width, speed,
 *  signal (phase, or null), pts, len, h0, h1, siblings, links: Map(id -> {lane, lo, hi, blo, scale}), next: [{lane,
 *  turn, conn}], pred: [conn]}; connectors {from, to, junction, turn, speed, pts, cum, len, yields, wait} with
 *  KeiSim's own give-way table (Town._make_yields). lanes.graph: {conns, truck (Set of lanes, or null)}. */
export function buildGraph(data) {
  const g = data.lane_graph;
  const lanes = g.lanes.map((l) => measure({
    id: l.id, kind: l.kind, road: l.road, from: l.from, to: l.to, index: l.index, n: l.n, width: l.width,
    speed: l.speed, signal: l.signal ?? null, junction: l.junction ?? null, turn: l.turn ?? null,
    stopSign: l.stop_sign ?? null, pts: l.pts, links: new Map(), siblings: [], next: [], pred: [],
  }));
  for (const l of g.lanes) {
    const L = lanes[l.id];
    L.siblings = l.sib.map((i) => lanes[i]);
    for (const [b, lo, hi, blo, scale] of l.links) L.links.set(b, { lane: lanes[b], lo, hi, blo, scale });
  }
  const conns = g.conns.map((c) => measure({ from: lanes[c.from], to: lanes[c.to], junction: c.junction, turn: c.turn,
    speed: c.speed, pts: c.pts, yields: [], wait: c.wait ?? null }));
  g.conns.forEach((c, i) => {
    const conn = conns[i];
    for (const [b, here, there] of c.yields || []) conn.yields.push({ b: conns[b], here, there });
    conn.from.next.push({ lane: conn.to, turn: conn.turn, conn });
    conn.to.pred.push(conn);
  });
  lanes.graph = { conns, truck: g.truck_lanes ? new Set(g.truck_lanes.map((i) => lanes[i])) : null };
  return lanes;
}

/** keisim route.lane_moves: the movements from lane `cur` at the end of its road for a car that may change lanes
 *  from cur-local sLo on: [{turn, src, pre, conn} | {turn, src, pre, ramp, merge}], src the lane it is made from
 *  (cur, or pre: a sibling it changes into first), conn the junction connector, or ramp (a ramp lane) and merge (the
 *  lane it merges into). truck: only the movements into those lanes, unless none is (over all the road's lanes;
 *  keisim filters each lane's connectors on its own, so that a truck may change into a lane whose exits are all
 *  narrow). */
export function laneMoves(cur, sLo = 0, truck = null) {
  const srcs = [[cur, sLo]];
  for (const b of cur.siblings) {
    const w = changeWindow(cur, sLo, b);
    if (w) srcs.push([b, linkMap(cur, b, w.aMin + w.lc)]);
  }
  const moves = [];
  for (const [src, lo] of srcs) {
    const pre = src === cur ? null : src;
    for (const o of src.next) moves.push({ turn: o.turn, src, pre, conn: o.conn });
    for (const lk of src.links.values()) {
      const R = lk.lane;
      if (R.kind !== 'ramp' || !changeWindow(src, lo, R)) continue;
      moves.push({ turn: R.turn, src, pre, ramp: R, merge: R.links.values().next().value.lane });
    }
  }
  if (!truck) return moves;
  const wide = moves.filter((m) => truck.has(m.conn ? m.conn.to : m.merge));
  return wide.length ? wide : moves;
}

/** keisim route.pick_move: the movement first (uniform over the turns there are), then the lane: the current one if
 *  it has it (else a change; LC.FREE_PROB of the time a change it does not need). */
export function pickMove(rnd, moves, cur) {
  const turns = ['left', 'straight', 'right'].filter((t) => moves.some((m) => m.turn === t));
  const t = turns[Math.floor(rnd() * turns.length)];
  const opts = moves.filter((m) => m.turn === t);
  const own = opts.filter((m) => m.src === cur), other = opts.filter((m) => m.src !== cur);
  const pick = own.length && (!other.length || rnd() >= LC.FREE_PROB) ? own : other;
  return pick[Math.floor(rnd() * pick.length)];
}

/** Did the path just change into lane piece P from a sibling (no change straight back: keisim Traffic._extend)? */
export const changedInto = (path, P) => {
  const prev = path.segs[path.segs.indexOf(P) - 1];
  return !!prev && prev.kind === 'change' && P.lane.siblings.includes(prev.from);
};

// ---------------------------------------------------------------- route paths
/** A route's growing polyline {pts, cum} made of segments: lane pieces {kind 'lane', lane, lo, hi} (the lane's own
 *  polyline from lane-local lo to hi), junction connectors {kind 'conn', conn, from, junction} and lane changes
 *  {kind 'change', from, to, a0, lc, aMin, aMax}; each with its route arc lengths s0, s1 and n1 (points so far). */
export class Path {
  constructor() { this.pts = []; this.cum = []; this.segs = []; this.hint = 1; }

  get end() { return this.cum.length ? this.cum[this.cum.length - 1] : 0; }
  get last() { return this.segs[this.segs.length - 1]; }

  push(p) {
    if (this.pts.length) {
      const q = this.pts[this.pts.length - 1], d = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (d < 1e-6) return;
      this.cum.push(this.end + d);
    } else this.cum.push(0);
    this.pts.push(p);
  }

  add(seg, points) {
    seg.s0 = this.end;
    for (const p of points) this.push(p);
    seg.s1 = this.end; seg.n1 = this.pts.length;
    this.segs.push(seg);
    return seg;
  }

  addLane(lane, lo = 0, hi = lane.len) {
    return this.add({ kind: 'lane', lane, lo, hi }, lo === 0 && hi === lane.len ? lane.pts : lanePiece(lane, lo, hi));
  }

  addConn(conn) { return this.add({ kind: 'conn', junction: conn.junction, from: conn.from, conn }, conn.pts); }

  addChange(A, B, a0, lc, win = null) {
    const m = Math.max(2, Math.ceil(lc / LC.DS)), pts = [];
    for (let i = 0; i <= m; i++) {
      const u = i / m, sa = a0 + u * lc, w = blendWeight(u);
      const pa = lanePoint(A, sa), pb = lanePoint(B, linkMap(A, B, sa));
      pts.push([(1 - w) * pa[0] + w * pb[0], (1 - w) * pa[1] + w * pb[1]]);
    }
    return this.add({ kind: 'change', from: A, to: B, a0, lc, aMin: win ? win.aMin : a0, aMax: win ? win.aMax : a0 }, pts);
  }

  /** Remove everything after segment `seg`. */
  dropAfter(seg) {
    const k = this.segs.indexOf(seg);
    this.segs.length = k + 1;
    this.pts.length = seg.n1; this.cum.length = seg.n1;
    this.hint = 1;
  }

  /** Move the end of the last segment, a lane piece, to lane-local hi (the path up to there stays as it was). */
  setLaneEnd(seg, hi) {
    if (hi >= seg.hi) {
      for (const p of lanePiece(seg.lane, seg.hi, hi)) this.push(p);
    } else {
      const sCut = seg.s0 + (hi - seg.lo);
      let n = this.pts.length;
      while (n > 1 && this.cum[n - 1] > sCut - 1e-9) n--;
      this.pts.length = n; this.cum.length = n;
      this.push(lanePoint(seg.lane, hi));
    }
    seg.hi = hi; seg.s1 = this.end; seg.n1 = this.pts.length;
    this.hint = 1;
  }

  /** Lane-local arc length of route position s on lane piece `seg` (A-local on a change). */
  localOn(seg, s) {
    if (seg.kind === 'lane') return seg.lo + (s - seg.s0);
    if (seg.kind === 'change') return seg.a0 + seg.lc * Math.max(0, Math.min(1, (s - seg.s0) / ((seg.s1 - seg.s0) || 1e-9)));
    return s - seg.s0;
  }

  /** Drop what lies more than `keep` m behind arc length s; returns the shift to subtract from s. */
  trim(s, keep = 30) {
    let i = 1;
    while (i < this.cum.length - 1 && this.cum[i] < s - keep) i++;
    if (i < 50) return 0;
    const base = this.cum[i - 1];
    this.pts = this.pts.slice(i - 1);
    this.cum = this.cum.slice(i - 1).map((c) => c - base);
    this.segs = this.segs.filter((g) => g.s1 > base);
    for (const g of this.segs) { g.s0 -= base; g.s1 -= base; g.n1 -= i - 1; }
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
    const i = this.locate(s), a = this.pts[i - 1], b = this.pts[i];
    return Math.atan2(b[1] - a[1], b[0] - a[0]);
  }

  segAt(s) {
    for (let k = 0; k < this.segs.length; k++) if (s < this.segs[k].s1 || k === this.segs.length - 1) return k;
    return this.segs.length - 1;
  }

  /** Append movement `mv` (laneMoves) to the path, whose last segment is a lane piece to its lane's end; sLo: from
   *  where (lane-local) a change may start. Returns {conn: its segment} or {ramp: {lane, junction, turn, c1, c2}}. */
  addMove(mv, sLo) {
    let P = this.last;
    if (mv.pre) {
      const w = changeWindow(P.lane, sLo, mv.pre) || fallbackWindow(P.lane, sLo);
      this.setLaneEnd(P, w.aMin);
      this.addChange(P.lane, mv.pre, w.aMin, w.lc, w);
      P = this.addLane(mv.pre, Math.min(mv.pre.len, linkMap(P.lane, mv.pre, w.aMin + w.lc)));
      sLo = P.lo;
    }
    if (mv.ramp) {
      const A = P.lane, R = mv.ramp, C = mv.merge;
      const w = changeWindow(A, sLo, R) || fallbackWindow(A, sLo);
      this.setLaneEnd(P, w.aMin);
      const c1 = this.addChange(A, R, w.aMin, w.lc, w);
      const rp = this.addLane(R, linkMap(A, R, w.aMin + w.lc));
      const w2 = changeWindow(R, rp.lo, C) || fallbackWindow(R, rp.lo);
      this.setLaneEnd(rp, w2.aMin);
      const c2 = this.addChange(R, C, w2.aMin, w2.lc, w2);
      this.addLane(C, Math.min(C.len, linkMap(R, C, w2.aMin + w2.lc)));
      return { ramp: { lane: R, junction: R.junction, turn: R.turn, from: A, c1, c2 } };
    }
    const conn = this.addConn(mv.conn);
    this.addLane(mv.conn.to);
    return { conn };
  }

  /** Lane change `ch` (a segment of this path) to start at A-local a0 instead (length lc): the path is rebuilt from
   *  there, with the same movements after it. Returns the new change segment. */
  retime(ch, a0, lc = ch.lc, aMin = ch.aMin, aMax = ch.aMax) {
    const k = this.segs.indexOf(ch), P = this.segs[k - 1];
    const rest = this.segs.slice(k + 1);
    this.dropAfter(P);
    this.setLaneEnd(P, a0);
    const out = this.addChange(ch.from, ch.to, a0, lc, { aMin, aMax });
    let lo = linkMap(ch.from, ch.to, a0 + lc);
    for (let i = 0; i < rest.length; i++) {
      const g = rest[i];
      if (g.kind === 'lane') {
        const hi = i === 0 ? Math.max(g.hi, Math.min(g.lane.len, lo)) : g.hi;
        this.addLane(g.lane, i === 0 ? Math.min(lo, hi) : g.lo, hi);
      } else if (g.kind === 'conn') this.addConn(g.conn);
      else {
        const p = this.last, a = Math.max(g.a0, p.lo);         // (a ramp's merge: not before the ramp is entered)
        if (a > p.hi) this.setLaneEnd(p, a);
        this.addChange(g.from, g.to, a, g.lc, g);
        lo = linkMap(g.from, g.to, a + g.lc);
        if (rest[i + 1] && rest[i + 1].kind === 'lane') { rest[i + 1] = { ...rest[i + 1], lo: Math.min(lo, rest[i + 1].hi) }; }
      }
    }
    return out;
  }
}

/** keisim Route._build for a window too short: change over what is left. */
function fallbackWindow(A, sLo) {
  const a = Math.max(sLo, Math.min(A.len - LC.MIN, A.len / 2));
  return { aMin: a, aMax: a, lc: Math.max(1.0, Math.min(LC.MIN, A.len - a - 1.0)) };
}
