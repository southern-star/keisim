// Route and navigation inputs for the in-browser KeiPilot (src/pilot.js), mirroring keisim/roadnet.py and
// keisim/route.py: one lane per road and direction, junction connectors as arc-like Béziers, and the model's
// two route inputs, the turn at the next signalized junction (command) and the point 4 m past that junction's
// exit (target point). All coordinates are KeiSim's (x east, y north, yaw counter-clockwise).
export const CMD = { left: 0, straight: 1, right: 2 };

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const heading = (a, b) => Math.atan2(b[1] - a[1], b[0] - a[0]);

/** Offset a polyline to the left by d (negative: right), using the averaged tangent at each vertex. */
function offsetPolyline(pts, d) {
  return pts.map((p, i) => {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
    const h = heading(a, b);
    return [p[0] - Math.sin(h) * d, p[1] + Math.cos(h) * d];
  });
}

/** keisim.geometry.arc_bezier: smooth, near-circular connector from pose (p0, h0) to pose (p1, h1). */
function arcBezier(p0, h0, p1, h1, n = 24) {
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

export function createNavigator(data) {
  const cfg = data.cfg;
  const off = (cfg.left_hand_traffic === false ? -1 : 1) * (cfg.lane_width || 3.5) / 2;
  const junctions = new Map(data.junctions.map((j) => [j.id, j]));
  const lanes = [];
  for (const r of data.roads) {
    lanes.push({ id: lanes.length, road: r.id, from: r.a, to: r.b, pts: offsetPolyline(r.center, off) });
    lanes.push({ id: lanes.length, road: r.id, from: r.b, to: r.a, pts: offsetPolyline(r.center, -off).reverse() });
  }
  for (const L of lanes) {
    L.len = 0;
    for (let i = 1; i < L.pts.length; i++) L.len += Math.hypot(L.pts[i][0] - L.pts[i - 1][0], L.pts[i][1] - L.pts[i - 1][1]);
    L.h0 = heading(L.pts[0], L.pts[1]);
    L.h1 = heading(L.pts[L.pts.length - 2], L.pts[L.pts.length - 1]);
  }
  // junction connectors: from every incoming lane to every outgoing lane of another road (no U-turns)
  for (const L of lanes) {
    L.next = lanes.filter((o) => o.from === L.to && o.road !== L.road).map((o) => {
      const dh = wrap(o.h0 - L.h1);
      return { lane: o, turn: dh > 0.5 ? 'left' : dh < -0.5 ? 'right' : 'straight' };
    });
  }

  // ---------------------------------------------------------------- the route: a growing polyline of lanes
  let pts, cum, decisions, last, choice, seg;
  const want = { turn: null };                      // the driver's pick for the next decision junction

  function append(points) {
    for (const p of points) {
      if (pts.length) {
        const q = pts[pts.length - 1], d = Math.hypot(p[0] - q[0], p[1] - q[1]);
        if (d < 1e-6) continue;
        cum.push(cum[cum.length - 1] + d);
      } else cum.push(0);
      pts.push(p);
    }
  }

  /** Way out of the junction at the end of lane `L`: the driver's pick at the first junction with a choice that
   *  the car has not entered yet, otherwise a random one (the only one at a bend). */
  function pick(L) {
    const J = junctions.get(L.to);
    const opts = L.next;
    if (!opts.length) return null;
    if (J && J.signalized && want.turn && !decisions.some((d) => d.sIn > seg.s + 1)) {
      const o = opts.find((c) => c.turn === want.turn);
      want.turn = null;
      if (o) return o;
    }
    return opts[Math.floor(Math.random() * opts.length)];
  }

  function extend(until) {
    while (cum[cum.length - 1] < until) {
      const c = pick(last);
      if (!c) break;
      const J = junctions.get(last.to);
      const conn = arcBezier(last.pts[last.pts.length - 1], last.h1, c.lane.pts[0], c.lane.h0);
      const sIn = cum[cum.length - 1];
      append(conn);
      const sOut = cum[cum.length - 1];
      if (J && J.signalized) decisions.push({ junction: J.id, turn: c.turn, sIn, sOut, lane: last });
      append(c.lane.pts);
      last = c.lane;
    }
  }

  /** Start on lane `L` at arc length `s0`; returns the pose there. */
  function reset(L, s0) {
    pts = []; cum = []; decisions = []; last = L;
    seg = { i: 0, s: 0 };
    append(L.pts);
    seg.s = Math.min(s0, L.len - 1);
    extend(seg.s + 250);
    const p = pointAt(seg.s), q = pointAt(seg.s + 0.5);
    return { x: p[0], y: p[1], yaw: heading(p, q) };
  }

  function locate(s) {
    s = Math.max(0, Math.min(s, cum[cum.length - 1]));
    let i = Math.max(1, Math.min(seg.i, cum.length - 1));
    while (i > 1 && cum[i - 1] > s) i--;
    while (i < cum.length - 1 && cum[i] < s) i++;
    return i;
  }
  function pointAt(s) {
    const i = locate(s), a = pts[i - 1], b = pts[i];
    const t = Math.max(0, Math.min(1, (s - cum[i - 1]) / ((cum[i] - cum[i - 1]) || 1e-9)));
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }

  /** Follow the car along the route (closest point within a window ahead); returns its lateral offset [m]. */
  function track(x, y) {
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

  /** Next junction with a real choice that has not been left yet (keisim Route.next_decision). */
  const nextDecision = () => decisions.find((d) => d.sOut > seg.s + 0.5) || null;

  /** Command and target point (world) for the model, as keisim Route.command / target_point_world. */
  function inputs() {
    const j = nextDecision();
    const end = cum[cum.length - 1];
    if (!j) return { cmd: CMD.straight, tp: pointAt(end), dist: null };
    return { cmd: CMD[j.turn], tp: pointAt(Math.min(j.sOut + 4, end)), dist: j.sIn - seg.s, turn: j.turn };
  }

  /** The driver asks for `turn` at the next decision junction not yet entered. Returns false if impossible. */
  function request(turn) {
    const k = decisions.findIndex((d) => d.sIn > seg.s + 1);
    if (k < 0) { want.turn = turn; return true; }
    const d = decisions[k];
    if (!d.lane.next.some((c) => c.turn === turn)) return false;
    if (d.turn === turn) return true;
    // cut the route at that junction's entry and grow it again with the new pick
    const keep = locate(d.sIn);
    pts.length = keep + 1; cum.length = keep + 1;
    decisions.length = k;
    last = d.lane;
    want.turn = turn;
    extend(seg.s + 250);
    return true;
  }

  /** Route points from s0 to s1 (for drawing). */
  function slice(s0, s1) {
    const out = [pointAt(s0)];
    for (let i = locate(s0); i < cum.length && cum[i] < s1; i++) out.push(pts[i]);
    out.push(pointAt(s1));
    return out;
  }

  const startLane = lanes.reduce((a, b) => (b.len > a.len ? b : a));
  return {
    lanes, reset, track, inputs, request, slice, nextDecision,
    get s() { return seg.s; },
    start: () => reset(startLane, startLane.len * 0.25),
  };
}
