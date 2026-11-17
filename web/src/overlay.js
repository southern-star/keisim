// The plan as a band of the car's width and the area the car's body sweeps (src/pilot.js draws them: ?overlay=band,
// ?overlay=swept): their geometry on the ground, in KeiSim coordinates (x, y in metres; yaw counter-clockwise).
// No browser needed: tools/overlay_check.mjs checks it in node.

/** A smooth curve through the points [[x, y], ...], a point about every `step` m: {P, s} with the arc length s of
 *  every point. Centripetal Catmull-Rom (alpha 0.5; the ends go on by a mirrored point): the uniform kind overshoots
 *  and loops where a path of points 2 m apart turns sharply. Done on the ground, before the camera: every piece
 *  between two of the points is short enough to be drawn as a straight line. */
export function densify(pts, step = 0.25) {
  const n = pts.length;
  if (n < 2) return { P: pts.map(([x, y]) => [x, y]), s: pts.map(() => 0) };
  const Q = [[2 * pts[0][0] - pts[1][0], 2 * pts[0][1] - pts[1][1]], ...pts,
    [2 * pts[n - 1][0] - pts[n - 2][0], 2 * pts[n - 1][1] - pts[n - 2][1]]];
  const knot = (a, b) => Math.sqrt(Math.max(Math.hypot(b[0] - a[0], b[1] - a[1]), 1e-6));
  const mix = (a, b, ta, tb, t) => { const u = (tb - t) / (tb - ta), v = (t - ta) / (tb - ta); return [u * a[0] + v * b[0], u * a[1] + v * b[1]]; };
  const P = [[pts[0][0], pts[0][1]]];
  for (let i = 1; i + 2 < Q.length; i++) {
    const p0 = Q[i - 1], p1 = Q[i], p2 = Q[i + 1], p3 = Q[i + 2];
    const t1 = knot(p0, p1), t2 = t1 + knot(p1, p2), t3 = t2 + knot(p2, p3);
    const m = Math.max(1, Math.ceil(Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / step));
    for (let k = 1; k <= m; k++) {
      const t = t1 + (t2 - t1) * k / m;
      const a1 = mix(p0, p1, 0, t1, t), a2 = mix(p1, p2, t1, t2, t), a3 = mix(p2, p3, t2, t3, t);
      P.push(mix(mix(a1, a2, 0, t2, t), mix(a2, a3, t1, t3, t), t1, t2, t));
    }
  }
  const s = [0];
  for (let i = 1; i < P.length; i++) s.push(s[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]));
  return { P, s };
}

/** The arc length, along the polyline (P, s), of its point nearest to (x, y). */
export function locate({ P, s }, x, y) {
  let best = Infinity, at = 0;
  for (let i = 0; i + 1 < P.length; i++) {
    const ax = P[i][0], ay = P[i][1], dx = P[i + 1][0] - ax, dy = P[i + 1][1] - ay, l2 = dx * dx + dy * dy;
    const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
    const d = Math.hypot(ax + t * dx - x, ay + t * dy - y);
    if (d < best) { best = d; at = s[i] + t * (s[i + 1] - s[i]); }
  }
  return at;
}

/** Gives the polyline (P, s) a point at arc length `at` (unless it has one there) and returns that point's index. */
export function splitAt({ P, s }, at) {
  const n = s.length;
  if (at <= s[0]) return 0;
  if (at >= s[n - 1]) return n - 1;
  let i = 1;
  while (s[i] < at) i++;
  if (s[i] - at < 1e-6) return i;
  if (at - s[i - 1] < 1e-6) return i - 1;
  const t = (at - s[i - 1]) / (s[i] - s[i - 1]);
  P.splice(i, 0, [P[i - 1][0] + t * (P[i][0] - P[i - 1][0]), P[i - 1][1] + t * (P[i][1] - P[i - 1][1])]);
  s.splice(i, 0, at);
  return i;
}

/** The band's edges: every point of the polyline moved `half` m to the left and to the right of the direction of
 *  travel there (the direction from the point before to the point after). */
export function bandEdges(P, half) {
  const L = [], R = [];
  for (let i = 0; i < P.length; i++) {
    const a = P[Math.max(i - 1, 0)], b = P[Math.min(i + 1, P.length - 1)];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, nx = -(b[1] - a[1]) / d * half, ny = (b[0] - a[0]) / d * half;
    L.push([P[i][0] + nx, P[i][1] + ny]);
    R.push([P[i][0] - nx, P[i][1] - ny]);
  }
  return { L, R };
}

/** The corners of a car's body (centre x, y): front left, front right, rear right, rear left, as keisim box_corners. */
export function boxCorners(x, y, yaw, length, width) {
  const c = Math.cos(yaw), s = Math.sin(yaw), hl = length / 2, hw = width / 2;
  return [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]].map(([lx, ly]) => [x + c * lx - s * ly, y + s * lx + c * ly]);
}

/** Convex hull, counter-clockwise (Andrew's monotone chain). */
export function hull(points) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (seq) => {
    const h = [];
    for (const q of seq) {
      while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], q) <= 0) h.pop();
      h.push(q);
    }
    h.pop();
    return h;
  };
  return [...half(p), ...half(p.reverse())];
}

/** The ground the car's body passes over along the poses [[x, y, yaw], ...] of a rollout, as convex pieces: the
 *  hull of the body at each two poses in a row (the bodies alone would leave saw teeth between the steps on the
 *  outside of a turn). All counter-clockwise: filled together (non-zero winding) they make the union. Poses less than
 *  `minStep` m and 0.5 deg from the last one kept are skipped (a car that stands is one piece). */
export function sweptHulls(poses, length, width, minStep = 0.02) {
  const kept = [];
  for (const p of poses) {
    const q = kept[kept.length - 1];
    if (!q || Math.hypot(p[0] - q[0], p[1] - q[1]) >= minStep || Math.abs(p[2] - q[2]) >= 0.0087) kept.push(p);
  }
  const boxes = kept.map(([x, y, yaw]) => boxCorners(x, y, yaw, length, width));
  if (boxes.length === 1) return [hull(boxes[0])];
  const out = [];
  for (let i = 0; i + 1 < boxes.length; i++) out.push(hull([...boxes[i], ...boxes[i + 1]]));
  return out;
}
