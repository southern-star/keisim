// KeiSim town export (scripts/export_town.py) -> three.js frame + spatial queries.
//
// KeiSim: metres, +x east, +y north.  three.js: +X east, -Z north, +Y up.
//   X = x - ox,  Z = -(y - oy)     with (ox, oy) = data.origin (bbox centre)

export const SURF = { terrain: 0, road: 1, sidewalk: 2, building: 3 };
export const SIDEWALK_Y = 0.15;   // sidewalk slab height above the road (KeiSim itself is flat)

export async function loadTown(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} — export it first:  uv run scripts/export_town.py --town <seed>`);
  const data = await res.json();
  if (data.format !== 'keisim-town') throw new Error(`${url}: not a KeiSim town export`);
  return prepareTown(data);
}

export function prepareTown(data) {
  const [ox, oy] = data.origin;
  /** KeiSim point [x, y] -> [X, Z] */
  const P = (p) => [p[0] - ox, oy - p[1]];
  /** KeiSim heading -> unit vector [X, Z] */
  const dirOf = (yaw) => [Math.cos(yaw), -Math.sin(yaw)];
  /** three.js rotation.y that turns local +Z toward KeiSim heading `yaw` */
  const rotYOf = (yaw) => Math.atan2(Math.cos(yaw), -Math.sin(yaw));
  const cfg = data.cfg;

  const [bx0, by0, bx1, by1] = data.bbox;
  const bounds = { x0: bx0 - ox, x1: bx1 - ox, z0: oy - by1, z1: oy - by0 };

  // ------------------------------------------------------------------ roads (centre lines)
  const roads = data.roads.map((r) => {
    const pts = r.center.map(P);
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const len = cum[cum.length - 1];
    /** {x, z, tx, tz, nx, nz}: point, unit tangent (a -> b) and left normal at arc length s */
    const at = (s) => {
      s = Math.min(Math.max(s, 0), len);
      let i = 1; while (i < cum.length - 1 && cum[i] < s) i++;
      const a = pts[i - 1], b = pts[i], seg = cum[i] - cum[i - 1] || 1e-9, t = (s - cum[i - 1]) / seg;
      const tx = (b[0] - a[0]) / seg, tz = (b[1] - a[1]) / seg;
      return { x: a[0] + (b[0] - a[0]) * t, z: a[1] + (b[1] - a[1]) * t, tx, tz, nx: tz, nz: -tx };
    };
    return { ...r, pts, cum, len, at };
  });

  const junctions = data.junctions.map((j) => ({ ...j, p: P(j.pos) }));
  const signals = data.signals.map((s) => ({ ...s, headP: P(s.head), poleP: P(s.pole), rotY: rotYOf(s.yaw) }));
  const trees = data.trees.map((t) => ({ ...t, p: P([t.x, t.y]) }));

  // ------------------------------------------------------------------ building lots
  // KeiSim building box: centre (x, y), yaw = heading of the road it faces, l along the road,
  // w deep, set back `setback` m behind the sidewalk's outer edge, on side `side` of the road.
  // Lot frame (Sakuragaoka convention): origin at the frontage centre on the sidewalk edge,
  // local +Z faces the street, lot spans local x in [-w/2, w/2], z in [-depth, 0].
  const buildings = data.buildings.map((b, i) => {
    const nx = -Math.sin(b.yaw) * b.side, ny = Math.cos(b.yaw) * b.side;   // KeiSim normal road -> building
    const back = b.w / 2 + b.setback;
    const front = P([b.x - nx * back, b.y - ny * back]);
    const faceYaw = Math.atan2(-ny, -nx);                                    // toward the road
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, c]) => {
      const ux = Math.cos(b.yaw), uy = Math.sin(b.yaw);
      return P([b.x + ux * a * b.l / 2 - uy * c * b.w / 2, b.y + uy * a * b.l / 2 + ux * c * b.w / 2]);
    });
    return { ...b, i, corners, lot: { x: front[0], z: front[1], rotY: rotYOf(faceYaw), w: b.l, depth: b.setback + b.w } };
  });

  const T = { data, cfg, origin: data.origin, P, dirOf, rotYOf, bounds, roads, junctions, signals, trees, buildings };

  // ------------------------------------------------------------------ surface grid + distance to road
  const grid = rasterize(T, 0.25);
  T.grid = grid;
  const cell = (x, z) => {
    const c = Math.floor((x - grid.x0) / grid.res), r = Math.floor((z - grid.z0) / grid.res);
    return c >= 0 && c < grid.W && r >= 0 && r < grid.H ? r * grid.W + c : -1;
  };
  /** SURF class at (x, z) (terrain outside the grid) */
  T.surfaceAt = (x, z) => { const k = cell(x, z); return k < 0 ? SURF.terrain : grid.cls[k]; };
  /** metres to the nearest road surface (0 on the road; large far away) */
  T.roadDist = (x, z) => { const k = cell(x, z); return k < 0 ? 99 : grid.dist[k]; };
  T.heightAt = (x, z) => (T.surfaceAt(x, z) === SURF.sidewalk ? SIDEWALK_Y : 0);

  // ------------------------------------------------------------------ surroundings
  const m = 12;
  T.play = { x0: bounds.x0 - m, x1: bounds.x1 + m, z0: bounds.z0 - m, z1: bounds.z1 + m };
  const g0 = 14, g1 = 200;   // far-town ring (low-detail houses) between these distances from the bbox
  const B = bounds;
  T.farTown = [
    { x0: B.x0 - g1, x1: B.x1 + g1, z0: B.z0 - g1, z1: B.z0 - g0 },   // north
    { x0: B.x0 - g1, x1: B.x1 + g1, z0: B.z1 + g0, z1: B.z1 + g1 },   // south
    { x0: B.x0 - g1, x1: B.x0 - g0, z0: B.z0 - g0, z1: B.z1 + g0 },   // west
    { x0: B.x1 + g0, x1: B.x1 + g1, z0: B.z0 - g0, z1: B.z1 + g0 },   // east
  ];
  T.farSkip = (x, z) => x > B.x0 - g0 && x < B.x1 + g0 && z > B.z0 - g0 && z < B.z1 + g0;

  T.views = makeViews(T);
  return T;
}

// ==================================================================== rasterisation
function rasterize(T, res) {
  const { x0, x1, z0, z1 } = T.bounds;
  const W = Math.ceil((x1 - x0) / res), H = Math.ceil((z1 - z0) / res);
  const grid = { res, x0, z0, W, H, cls: new Uint8Array(W * H), dist: new Float32Array(W * H).fill(99), ok: false };
  const doc = globalThis.document;
  const cv = doc && doc.createElement ? doc.createElement('canvas') : null;
  const g = cv && (cv.width = W, cv.height = H, cv.getContext('2d', { willReadFrequently: true }));
  if (!g || !g.getImageData) return grid;
  const px = (p) => { const q = T.P(p); return [(q[0] - x0) / res, (q[1] - z0) / res]; };
  const ring = (r) => { r.forEach((p, i) => { const q = px(p); if (i) g.lineTo(q[0], q[1]); else g.moveTo(q[0], q[1]); }); g.closePath(); };
  const fill = (rings, v) => { g.fillStyle = `rgb(${v},0,0)`; g.beginPath(); for (const r of rings) ring(r); g.fill('evenodd'); };
  g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  const shapeRings = (shapes) => shapes.flatMap((s) => [s.outer, ...s.holes]);
  fill(shapeRings(T.data.surfaces.sidewalk), 80);
  fill(shapeRings(T.data.surfaces.road), 160);
  g.fillStyle = 'rgb(240,0,0)';
  for (const b of T.buildings) { g.beginPath(); b.corners.forEach((q, i) => { const c = [(q[0] - x0) / res, (q[1] - z0) / res]; if (i) g.lineTo(c[0], c[1]); else g.moveTo(c[0], c[1]); }); g.closePath(); g.fill(); }
  const img = g.getImageData(0, 0, W, H).data;
  const cls = grid.cls;
  let any = 0;
  for (let i = 0; i < W * H; i++) {
    const v = img[i * 4];
    cls[i] = v < 40 ? 0 : v < 120 ? 2 : v < 200 ? 1 : 3;
    any |= cls[i];
  }
  grid.ok = any > 0;
  // two-pass chamfer distance (metres) to the nearest road cell
  const d = grid.dist, D1 = res, D2 = res * Math.SQRT2;
  for (let i = 0; i < W * H; i++) if (cls[i] === 1) d[i] = 0;
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const i = r * W + c; let v = d[i];
    if (c > 0) v = Math.min(v, d[i - 1] + D1);
    if (r > 0) { v = Math.min(v, d[i - W] + D1); if (c > 0) v = Math.min(v, d[i - W - 1] + D2); if (c < W - 1) v = Math.min(v, d[i - W + 1] + D2); }
    d[i] = v;
  }
  for (let r = H - 1; r >= 0; r--) for (let c = W - 1; c >= 0; c--) {
    const i = r * W + c; let v = d[i];
    if (c < W - 1) v = Math.min(v, d[i + 1] + D1);
    if (r < H - 1) { v = Math.min(v, d[i + W] + D1); if (c < W - 1) v = Math.min(v, d[i + W + 1] + D2); if (c > 0) v = Math.min(v, d[i + W - 1] + D2); }
    d[i] = v;
  }
  return grid;
}

// ==================================================================== camera spots
/** Player yaw (degrees; 0 = north/-Z, 90 = west) looking along (dx, dz). */
export const yawOf = (dx, dz) => Math.atan2(-dx, -dz) * 180 / Math.PI;

/** Side of the road (+1 = left of a -> b) that carries the utility poles (see world/poles.js). */
export const poleSide = (road) => (road.id % 2 === 0 ? 1 : -1);

function makeViews(T) {
  const { cfg } = T;
  const walkOff = cfg.road_half_width + 0.85;          // on the sidewalk, kerb side (trees stand at the back edge)
  const byDeg = (j) => T.roads.filter((r) => r.a === j.id || r.b === j.id);
  const obstacles = [...T.trees.map((t) => t.p), ...T.signals.map((s) => s.poleP)];
  // hero: on the sidewalk ~34 m before the signalised junction nearest to the town centre, facing it
  const sig = T.junctions.filter((j) => j.signalized).sort((a, b) => Math.hypot(...a.p) - Math.hypot(...b.p));
  const J = sig[0] || T.junctions[0];
  const arms = byDeg(J).sort((a, b) => b.len - a.len);
  const views = [];
  /** Walking eye on the pole-free sidewalk of `road`, `dist` m before junction `fromJ`, facing it;
   *  slides along the road until trunks and masts are >= 2.4 m away. */
  const onRoad = (road, fromJ, dist, label = '') => {
    const atA = road.a === fromJ.id;
    const sgn = atA ? -1 : 1;                             // travel direction toward the junction
    const side = -poleSide(road) * sgn;                   // left-of-travel sign of the pole-free side
    for (const dd of [0, 3, -3, 6, -6, 9, -9, 12]) {
      const d = Math.min(Math.max(dist + dd, 6), road.len - 4);
      const f = road.at(atA ? d : road.len - d);
      const tx = f.tx * sgn, tz = f.tz * sgn;
      const nx = tz * side, nz = -tx * side;
      const x = f.x + nx * walkOff, z = f.z + nz * walkOff;
      if (obstacles.every((o) => Math.hypot(o[0] - x, o[1] - z) > 2.4) || dd === 12) return { x, z, yaw: yawOf(tx, tz), pitch: 1.5, label };
    }
    return null;
  };
  const push = (v) => { if (v) views.push(v); };
  if (arms[0]) push(onRoad(arms[0], J, 34, 'street'));
  if (arms[1]) push(onRoad(arms[1], J, 16, 'junction'));
  const c = [(T.bounds.x0 + T.bounds.x1) / 2, (T.bounds.z0 + T.bounds.z1) / 2];
  const span = Math.max(T.bounds.x1 - T.bounds.x0, T.bounds.z1 - T.bounds.z0);
  views.push({ x: c[0] + span * 0.08, y: span * 0.2 + 14, z: T.bounds.z1 + span * 0.06, yaw: 0, pitch: -26, label: 'overview', fly: true });
  if (arms[2]) push(onRoad(arms[2], J, 55, 'block'));
  const far = sig[sig.length - 1] || T.junctions[T.junctions.length - 1];
  const farArm = byDeg(far).sort((a, b) => b.len - a.len)[0];
  if (farArm) push(onRoad(farArm, far, 30, 'far junction'));
  return views;
}
