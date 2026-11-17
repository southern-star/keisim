// Check of the pilot overlay's geometry (src/overlay.js) and of the rollout that gives the band its length
// (src/pilot.js rollout), in node: no browser, GPU or driving model. Prints one line per check and fails (exit 1) if
// one does not hold.
//   node tools/overlay_check.mjs
// Needs three in node_modules (npm install): src/pilot.js imports it.
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const imp = (p) => import(pathToFileURL(path.join(root, p)).href);
const { densify, locate, splitAt, bandEdges, hull, sweptHulls } = await imp('src/overlay.js');
const { rollout } = await imp('src/pilot.js');

const HALF = 1.85 / 2, LENGTH = 4.5, WIDTH = 1.85;
let bad = 0;
function check(name, ok, detail) {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}: ${detail}`);
  if (!ok) bad++;
}
// the model's path: 10 points 2 m apart ahead of the car centre (the origin, heading +x); r: a left curve of that radius
const straight = Array.from({ length: 10 }, (_, k) => [2 * (k + 1), 0]);
const arc = (r) => Array.from({ length: 10 }, (_, k) => { const a = 2 * (k + 1) / r; return [r * Math.sin(a), r * (1 - Math.cos(a))]; });
const withOrigin = (p) => [[0, 0], ...p];
const area = (poly) => poly.reduce((a, p, i) => { const q = poly[(i + 1) % poly.length]; return a + (p[0] * q[1] - q[0] * p[1]) / 2; }, 0);
const segDist = (p, a, b) => {
  const dx = b[0] - a[0], dy = b[1] - a[1], t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
};

// 1. a straight path: the curve stays on it, a point at most every 0.25 m, the edges 0.925 m to the left and right
{
  const line = densify(withOrigin(straight));
  const off = Math.max(...line.P.map((p) => Math.abs(p[1])));
  const gap = Math.max(...line.s.slice(1).map((s, i) => s - line.s[i]));
  const { L, R } = bandEdges(line.P, HALF);
  const edge = Math.max(...L.map((p) => Math.abs(p[1] - HALF)), ...R.map((p) => Math.abs(p[1] + HALF)));
  check('straight path', off < 1e-9 && gap <= 0.25 + 1e-9 && Math.abs(line.s[line.s.length - 1] - 20) < 1e-9 && edge < 1e-9,
    `${line.P.length} points, off the line ${off.toExponential(1)} m, largest gap ${gap.toFixed(3)} m, length ${line.s[line.s.length - 1].toFixed(3)} m, edges off ±0.925 by ${edge.toExponential(1)} m`);
}

// 2. curves: the smoothed path goes through every one of the model's points and stays on the circle through them
//    (the straight pieces between the points cut inside it), the edges keep their distance, no fold on the inside.
//    The first and the last piece have no point beyond them to curve by (a mirrored one stands in): they are
//    between the circle and the straight piece.
for (const r of [15, 6]) {
  const pts = withOrigin(arc(r)), line = densify(pts);
  const rad = (p) => Math.hypot(p[0], p[1] - r);
  const piece = line.s.map((s) => Math.min(9, Math.floor(s / line.s[line.s.length - 1] * 10 - 1e-9)));
  const mid = (k) => k >= 1 && k <= 8;
  const err = Math.max(...line.P.filter((p, i) => mid(piece[i])).map((p) => Math.abs(rad(p) - r)));
  const ends = Math.max(...line.P.filter((p, i) => !mid(piece[i])).map((p) => Math.abs(rad(p) - r)));
  const chord = r - Math.sqrt(r * r - 1);                  // how far the middle of a 2 m chord is inside the circle
  const through = Math.max(...pts.map((q) => Math.min(...line.P.map((p) => Math.hypot(p[0] - q[0], p[1] - q[1])))));
  const { L, R } = bandEdges(line.P, HALF);
  const edge = Math.max(...L.map((p, i) => (mid(piece[i]) ? Math.abs(rad(p) - (r - HALF)) : 0)), ...R.map((p, i) => (mid(piece[i]) ? Math.abs(rad(p) - (r + HALF)) : 0)));
  let fold = 0;
  for (let i = 0; i + 1 < L.length; i++) {
    const tx = line.P[i + 1][0] - line.P[i][0], ty = line.P[i + 1][1] - line.P[i][1];
    for (const e of [L, R]) if ((e[i + 1][0] - e[i][0]) * tx + (e[i + 1][1] - e[i][1]) * ty <= 0) fold++;
  }
  check(`curve of ${r} m`, err < chord / 10 && ends < chord && through < 1e-9 && edge < chord / 5 && fold === 0,
    `off the circle ${(err * 1000).toFixed(1)} mm (first and last piece ${(ends * 1000).toFixed(1)} mm; straight pieces: ${(chord * 1000).toFixed(1)} mm), ` +
    `edges off by ${(edge * 1000).toFixed(1)} mm, misses a point by ${through.toExponential(1)} m, folds ${fold}`);
}

// 3. where the car is on the path, and a point put there
{
  const line = densify(withOrigin(straight));
  const at = locate(line, 3.3, 0.4), n = line.P.length;
  const k = splitAt(line, 5.1), again = splitAt(line, 5.1);
  check('locate, splitAt', Math.abs(at - 3.3) < 1e-9 && Math.abs(line.P[k][0] - 5.1) < 1e-9 && line.P.length === n + 1 && again === k &&
    splitAt(line, -1) === 0 && splitAt(line, 99) === line.P.length - 1,
  `car at (3.3, 0.4) is ${at.toFixed(3)} m along; the point at 5.1 m is index ${k} of ${line.P.length}`);
}

// 4. the rollout: 3 s at the target speed, nothing from a stand, a stop in between when the target is 0
{
  const car = (v) => ({ x: 0, y: 0, yaw: 0, v, steer: 0 });
  const go = rollout(car(8), straight, 8), stand = rollout(car(0), straight, 0), stop = rollout(car(8), straight, 0);
  const side = Math.max(...go.poses.map((p) => Math.abs(p[1])));
  check('rollout, straight', Math.abs(go.dist - 24) < 0.48 && stand.dist === 0 && stop.dist > 0 && stop.dist < 24 && side < 1e-6 && go.poses.length === 61,
    `8 m/s kept: ${go.dist.toFixed(2)} m in 3 s; standing: ${stand.dist} m; 8 m/s braking to 0: ${stop.dist.toFixed(2)} m`);
}

// 5. the ground the body passes over: as wide as the car on a straight path; in a curve it leaves the band on the
//    outside (the front swings out) — what the band alone does not show
{
  const car = { x: 0, y: 0, yaw: 0, v: 4, steer: 0 };
  const hs = sweptHulls(rollout(car, straight, 4).poses, LENGTH, WIDTH);
  const ys = hs.flat().map((p) => p[1]);
  const width = Math.max(...ys) - Math.min(...ys);
  const ccw = hs.every((h) => area(h) > 0);
  check('swept, straight', Math.abs(width - WIDTH) < 0.05 && ccw && hs.length === 60, `${hs.length} pieces, ${width.toFixed(3)} m wide, all counter-clockwise: ${ccw}`);
  for (const r of [10, 6]) {
    const roll = rollout({ ...car, steer: Math.atan(2.7 / r) }, arc(r), 4);     // already steering along the curve
    const hulls = sweptHulls(roll.poses, LENGTH, WIDTH), c = [0, r];
    const out = Math.max(...hulls.flat().map((p) => Math.hypot(p[0] - c[0], p[1] - c[1]))) - (r + HALF);
    const inn = (r - HALF) - Math.min(...hulls.map((h) => Math.min(...h.map((p, i) => segDist(c, p, h[(i + 1) % h.length])))));
    check(`swept, curve of ${r} m`, out > 0.05 && hulls.every((h) => area(h) > 0),
      `${roll.dist.toFixed(1)} m in 3 s; outside the band by ${out.toFixed(2)} m, inside it by ${inn.toFixed(2)} m`);
  }
  const still = sweptHulls(rollout({ ...car, v: 0 }, straight, 0).poses, LENGTH, WIDTH);
  check('swept, standing', still.length === 1 && Math.abs(area(still[0]) - LENGTH * WIDTH) < 1e-9, `${still.length} piece of ${area(still[0]).toFixed(3)} m² (the body: ${(LENGTH * WIDTH).toFixed(3)})`);
  check('hull', area(hull([[0, 0], [2, 0], [2, 1], [0, 1], [1, 0.5], [1, 0]])) === 2, 'a 2 x 1 box with points inside and on its edge: 2 m²');
}

// 6. the cost of a frame's geometry (the drawing itself is the browser's: window.__pilot.state().overlay.ms)
{
  const car = { x: 0.8, y: 0.05, yaw: 0.02, v: 6, steer: 0.1 }, pts = arc(8), t0 = performance.now(), n = 200;
  for (let i = 0; i < n; i++) {
    const roll = rollout(car, pts, 6), line = densify(withOrigin(pts));
    splitAt(line, locate(line, car.x, car.y) + roll.dist);
    bandEdges(line.P, HALF);
    sweptHulls(roll.poses, LENGTH, WIDTH);
  }
  console.log(`     geometry of one frame (rollout, band, swept pieces): ${((performance.now() - t0) / n).toFixed(3)} ms`);
}
console.log(bad ? `${bad} check(s) failed` : 'all checks hold');
process.exitCode = bad ? 1 : 0;
