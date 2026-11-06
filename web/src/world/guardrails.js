// ガードレール — KeiSim's guardrails (Town.guardrails, RoadSpec.guardrail; keisim/curve_course.py): a white beam
// 0.45–0.80 m above the ground on the road's side of a line of posts (one every 4 m), as keisim/render/meshes.py
// draws them. Towns without guardrails build nothing. Ego mode labels them pole (src/ego.js), as KeiSim does.
import * as THREE from 'three';

const DS = 2.0;          // m per beam panel (keisim GUARDRAIL_DS)

/** Points every ~DS m along a KeiSim polyline (the export simplifies straight runs to their ends). */
function resample(pts) {
  const out = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i], d = Math.hypot(b[0] - a[0], b[1] - a[1]), n = Math.max(1, Math.round(d / DS));
    for (let k = 1; k <= n; k++) out.push([a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);
  }
  return out;
}

export function build(ctx) {
  const T = ctx.town, rails = T.data.guardrails || [];
  if (!rails.length) return;
  const mRail = ctx.mat.toon('#e8e9eb', { paint: 0.05 });
  const mPost = ctx.mat.toon('#cdced2', { paint: 0.05 });
  const root = new THREE.Group(); root.name = 'guardrails';
  const K = ctx.kit(root);
  for (const g of rails) {
    const p = resample(g.pts);
    for (let i = 1; i < p.length; i++) {
      const a = p[i - 1], b = p[i], dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
      if (len < 1e-3) continue;
      // the road lies to the right of the line for a rail on its left (side +1), to the left for side -1
      const ix = -g.side * (-dy / len), iy = -g.side * (dx / len);
      const [X, Z] = T.P([(a[0] + b[0]) / 2 + ix * 0.06, (a[1] + b[1]) / 2 + iy * 0.06]);
      const yaw = Math.atan2(dy, dx);                    // KeiSim heading; three.js: rotation.y = heading (z mirrored)
      const g0 = ctx.L.heightAt(X, Z);
      K.box(len + 0.04, 0.35, 0.06, mRail, [X, g0 + 0.625, Z], [0, yaw, 0]);
      if (i % 2 === 1) {
        const [PX, PZ] = T.P(a);
        K.box(0.1, 0.75, 0.1, mPost, [PX, ctx.L.heightAt(PX, PZ) + 0.375, PZ], [0, yaw, 0]);
      }
    }
  }
  ctx.addStatic(root);
}
