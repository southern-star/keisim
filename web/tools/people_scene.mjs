// A street with pedestrians for the checks and the bench of the detailed pedestrians (tools/people_check.mjs,
// tools/people_bench.mjs): KeiSim's rows [id, x, y, yaw, height, shirt rgb, pants rgb, skin rgb, phase] along the
// longest road of a town that has a pavement, and a camera on that road. Colours and heights as KeiSim draws them
// (keisim/traffic.py Pedestrians; the children that run out: keisim/parked.py).
import { rng } from './golden.mjs';

const SHIRTS = [[40, 60, 200], [200, 40, 40], [40, 160, 40], [230, 230, 230], [30, 30, 30], [230, 140, 0], [150, 50, 150], [220, 200, 60], [120, 120, 120], [80, 150, 200]];
const PANTS = [[25, 30, 40], [30, 60, 90], [30, 30, 30], [100, 110, 120], [110, 50, 60], [150, 150, 150]];
const SKIN = [[220, 180, 150], [205, 160, 120], [170, 125, 90], [130, 90, 60]];
const KID_SHIRTS = [[230, 60, 40], [30, 120, 230], [240, 200, 40], [200, 60, 200], [90, 200, 60], [240, 240, 240]];

/** Run in the page: the longest straight road with a pavement, in KeiSim's coordinates. */
export const findRoad = () => {
  const r = window.__town.roads.filter((q) => q.sidewalk >= 2 && q.center.length === 2).sort((a, b) => b.length - a.length)[0];
  return { a: r.center[0], b: r.center[1], length: r.length, hw: r.hw, sidewalk: r.sidewalk };
};

/**
 * { cam, ped, far }: near people 4 .. 42 m ahead of the camera (on both pavements and crossing the road), far ones
 * 50 m and more ahead. Every fifth of the near ones is a child. t moves the phases on (the people walk on the spot).
 */
export function street(road, { near = 20, far = 4, t = 0, seed = 9 } = {}) {
  const r = rng(seed), ux = (road.b[0] - road.a[0]) / road.length, uy = (road.b[1] - road.a[1]) / road.length, nx = -uy, ny = ux;
  const yaw = Math.atan2(uy, ux), x0 = road.a[0] + ux * 4, y0 = road.a[1] + uy * 4, off = road.hw + road.sidewalk / 2;
  const row = (id, f, l, kid) => {
    const h = kid ? 1.0 + 0.35 * r() : 1.5 + 0.35 * r(), k = 0.85 + 0.25 * r();
    const shirt = kid ? KID_SHIRTS[Math.floor(r() * 6)] : SHIRTS[Math.floor(r() * 10)].map((v) => Math.round(v * k));
    const pants = kid ? PANTS[0] : PANTS[Math.floor(r() * 6)], skin = kid ? SKIN[1] : SKIN[Math.floor(r() * 4)];
    return [id, x0 + ux * f + nx * l, y0 + uy * f + ny * l, yaw + 6.283 * r(), +h.toFixed(2), ...shirt, ...pants, ...skin, +(6.283 * r() + 4.2 * 1.3 * t).toFixed(3)];
  };
  const ped = [], out = [];
  for (let i = 0; i < near; i++) ped.push(row(i, 4 + 38 * (i + 0.5) / near, [off, -off, 0.8 * road.hw * (2 * r() - 1)][i % 3] + 0.6 * (r() - 0.5), i % 5 === 4));
  for (let i = 0; i < far; i++) out.push(row(1000 + i, 50 + 6 * i, i % 2 ? off : -off, false));
  return { cam: { pos: [x0, y0, 1.4], fwd: [ux, uy, -0.05], vfov: 61.6 }, ped, far: out, x0, y0, ux, uy };
}
