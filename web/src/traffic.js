// Surrounding traffic for the in-browser KeiPilot (index.html?pilot=1, src/pilot.js): KeiSim's NPC vehicles in a
// lighter form. Each car follows a random route through the lanes of src/nav.js, keeps its distance with the IDM
// (to the cars and the ego ahead on its path), stops at red and yellow like keisim/traffic.py, and does not enter a
// junction while a car from another approach is inside it or its exit is full (the towns here have split signals:
// one approach at a time, so turns never cross a green stream). Cars stuck for long where the ego camera cannot see
// them are moved elsewhere. All coordinates are KeiSim's (x east, y north, yaw counter-clockwise).
import { arcBezier, heading, wrap } from './nav.js';

const TL = { RED: 0, YELLOW: 1, GREEN: 2 };
const IDM = { a: 2.0, b: 3.0, T: 1.2, s0: 2.5 };            // keisim TrafficConfig
const LAT_ACC = 2.0;
const STOP_SETBACK = 5.0;                                   // keisim TownConfig.stop_line_setback
const LOOK = 45.0;                                          // m of path searched for a leader
// keisim/traffic.py CAR_COLORS (BGR there) as RGB
const COLORS = [[235, 235, 235], [238, 240, 240], [195, 190, 190], [32, 30, 30], [125, 120, 120], [25, 60, 140],
  [170, 40, 40], [190, 110, 30], [50, 70, 70], [90, 140, 180], [60, 130, 60]];

/** keisim Junction.signal_state */
export function signalState(J, phase, t) {
  if (!J.signalized) return TL.GREEN;
  const per = J.green + J.yellow + J.allred;
  const T = per * J.phases;
  const tau = (((t + J.offset) % T) + T) % T;
  const k = Math.floor(tau / per);
  if (k !== phase) return TL.RED;
  const r = tau - k * per;
  return r < J.green ? TL.GREEN : r < J.green + J.yellow ? TL.YELLOW : TL.RED;
}

/** A car's route: a growing polyline of lanes and junction connectors, chosen at random at every junction. */
class Route {
  constructor(lane, rnd) {
    this.rnd = rnd;
    this.pts = []; this.cum = []; this.segs = [];
    this.last = lane; this.hint = 1;
    this.append(lane.pts);
    this.segs.push({ kind: 'lane', lane, s0: 0, s1: this.end });
  }

  get end() { return this.cum[this.cum.length - 1]; }

  append(points) {
    for (const p of points) {
      if (this.pts.length) {
        const q = this.pts[this.pts.length - 1], d = Math.hypot(p[0] - q[0], p[1] - q[1]);
        if (d < 1e-6) continue;
        this.cum.push(this.end + d);
      } else this.cum.push(0);
      this.pts.push(p);
    }
  }

  extend(until) {
    while (this.end < until) {
      const opts = this.last.next;
      if (!opts.length) return;
      const c = opts[Math.floor(this.rnd() * opts.length)];
      const a = this.end;
      this.append(arcBezier(this.last.pts[this.last.pts.length - 1], this.last.h1, c.lane.pts[0], c.lane.h0, 16));
      this.segs.push({ kind: 'conn', junction: this.last.to, from: this.last, s0: a, s1: this.end });
      const b = this.end;
      this.append(c.lane.pts);
      this.segs.push({ kind: 'lane', lane: c.lane, s0: b, s1: this.end });
      this.last = c.lane;
    }
  }

  /** Drop what lies more than `keep` m behind arc length s; returns the shift to subtract from s. */
  trim(s, keep = 30) {
    let i = 1;
    while (i < this.cum.length - 1 && this.cum[i] < s - keep) i++;
    if (i < 50) return 0;
    const base = this.cum[i - 1];
    this.pts = this.pts.slice(i - 1);
    this.cum = this.cum.slice(i - 1).map((c) => c - base);
    this.segs = this.segs.filter((g) => g.s1 > base).map((g) => ({ ...g, s0: g.s0 - base, s1: g.s1 - base }));
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
    const i = this.locate(s);
    return heading(this.pts[i - 1], this.pts[i]);
  }

  segAt(s) {
    for (let k = 0; k < this.segs.length; k++) if (s < this.segs[k].s1 || k === this.segs.length - 1) return k;
    return this.segs.length - 1;
  }
}

/**
 * opts: {count, seed}. Returns {step(dt, t, ego), drawList(cx, cy, r), reset(ego), contacts, n}.
 * ego: {x, y, yaw, v} (centre), 4.5 x 1.85 m.
 */
export function createTraffic(data, lanes, { count = 24, seed = 1 } = {}) {
  let st = seed >>> 0 || 1;
  const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  const junctions = new Map(data.junctions.map((j) => [j.id, j]));
  // the signal phase of every lane that ends at a signalised junction: the head facing it
  const phaseOf = new Map();
  for (const L of lanes) {
    const J = junctions.get(L.to);
    if (!J || !J.signalized) continue;
    let best = null, bd = 0.7;
    for (const h of data.signals) {
      if (h.junction !== J.id) continue;
      const d = Math.abs(wrap(h.yaw - (L.h1 + Math.PI)));
      if (d < bd) { bd = d; best = h; }
    }
    if (best) phaseOf.set(L.id, best.phase);
  }
  const roadLanes = lanes.filter((L) => L.len > 30);
  const totalLen = roadLanes.reduce((a, L) => a + L.len, 0);
  const cars = [];
  let contacts = 0, lastContact = -1e9;

  function sampleDims() {
    const r = rnd();
    if (r < 0.62) return [4.0 + 0.8 * rnd(), 1.68 + 0.17 * rnd(), 1.4 + 0.2 * rnd(), 0];
    if (r < 0.78) return [3.35 + 0.1 * rnd(), 1.48, 1.6 + 0.2 * rnd(), 0];          // kei car
    if (r < 0.92) return [4.6 + 0.4 * rnd(), 1.8 + 0.15 * rnd(), 1.75 + 0.25 * rnd(), 0];
    return [6.0 + 1.6 * rnd(), 2.0 + 0.25 * rnd(), 2.6 + 0.4 * rnd(), 1];             // small truck
  }

  /** A free spot on a random lane, away from the ego and the other cars, never right before a stop line. */
  function place(car, ego, avoidEgo) {
    for (let tries = 0; tries < 60; tries++) {
      let x = rnd() * totalLen, L = roadLanes[0];
      for (const l of roadLanes) { if (x < l.len) { L = l; break; } x -= l.len; }
      const hi = phaseOf.has(L.id) ? L.len - STOP_SETBACK - 20 : L.len - 4;
      if (hi < 6) continue;
      const s = 4 + rnd() * (hi - 4);
      const route = new Route(L, rnd);
      const p = route.pointAt(s);
      if (ego && Math.hypot(p[0] - ego.x, p[1] - ego.y) < avoidEgo) continue;
      if (cars.some((c) => c !== car && c.xy && Math.hypot(c.xy[0] - p[0], c.xy[1] - p[1]) < 12)) continue;
      route.extend(s + 80);
      Object.assign(car, { route, s, seg: route.segAt(s), xy: p, yaw: route.headingAt(s), stuck: 0 });
      return true;
    }
    return false;
  }

  function spawn(ego) {
    cars.length = 0;
    for (let i = 0; i < count; i++) {
      const [L, W, H, kind] = sampleDims();
      const vf = 0.8 + 0.25 * rnd();
      const c = COLORS[Math.floor(rnd() * COLORS.length)].map((x) => Math.max(0, Math.min(255, x * (0.9 + 0.18 * rnd()))));
      const car = { id: i, L, W, H, kind, rgb: c, vf, v: 0, brake: false };
      if (place(car, ego, 30)) {
        car.v = (0.3 + 0.5 * rnd()) * 10 * vf;
        cars.push(car);
      }
    }
  }

  /** Lane occupancy and junction boxes, from the cars' current segments. */
  function occupancy(ego) {
    const onLane = new Map(), box = new Map();
    for (const c of cars) {
      const g = c.route.segs[c.seg];
      if (g.kind === 'lane') {
        if (!onLane.has(g.lane.id)) onLane.set(g.lane.id, []);
        onLane.get(g.lane.id).push({ s: c.s - g.s0, v: c.v, L: c.L });
      } else {
        if (!box.has(g.junction)) box.set(g.junction, new Set());
        box.get(g.junction).add(g.from.id);
      }
    }
    if (ego) {
      for (const J of junctions.values()) {
        if (J.signalized && Math.hypot(ego.x - J.pos[0], ego.y - J.pos[1]) < J.radius) {
          if (!box.has(J.id)) box.set(J.id, new Set());
          box.get(J.id).add(-1);                         // the ego (approach unknown): blocks everyone
        }
      }
    }
    return { onLane, box };
  }

  /** Bumper-to-bumper gap and speed of the nearest car (or the ego) on the path ahead. */
  function leader(c, ego) {
    let gap = Infinity, vl = 0;
    const others = cars.filter((o) => o !== c && Math.abs(o.xy[0] - c.xy[0]) < LOOK + 8 && Math.abs(o.xy[1] - c.xy[1]) < LOOK + 8)
      .map((o) => ({ xy: o.xy, yaw: o.yaw, v: o.v, L: o.L }));
    if (ego) others.push({ xy: [ego.x, ego.y], yaw: ego.yaw, v: ego.v, L: 4.5 });
    if (!others.length) return { gap, vl };
    for (let a = 1.0; a <= LOOK; a += 1.5) {
      const p = c.route.pointAt(c.s + a);
      for (const o of others) {
        if (Math.abs(o.xy[0] - p[0]) > 2.6 || Math.abs(o.xy[1] - p[1]) > 2.6) continue;
        if (Math.hypot(o.xy[0] - p[0], o.xy[1] - p[1]) < 1.9) {
          const g = a - c.L / 2 - o.L / 2;
          if (g < gap) {
            gap = g;
            vl = o.v * Math.cos(wrap(o.yaw - c.route.headingAt(c.s + a)));
          }
        }
      }
      if (gap < Infinity && a > gap + 8) break;
    }
    return { gap, vl };
  }

  function step(dt, t, ego) {
    const occ = occupancy(ego);
    for (const c of cars) {
      const k = c.seg, g = c.route.segs[k];
      const v0 = 10 * c.vf;
      // curve speed envelope over the next 30 m (keisim: sqrt(vc^2 + 4 d))
      let vmax = v0;
      for (let a = 2; a <= 30; a += 4) {
        const dh = Math.abs(wrap(c.route.headingAt(c.s + a + 2) - c.route.headingAt(c.s + a - 2)));
        if (dh > 1e-3) vmax = Math.min(vmax, Math.sqrt(LAT_ACC / (dh / 4) + 4 * a));
      }
      let { gap, vl } = leader(c, ego);
      // stop line: red / yellow (if it can stop), or the box rule / a full exit at green
      if (g.kind === 'lane' && phaseOf.has(g.lane.id)) {
        const d = g.s0 + g.lane.len - STOP_SETBACK - (c.s + c.L / 2);
        if (d > -0.5 && d < LOOK) {
          const J = junctions.get(g.lane.to);
          const sig = signalState(J, phaseOf.get(g.lane.id), t);
          let stop = sig === TL.RED || (sig === TL.YELLOW && d > c.v * c.v / 7 + 1);
          if (!stop && d < 12) {
            const inBox = occ.box.get(J.id);
            if (inBox && [...inBox].some((a) => a !== g.lane.id)) stop = true;
            const exit = c.route.segs[k + 2];
            if (!stop && exit && exit.kind === 'lane') {
              const q = occ.onLane.get(exit.lane.id) || [];
              if (q.some((o) => o.s - o.L / 2 < c.L + 3 && o.v < 2)) stop = true;
            }
          }
          if (stop && d - 0.6 < gap) { gap = d - 0.6; vl = 0; }
        }
      }
      let acc = IDM.a * (1 - (c.v / Math.max(vmax, 0.5)) ** 4);
      if (gap < Infinity) {
        const ss = IDM.s0 + Math.max(0, c.v * IDM.T + c.v * (c.v - vl) / (2 * Math.sqrt(IDM.a * IDM.b)));
        acc = Math.min(acc, IDM.a * (1 - (c.v / Math.max(vmax, 0.5)) ** 4 - (ss / Math.max(gap, 0.05)) ** 2));
      }
      acc = Math.max(-8, Math.min(IDM.a, acc));
      c.brake = acc < -0.4 || c.v < 0.2;
      c.v = Math.max(0, c.v + acc * dt);
      c.s += c.v * dt;
      c.stuck = c.v < 0.1 ? c.stuck + dt : 0;
    }
    for (const c of cars) {
      c.route.extend(c.s + 80);
      c.s -= c.route.trim(c.s);
      c.seg = c.route.segAt(c.s);
      c.xy = c.route.pointAt(c.s);
      c.yaw = c.route.headingAt(c.s);
      // stuck for long where the ego camera cannot see it (behind, beside or far): move it elsewhere
      if (c.stuck > 40 && ego) {
        const dx = c.xy[0] - ego.x, dy = c.xy[1] - ego.y, dist = Math.hypot(dx, dy);
        const off = Math.abs(wrap(Math.atan2(dy, dx) - ego.yaw));
        if (dist > 110 || (dist > 8 && off > 1.05)) place(c, ego, 50);
      }
    }
    if (ego) {                                              // contacts with the ego (bodies overlapping)
      for (const c of cars) {
        if (Math.hypot(c.xy[0] - ego.x, c.xy[1] - ego.y) > 6) continue;
        if (overlap(c, ego) && t - lastContact > 2) { contacts++; lastContact = t; }
      }
    }
  }

  /** Oriented-box overlap of a car and the ego (separating axes). */
  function overlap(c, e) {
    const boxes = [[c.xy[0], c.xy[1], c.yaw, c.L / 2, c.W / 2], [e.x, e.y, e.yaw, 2.25, 0.925]];
    const corners = boxes.map(([x, y, yaw, hl, hw]) => {
      const u = [Math.cos(yaw), Math.sin(yaw)], n = [-u[1], u[0]];
      return [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => [x + u[0] * hl * a + n[0] * hw * b, y + u[1] * hl * a + n[1] * hw * b]);
    });
    for (const [x, y, yaw] of boxes) {
      for (const ax of [[Math.cos(yaw), Math.sin(yaw)], [-Math.sin(yaw), Math.cos(yaw)]]) {
        const pr = corners.map((cs) => cs.map((p) => p[0] * ax[0] + p[1] * ax[1]));
        if (Math.max(...pr[0]) < Math.min(...pr[1]) || Math.max(...pr[1]) < Math.min(...pr[0])) return false;
      }
    }
    return true;
  }

  /** actors.js format: [id, x, y, yaw, L, W, H, r, g, b, kind, brake] for the cars within r of (cx, cy). */
  function drawList(cx, cy, r = 140) {
    const out = [];
    for (const c of cars) {
      if (Math.abs(c.xy[0] - cx) > r || Math.abs(c.xy[1] - cy) > r) continue;
      out.push([c.id, c.xy[0], c.xy[1], c.yaw, c.L, c.W, c.H, c.rgb[0], c.rgb[1], c.rgb[2], c.kind, c.brake ? 1 : 0]);
    }
    return out;
  }

  function reset(ego) {
    spawn(ego);
    contacts = 0;
  }

  return { step, drawList, reset, get contacts() { return contacts; }, get n() { return cars.length; }, cars };
}
