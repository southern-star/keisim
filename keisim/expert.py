"""Privileged rule-based expert.

The expert's plan is intentionally *observable*: the path is the route centre
line ahead, and the target speed depends only on distances to things the camera
can see (lead vehicle, pedestrians, stop line of a red light, road curvature).
That keeps the imitation targets learnable from a single camera frame.
"""
from __future__ import annotations

import math

import numpy as np

from .config import TL_GREEN, TL_NONE, TL_RED, TL_YELLOW
from .bicycles import BIKE_LEN, BIKE_R, NCB
from .geometry import world_to_local, wrap
from .route import LC_CHECK, LC_PUSH, LC_PUSH_TO, change_room, link_map
from .traffic import (BIKE_END, BIKE_FREE, BIKE_IN, BIKE_KEEP, BIKE_LEFT, BIKE_LEFT_WAIT, BIKE_MIN, BIKE_OUT,
                      BIKE_PASS_DV, BIKE_PASS_V, CW_MARGIN, CW_REACH, PASS_SPEED, YIELD_GAP, Traffic, corridor_gaps,
                      vehicle_circles)

PATH_S = np.arange(1, 11) * 2.0          # label waypoints every 2 m up to 20 m
LABEL_VERSION = 7                        # 3: yellow per traffic law; 4: at yellow, stop whenever the ego still can;
                                         # 5: and always when creeping (< 1 m/s) with the bumper at the line;
                                         # 6: 止まれ: 徐行 (CROSSING_V) from the stop line to the wait point;
                                         # 7: past a turn's wait point every crossing ahead counts (not only the
                                         # first); a slow car that cannot stop at the line for a full exit or a busy
                                         # box stops short of the junction's entry
YELLOW_VERSION = 5                       # the last label version that changed the yellow rule
TL_MARGIN = 2.5                          # stop this far before the stop line (keeps it in view)
BOX_V = 3.0                              # m/s: slower than this, a car past the point where it can stop at the line
                                         # for a full exit or a busy box still stops short of the junction's entry
BOX_MARGIN = 1.0                         # m: ... while its front bumper can stop this far before the entry
BOX_IN = 5.0                             # m: the stop line to the junction's entry (TownConfig.stop_line_setback) for
                                         # frames recorded before lt_in
YIELD_TTA_MAX = 8.0                      # yield_tta labels are clipped here (8 s or more: nobody coming)
LATE_COMMIT = 1.0                        # m: past its wait point, a right turn still stops for oncoming traffic until
                                         # its front bumper is this close to the crossing (centre lines), at each
                                         # crossing still ahead
LATE_MARGIN = 1.0                        # s: ... if a moving car with priority arrives sooner than it clears + this
CROSSING_V = 10.0 / 3.6                  # m/s: 徐行 at the wait point of an unsignalised crossing the ego gives way at
CROSSING_B = 1.6                         # m/s^2: ... slowing down towards it (the curve envelope's deceleration)
STOP_SIGN_MARGIN = 1.0                   # m: 止まれ: the front bumper stops this far before the stop line (still in view)
STOP_SIGN_ZONE = 1.5                     # m: ... and the stop counts when the ego stands within this much more of it
STOP_SIGN_V = 0.05                       # m/s: standing (a full stop: 一時停止)
BIKE_GAP0 = 3.5                          # m: bicycles (TrafficConfig.bicycles): the standstill gap behind one
BIKE_SIDE = 2.6                          # m beyond the ego's side: a bicycle riding this close to the left of the
                                         # route's line is "beside the route" (the ego passes it or stays behind it)
BIKE_START = 8.0                         # m into a road: the ego's move over for a bicycle grows from none to all of it
BIKE_END_LEFT = (25.0, 35.0)             # m before the end of a lane the route leaves to the left: no passing a
                                         # bicycle that would be closer to the turn than this when the ego is past it
                                         # (traffic.BIKE_END elsewhere)
BIKE_LEFT_MARGIN = 1.0                   # s: 巻き込み確認: the ego turns left ahead of a bicycle only if it has cleared
                                         # the bicycle's line this long before the bicycle gets there
STOP_SIGN_OVER = 1.5                     # m: a car that rolled this far past the line without stopping stops there;
                                         # further in, the junction's own rules take over


def clear_time(d_cross, v, length):
    """Time for a car whose front bumper is `d_cross` m before a crossing (centre lines), at speed `v`, to have its
    rear past the crossing lane (about half a lane width beyond the centre line), at 2 m/s^2 up to 5 m/s."""
    return Traffic._reach(max(d_cross, 0.0) + 2.3 + length, v, 5.0, 2.0)


def stop_profile(x, b=2.5):
    """Target speed with `x` metres left to the stopping point.
    sqrt(2bx) far away, a slow linear creep close in -> small distance errors near
    the stopping point only cause small speed errors (easier to learn from vision)."""
    if x <= 0.25:
        return 0.0
    return min(math.sqrt(2 * b * x), 0.8 * x + 0.2)


def light_stop(v, d, st, t_rem, exit_blocked, d_in=BOX_IN):
    """The expert's decision at the next stop line, `d` metres ahead of the front bumper (< 0: past it, short of
    the junction's entry `d_in` m beyond the line), at ego speed `v`: (the reason it stops, metres to its stopping
    point for stop_profile), or None. This is the only place where the ego speed enters the plan.
    Red and yellow: stop unless even a hard stop (6 m/s^2) is impossible. Japanese traffic law lets a car go on at
    yellow only when it is too close to stop safely. Label version 3 read "safely" as a comfortable 3.5 m/s^2, but
    that per-frame rule could flip to "go" halfway through a stop when the ego braked a little late (it then
    reached the line just after red). With the hard-stop criterion the decision does not flip while the ego brakes,
    and "go" is left only when a stop is physically impossible, i.e. the line is under a second away.
    The remaining yellow time `t_rem` is not used: a camera + speed model could not see it.
    Label version 5: the 0.3 m margin (a moving car whose bumper is about at the line) does not apply below 1 m/s,
    so a car creeping up a queue with its bumper at the line stops there at yellow and red instead of following the
    car ahead across it (seen in queues behind cars waiting to turn right inside two-phase junctions).
    Label version 7: a car slower than BOX_V that can no longer stop at the line for a full exit or a busy box (it
    crept up to the line or over it, e.g. a model setting off before the box was clear) stops where it is, short of
    the junction's entry, while it still can. Before, the label said "go" as soon as the line was out of reach (from
    0.4 m before it at 4.5 km/h), which taught models that edging over the line commits them: mc10 set off early,
    passed the line at 5 km/h and hit the car still clearing the junction (grid long3c_dense 1018/9401, long3d_dense
    1014/9500 and 9501: three of its nine collisions)."""
    can_stop = d > v * v / (2 * 6.0) + (0.3 if v >= 1.0 else 0.0)
    if st in (TL_RED, TL_YELLOW) and can_stop:
        return "red_light", d - TL_MARGIN
    if d < 12.0 and exit_blocked:
        if can_stop:                         # don't block the box: wait at the stop line if the exit is full
            return "junction_blocked", d - TL_MARGIN
        if v < BOX_V and d + d_in - BOX_MARGIN > v * v / (2 * 6.0) + 0.3:
            return "junction_blocked", 0.0
    return None


def target_for_speed(v, lab):
    """The expert's target speed had the ego been driving at `v`, from the speed-independent parts of its plan
    (target_nolight, lt_over, lt_d, lt_st, lt_trem, lt_blocked, optional lt_in; see Expert.plan). Relabels recorded frames with
    counterfactual ego speeds, so a speed-input model learns how the speed changes the decision.
    止まれ (optional ss_d, ss_go: the pending stop line ahead and the target without it): standing (v < STOP_SIGN_V)
    within STOP_SIGN_MARGIN + STOP_SIGN_ZONE of the line, the stop is made and the ego goes on (ss_go); at any other
    speed the line still binds (target_nolight). A single frame cannot tell "stopped here already" from "about to
    stop", except by the speed: these pairs, from the same images, are what teaches it (models trained only on the
    states their DAgger visited either froze at the line or rolled over it)."""
    target = float(lab["target_nolight"])
    ss_d, ss_go = float(lab.get("ss_d", math.nan)), float(lab.get("ss_go", math.nan))
    if math.isfinite(ss_d) and math.isfinite(ss_go) and v < STOP_SIGN_V and ss_d <= STOP_SIGN_MARGIN + STOP_SIGN_ZONE:
        target = ss_go
    if lab["lt_over"] and v < 3.0:
        target = 0.0
    d = float(lab["lt_d"])
    if math.isfinite(d):
        d_in = float(lab.get("lt_in", math.nan))
        stop = light_stop(v, d, int(lab["lt_st"]), float(lab["lt_trem"]), bool(lab["lt_blocked"]),
                          d_in if math.isfinite(d_in) else BOX_IN)
        if stop:
            target = min(target, stop_profile(stop[1], Expert.B_COMF))
    return 0.0 if target < 0.3 else target


class Expert:
    B_COMF = 2.5          # m/s^2 used for stopping profiles
    GAP0_VEH = 4.0        # standstill gap to vehicles (bumper to bumper)
    GAP0_PED = 5.0        # standstill gap to pedestrians
    A_LAT = 2.0

    def __init__(self, env):
        self.env = env
        self._ss_route, self._ss_done = None, set()    # 止まれ lines (route s) the ego has stopped at on this route
        self._bk_wait = 0.0                            # s waited before a left turn for a bicycle standing beside
        self.last_bikes = None                         # bike_scan of the ego's own last plan (collect.py)

    def path_local(self, route, s, pose, bikes=None):
        """The label waypoints in the frame of `pose`. With bicycles about (bike_scan; computed here when `bikes` is
        not given) the path moves over to the right inside the lane round the ones the ego passes."""
        sw = np.minimum(s + PATH_S, route.poly.length)
        pts = route.poly.interp(sw)
        w = self.env.world
        if bikes is None and getattr(w, "bikes", None) is not None and w.bikes.n:
            bikes = self.bike_scan(route, s, pose, w.ego.v)
        if bikes is not None and bikes["want"].any():
            reach = w.ego.LENGTH / 2 + BIKE_LEN / 2
            off = np.zeros(len(sw))
            for j in np.nonzero(bikes["want"] > 0.0)[0]:
                ahead = bikes["s"][j] - sw - reach          # the ego's front bumper to the bicycle's rear wheel
                behind = -ahead - 2 * reach                 # its rear bumper to the bicycle's front wheel
                f = np.where(ahead > 0.0, (BIKE_IN[0] - ahead) / (BIKE_IN[0] - BIKE_IN[1]),
                             np.where(behind > 0.0, (BIKE_OUT[1] - behind) / (BIKE_OUT[1] - BIKE_OUT[0]), 1.0))
                # ... and only on the bicycle's own road, from BIKE_START into it: none of it in the junction before
                g = np.clip((sw - bikes["s0"][j]) / BIKE_START, 0.0, 1.0)
                off = np.maximum(off, np.minimum(bikes["want"][j] * np.clip(f, 0.0, 1.0), bikes["cap"][j] * g))
            h = route.poly.heading(sw)
            pts = pts + np.stack([np.sin(h), -np.cos(h)], -1) * off[:, None]
        return world_to_local(pts, *pose)

    def bike_scan(self, route, s, pose, v):
        """Bicycles (TrafficConfig.bicycles) riding beside the route, for a car at `pose` (route position `s`, speed
        `v`) -> {"along": riding along the route within BIKE_SIDE of the car's side, on its left; "s": their route
        positions; "d": how far left of the route's line; "room": how far the ego may move over to the right for each
        (as the NPCs: inside its lane, BIKE_KEEP from the lane's edge, nothing on connectors, during lane changes
        and where the pass would end within BIKE_END of the lane's end, BIKE_END_LEFT before a left turn, and
        growing over the first BIKE_START of a road; beside a bicycle the ego stays as far over as it is); "cap",
        "s0": that room without the last rule, and where the bicycle's road begins on the route (path_local);
        "want": how far it does move over (BIKE_FREE beside the bicycle, if there is that much room); "beside": alongside the ego with its rear wheel not yet past the front
        bumper, and clear of its side (nothing to stop for: the ego keeps its line); "visible": seen by the camera
        (EnvConfig.expert_bike_visible; else all); "behind_unseen": not seen, behind the ego's front and faster than
        it (left out of everything above and of bike_rules)}. No state: the path of a
        displaced camera (collect.py) uses it too."""
        env = self.env
        w = env.world
        bk, town, ego = w.bikes, w.town, w.ego
        hl, hw, hb = ego.LENGTH / 2, ego.WIDTH / 2, BIKE_LEN / 2
        n = bk.n
        out = {"along": np.zeros(n, bool), "beside": np.zeros(n, bool), "s": np.zeros(n), "d": np.zeros(n),
               "room": np.zeros(n), "want": np.zeros(n), "cap": np.zeros(n), "s0": np.zeros(n),
               "visible": self.bikes_seen(pose) if env.cfg.expert_bike_visible else np.ones(n, bool),
               "behind_unseen": np.zeros(n, bool)}
        lat = route.poly.project(np.array([pose[0], pose[1]], float), s - 8.0, s + 8.0)[1]
        p0 = route.poly.interp(s)
        for j in np.nonzero(np.hypot(*(bk.xy - p0).T) < 60.0)[0]:
            s_j, d_j = route.poly.project(bk.xy[j], s - 15.0, s + 55.0)
            if not 0.2 < d_j < hw + BIKE_SIDE:
                continue
            if abs(math.hypot(*(bk.xy[j] - route.poly.interp(s_j))) - d_j) > 0.3:
                continue                                    # beyond the ends of that stretch
            if abs(wrap(bk.yaw[j] - float(route.poly.heading(s_j)))) > 0.7:
                continue                                    # crossing the route: an obstacle like any other
            if not out["visible"][j] and float(bk.v[j]) > v + 0.3 and s_j < s + hl:
                out["behind_unseen"][j] = True              # expert_bike_visible: coming up unseen from behind
                continue
            out["along"][j], out["s"][j], out["d"][j] = True, s_j, d_j
            k = route.lane_index_at(s_j)
            lane = town.lanes[route.lanes[k]]
            vb = float(bk.v[j])
            c = cap = 0.0
            s0 = s_j - route.local_s(s_j, k)                # where the bicycle's lane begins on the route
            if lane.kind == "road" and not any(ch["s_a"] - 30.0 < s_j < ch["s_b"] + 10.0 for ch in route.changes):
                # where the bicycle will be when the ego is past it: no passing into the junction
                behind = s - s_j - hl - hb
                s_done = route.local_s(s_j, k)
                if behind < BIKE_OUT[0]:
                    s_done += vb * (BIKE_OUT[0] - behind) / max(max(v, vb + BIKE_PASS_DV) - vb, 1.0)
                nxt = town.lanes[route.lanes[k + 1]] if k + 1 < len(route.lanes) else None
                e0, e1 = BIKE_END_LEFT if nxt is not None and nxt.kind == "conn" and nxt.turn == "left" else BIKE_END
                cap = max(0.0, lane.width / 2 - hw - BIKE_KEEP) * min(max((lane.length - s_done - e0) / (e1 - e0), 0.0), 1.0)
                # the move over begins on the bicycle's road (path_local): less of it beside one that has just
                # entered. A path moved over for a bicycle on the road after a right turn ran through the junction
                # 0.5 m to the right, and the ego waiting there for oncoming cars stood in their way (long3c
                # 1012/9400: the oncoming car and the ego blocked each other for good)
                c = cap * min(max((s_j - s0) / BIKE_START, 0.0), 1.0)
            ds = s_j - s
            if -(hl + hb + BIKE_OUT[0]) < ds < hl + hb + 2.0:
                hold = min(max(-lat, 0.0), 0.8)
                if hold > c:
                    c = cap = hold
                    s0 = -1e9
                out["beside"][j] = ds < hl + hb + 0.3 and d_j - lat - BIKE_R > hw + 0.1
            out["room"][j], out["cap"][j], out["s0"][j] = c, cap, s0
            out["want"][j] = min(max(hw + 0.3 + BIKE_FREE + 0.05 - d_j, 0.0), c)
        return out

    def bikes_seen(self, pose):
        """The riders the rig's first camera on `pose` sees right now (EnvConfig.expert_bike_visible)."""
        w = self.env.world
        c = self.env.cfg.camera
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        cam = (x + c.x * math.cos(yaw) - c.y * math.sin(yaw), y + c.x * math.sin(yaw) + c.y * math.cos(yaw))
        return w.bikes.visible_from(cam, yaw, math.radians(c.fov_deg) / 2, c.max_dist, w.town)

    def bike_rules(self, route, s, v, bikes, pts, cum, real):
        """The speed rules for bicycles -> target speed (inf: none binds). Behind one the ego cannot pass (less
        than BIKE_MIN beside it on the path bike_scan allows) it follows, BIKE_GAP0 behind at a stand. Passing
        with less than BIKE_FREE beside one: at BIKE_PASS_V, or BIKE_PASS_DV faster than the bicycle. Bicycles
        crossing the route are obstacles with BIKE_MIN beside them, and so is where they will be within 2.4 s.
        巻き込み確認: before a left turn (the front bumper within BIKE_LEFT of the lane's end and not past it) the ego
        stops at its stop line (or the lane's end) while a bicycle beside or behind it would reach the turn before
        the ego has cleared it plus BIKE_LEFT_MARGIN; one standing beside goes first (for BIKE_LEFT_WAIT s at most).
        The riders behind are outside a camera looking ahead: a privileged check (the mirrors), unless
        EnvConfig.expert_bike_visible, which leaves the 巻き込み確認 to the riders the camera sees (bikes["visible"])."""
        w = self.env.world
        bk, town, ego = w.bikes, w.town, w.ego
        hl, hw, hb = ego.LENGTH / 2, ego.WIDTH / 2, BIKE_LEN / 2
        target = math.inf
        cxy, cr, own, csp, cyw = bk.circles(predict=True)
        keep = (np.arange(len(own)) < bk.n * NCB) | ~bikes["along"][own]     # no predictions for the ones alongside
        keep &= ~bikes["behind_unseen"][own]             # (expert_bike_visible: nor any circle of these)
        one = np.ones((1, len(pts)), bool)
        org, hd = np.array([[ego.x, ego.y]]), np.array([ego.yaw])
        sweep = 0.0                     # EnvConfig.expert_bike_sweep: the front corner off the path
        if self.env.cfg.expert_bike_sweep:
            sweep = hl * abs(math.sin(float(wrap(ego.yaw - float(route.poly.heading(s))))))
        for free in (False, True):
            m = keep if free else keep & ~bikes["beside"][own]
            if not m.any():
                continue
            mg = (BIKE_FREE if free else BIKE_MIN) - bikes["room"][own[m]] + sweep
            g, vl, _ = corridor_gaps(pts[None], cum[None], one, org, hd, np.array([hl]), np.array([hw]), cxy[m],
                                     cr[m], own[m], csp[m], cyw[m], np.array([-1]), mg[None, :])
            if not np.isfinite(g[0]):
                continue
            if free:
                v_p = max(BIKE_PASS_V, float(vl[0]) + BIKE_PASS_DV)
                target = min(target, math.sqrt(v_p ** 2 + 2 * 1.6 * max(float(g[0]), 0.0)))
            else:
                target = min(target, stop_profile(float(g[0]) - BIKE_GAP0, self.B_COMF))
        # 巻き込み確認
        s_front = s + hl
        stand = False                   # a rider standing beside the ego before its left turn
        for jn in route.junctions:
            if jn["s_in"] - s_front > BIKE_LEFT:
                break
            conn = town.lanes[jn["lane"]]
            if jn["s_in"] < s_front - 0.5 or conn.kind != "conn" or conn.turn != "left":
                continue
            lin = town.lanes[conn.pred[0]]
            d = jn["s_in"] - s_front - (lin.length - lin.stop_s if lin.stop_s is not None else 0.0)
            if d <= -0.5:
                break
            t_clear = clear_time(jn["s_in"] + 1.0 - s_front, min(v, 5.0), ego.LENGTH) + BIKE_LEFT_MARGIN
            for j in np.nonzero(bikes["along"] & bikes["visible"])[0]:
                ds, vb = bikes["s"][j] - s, float(bk.v[j])
                if ds >= hl + hb + 1.0:
                    continue                                # ahead of the ego: the corridor check
                if vb < 0.3:
                    if ds < -(hl + hb):
                        continue                            # standing behind the ego
                    stand = True
                    if self._bk_wait > BIKE_LEFT_WAIT:
                        continue                            # standing there for good: the ego goes first
                    t_b = 0.0
                else:
                    t_b = Traffic._reach(max(jn["s_in"] + 1.0 - bikes["s"][j] - hb, 0.0), vb, float(bk.cruise[j]), 1.0)
                if t_b < t_clear:
                    target = min(target, stop_profile(max(d - 0.6, 0.0), self.B_COMF))
            break
        if real:
            # the wait lasts until the ego is past its stop line (or the rider rides on): were it to end with the
            # rule, the ego would set off for one step every BIKE_LEFT_WAIT s, and stand beside the rider for good
            if not stand:
                self._bk_wait = 0.0
            elif v < 0.3:
                self._bk_wait += self.env.cfg.dt * self.env.cfg.action_repeat
        return target

    def lane_change_gap(self, route, s, v):
        """The next lane change of the route if it has not started and starts within LC_CHECK m of the ego (route
        position `s`), or within the ego's stopping distance: (index in route.changes, [(dx, v)] of the cars in its
        target lane that leave no room for it (route.change_room), dx along that lane from the ego, + ahead).
        (None, []) otherwise. The cars on the connectors into the target lane count too (just out of the junction
        behind), and the NPCs changing into it (Traffic.lane_cars). A camera looking ahead cannot see the cars behind:
        this is a privileged check (mirrors / a rear camera)."""
        w = self.env.world
        town = w.town
        for i, ch in enumerate(route.changes):
            if s >= ch["s_a"] - 0.3:
                continue                                   # under way or done
            if ch["s_a"] - s > max(LC_CHECK, v * v / (2 * self.B_COMF) + 3.0):
                break                                      # (checked from where the ego could stop before it)
            A, B = town.lanes[ch["from"]], town.lanes[ch["to"]]
            s_e = link_map(A, B, route.local_s(s, ch["k"] - 1))
            bad = [(sn - s_e, vn) for sn, vn, ln in w.traffic.lane_cars(B.id)
                   if not change_room(sn - s_e, vn, ln, v, w.ego.LENGTH)]
            return i, bad
        return None, []

    def plan(self, pose=None, s=None):
        env = self.env
        w = env.world
        route = env.route
        ego = w.ego
        real = pose is None and s is None           # the ego itself (not a displaced camera): updates the 止まれ memory
        if pose is None:
            pose = ego.pose
        if s is None:
            s = env.s_ego
        v = ego.v
        hl = ego.LENGTH / 2
        # --- lane change (multi-lane roads, route.py): just before it starts, the target lane needs room beside and
        # behind the ego. Without it the change moves on along its window (the path ahead changes) while the ego
        # lets the car in the way pass (or falls behind it); at the window's end the ego waits for the room there.
        lc_i, lc_bad = self.lane_change_gap(route, s, v) if route.changes else (None, [])
        if lc_bad and real:
            ch = route.changes[lc_i]
            if ch["s_a"] - s < LC_PUSH and ch["a0"] < ch["a_max"]:
                route.retime_change(lc_i, route.local_s(s, ch["k"] - 1) + LC_PUSH_TO)
        bikes = self.bike_scan(route, s, pose, v) if w.bikes is not None and w.bikes.n else None
        if real:
            self.last_bikes = bikes
        path = self.path_local(route, s, pose, bikes)
        reason = "cruise"
        if w.town.speed_varies:          # road types: the route's own limits
            v_lim = route.speed_limit_ahead(s + hl)
        else:
            v_lim = w.town.cfg.speed_limit
        target = v_lim

        # --- curvature envelope
        ss = np.arange(0.0, 45.0, 1.0) + s
        ss = ss[ss < route.poly.length]
        k = np.abs(route.curvature_at(ss))
        vc = np.sqrt(self.A_LAT / np.maximum(k, 1e-4))
        v_curve = float(np.min(np.sqrt(vc ** 2 + 2 * 1.6 * (ss - s))))
        if v_curve < target:
            target, reason = v_curve, "curve"

        # --- obstacles along the route corridor
        look = 42.0
        sp = np.arange(0.0, look, 1.0) + s
        sp = sp[sp < route.poly.length]
        pts = route.poly.interp(sp)
        cum = sp - s
        tr = w.traffic
        cxy, cr, cow = tr.circles()
        csp, cyw = tr.v[cow], tr.yaw[cow]
        cmg = np.full(len(cow), 0.45)
        peds = w.peds
        if peds.n:
            # current + short-horizon predicted positions for crossing pedestrians
            vel = peds.velocity_vectors()
            crossing = peds.mode == peds.CROSS
            pp = [peds.xy]
            for tau in (0.8, 1.6, 2.4):
                pp.append(np.where(crossing[:, None], peds.xy + vel * tau, peds.xy))
            pxy = np.concatenate(pp)
            m = len(pp)
            cxy = np.concatenate([cxy, pxy])
            cr = np.concatenate([cr, np.full(len(pxy), 0.35)])
            cow = np.concatenate([cow, np.tile(-2 - np.arange(peds.n), m)])
            csp = np.concatenate([csp, np.zeros(len(pxy))])
            cyw = np.concatenate([cyw, np.zeros(len(pxy))])
            if w.town.in_road_walkways or peds.cw_on:   # people walking in a 路側帯: passed closely (slowly, see
                # below); people waiting at a zebra's kerb: closely too
                close = peds.walking_in_road() | peds.waiting_at_kerb()
                cmg = np.concatenate([cmg, np.tile(np.where(close, 0.25, 1.0), m)])
            else:
                cmg = np.concatenate([cmg, np.full(len(pxy), 1.0)])
        gap, lead_v, lead_o = corridor_gaps(pts[None], cum[None], np.ones((1, len(pts)), bool),
                                            np.array([[ego.x, ego.y]]), np.array([ego.yaw]), np.array([hl]),
                                            np.array([ego.WIDTH / 2]), cxy, cr, cow, csp, cyw,
                                            np.array([-1]), cmg[None, :])
        g = float(gap[0])
        if np.isfinite(g):
            g0 = self.GAP0_PED if lead_o[0] <= -2 else self.GAP0_VEH
            v_obs = stop_profile(g - g0, self.B_COMF)
            if v_obs < target:
                target, reason = v_obs, ("pedestrian" if lead_o[0] <= -2 else "vehicle")
        if w.town.in_road_walkways and peds.n:
            # pass people walking beside the lane at PASS_SPEED (they are within the usual 1 m margin)
            walk = peds.walking_in_road()
            if walk.any():
                wxy = peds.xy[walk]
                k = len(wxy)
                g_pass, _, _ = corridor_gaps(pts[None], cum[None], np.ones((1, len(pts)), bool),
                                             np.array([[ego.x, ego.y]]), np.array([ego.yaw]), np.array([hl]),
                                             np.array([ego.WIDTH / 2]), wxy, np.full(k, 0.35), -2 - np.arange(k),
                                             np.zeros(k), np.zeros(k), np.array([-1]), 1.0)
                if np.isfinite(g_pass[0]):
                    v_pass = math.sqrt(PASS_SPEED ** 2 + 2 * 1.6 * max(float(g_pass[0]), 0.0))
                    if v_pass < target:
                        target, reason = v_pass, "pedestrian"

        if bikes is not None:
            v_bk = self.bike_rules(route, s, v, bikes, pts, cum, real)
            if v_bk < target:
                target, reason = v_bk, "bicycle"

        if tr.lcp:          # an NPC that has begun to change into the ego's lane ahead of it: follow it from then on
            k_l = route.lane_index_at(s)
            lane_id = route.lanes[k_l]
            if w.town.lanes[lane_id].kind == "road":
                s_l = route.local_s(s, k_l)
                for j, s_b in tr.changing_into(lane_id):
                    if s_b > s_l:
                        g_lc = s_b - s_l - (ego.LENGTH + tr.dims[j, 0]) / 2
                        v_lc = stop_profile(g_lc - self.GAP0_VEH, self.B_COMF)
                        if v_lc < target:
                            target, reason = v_lc, "vehicle"
        if lc_bad:
            ch = route.changes[lc_i]
            for dx, vn in lc_bad:              # fall in behind the car ahead (one behind: it passes, or the ego waits)
                if dx > 0.0:
                    v_let = max(0.0, vn - 1.0)
                    if v_let < target:
                        target, reason = v_let, "lane_change"
            if ch["a0"] >= ch["a_max"] - 1e-6:      # (stops a metre short: closer in, the change counts as begun)
                v_w = stop_profile(ch["s_a"] - s - 1.0, self.B_COMF)
                if v_w < target:
                    target, reason = v_w, "lane_change"

        # --- give way (two-phase signals): a right turn waits at its connector's wait point, inside the
        # junction, while oncoming traffic would reach the crossing within YIELD_GAP s. Independent of the ego
        # speed, so it is part of target_nolight (the counterfactual labels keep it as it is).
        # Past the wait point (a driving model that came in too fast, or the expert just after setting off) it still
        # stops short of the oncoming lane when a moving car with priority would arrive before the ego has cleared
        # the crossing (plus LATE_MARGIN): earlier labels said "go on" there, which taught models to keep rolling
        # into the oncoming lane after an overshoot. This late check uses the ego speed (rare, on-policy frames).
        # At signals it looks at each crossing still ahead (label version 7), at priority crossings at the first.
        # Unsignalised crossings with 3+ arms (Town.from_graph: priority crossings) where the ego gives way: 徐行
        # (CROSSING_V) at the wait point, whether a car is coming or not (道路交通法 36 / 42 条: the minor road's
        # approach to a crossing with poor visibility). The cars with priority come from the sides, mostly outside
        # the camera's 100 deg until the last metres: at the old 30 km/h a yield began 9 m before the wait point
        # (median), with the car in view at that moment only half of the time, and models braked too late.
        # After a 止まれ (label version 6): no faster than CROSSING_V from the stop line on (its stop zone, where
        # the line binds harder until the stop is made) up to the wait point. The envelope alone let the ego speed
        # up to 16-21 km/h over the 5-10 m from its stop to the wait point, too fast where the corner hides the cars
        # with priority (KeiView's walls and hedges): models that saw a car late hit it.
        yield_tta = math.nan          # label for the model's auxiliary head: earliest arrival of a car with priority
        ss_creep = False              # 徐行 after a 止まれ line (diagnostics)
        if tr.has_yields:
            s_front = s + hl

            def tables():
                if not env.cfg.expert_yield_visible:
                    if env.cfg.expert_bike_visible and bikes is not None:
                        return tr.yield_tables(bikes_visible=bikes["visible"])
                    return tr.yield_tables()
                # only the cars the camera (on `pose`: a displaced camera's own view) shows right now
                c = env.cfg.camera
                x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
                cam = (x + c.x * math.cos(yaw) - c.y * math.sin(yaw), y + c.x * math.sin(yaw) + c.y * math.cos(yaw))
                half = math.radians(c.fov_deg) / 2
                return tr.yield_tables(visible=tr.visible_from(cam, yaw, half, c.max_dist, w.town),
                                       bikes_visible=None if w.bikes is None else
                                       w.bikes.visible_from(cam, yaw, half, c.max_dist, w.town))
            for jn in route.junctions:
                if jn["s_out"] < s_front:
                    continue
                if jn["s_in"] - s_front > 45.0:
                    break
                conn = w.town.lanes[jn["lane"]]
                if conn.yields:
                    d_wait = jn["s_in"] + conn.wait_s - s_front
                    d_cross = jn["s_in"] + conn.yields[0][1] - s_front
                    if d_wait >= -0.5:
                        if not jn["signalized"] and len(w.town.junctions[jn["junction"]].arms) >= 3:
                            v_x = math.sqrt(CROSSING_V ** 2 + 2 * CROSSING_B * max(d_wait, 0.0))
                            lin = w.town.lanes[conn.pred[0]]            # the approach: its 止まれ line, if any
                            if lin.stop_sign_s is not None and s_front >= (jn["s_in"] - lin.length + lin.stop_sign_s
                                                                           - STOP_SIGN_MARGIN - STOP_SIGN_ZONE):
                                v_x, ss_creep = min(v_x, CROSSING_V), True
                            if v_x < target:
                                target, reason = v_x, "crossing"
                        tta = tr.min_arrival(conn, w.t, tables())
                        yield_tta = min(tta, YIELD_TTA_MAX)
                        if tta < YIELD_GAP:
                            v_y = stop_profile(max(d_wait, 0.0), self.B_COMF)
                            if v_y < target:
                                target, reason = v_y, "yield"
                    elif not jn["signalized"]:
                        # priority crossings: the first crossing only (past the near lane, a stop for the far one
                        # would leave the car standing across the near one)
                        if d_cross > LATE_COMMIT:
                            tta = tr.min_arrival(conn, w.t, tables(), moving_only=True)
                            yield_tta = min(tta, YIELD_TTA_MAX)
                            if tta < clear_time(d_cross, v, ego.LENGTH) + LATE_MARGIN:
                                target, reason = 0.0, "yield"
                    else:
                        # signalised (label version 7): every crossing still ahead, each against the time the ego
                        # needs to clear it. With the first one only, a right turn that had passed the oncoming lane
                        # went on into the oncoming left turn merging into the same exit: mc10 rolled through the
                        # wait point and the labels said "go" from there (5 of its 9 grid collisions, e.g. long3e
                        # 1013/9601, long3f 1011/9700)
                        on = None
                        for y in conn.yields:
                            d_y = jn["s_in"] + y[1] - s_front
                            if d_y <= LATE_COMMIT:
                                continue
                            on = tables() if on is None else on
                            tta = tr.min_arrival(conn, w.t, on, moving_only=True, yields=(y,))
                            yield_tta = min(tta, YIELD_TTA_MAX) if math.isnan(yield_tta) else min(yield_tta, tta)
                            if tta < clear_time(d_y, v, ego.LENGTH) + LATE_MARGIN:
                                target, reason = 0.0, "yield"
                break

        # --- crosswalks at signals (TrafficConfig.crosswalks; 道路交通法 38 条): the ego gives way to people on a
        # zebra its route crosses, or setting off from its kerb, who would reach the ego's stretch of it within
        # CW_REACH s (Pedestrians.crosswalk_blocked); it stops with its front bumper CW_MARGIN before the zebra.
        # A turn that waits at its connector's wait point (a right turn at a two-phase signal) holds there instead
        # while the zebra of its exit is taken within CW_REACH s plus the drive on to it, so that it does not stand
        # across the oncoming lanes. Past the zebra's start (committed) the corridor check alone stops for someone
        # in the way. Independent of the ego speed: part of target_nolight.
        if peds.cw_on and route.crossings:
            s_front = s + hl
            for s0c, _, ci, u0, u1, kc in route.crossings:
                d = s0c - s_front
                if d > 45.0:
                    break
                if d < -0.5:
                    continue
                stop_at = None
                conn = w.town.lanes[route.lanes[kc - 1]] if kc > 0 else None
                if conn is not None and conn.kind == "conn" and conn.yields and conn.wait_s is not None:
                    d_wait = route.offsets[kc - 1] + conn.wait_s - route.s0[kc - 1] - s_front
                    if d_wait >= -0.5 and peds.crosswalk_blocked(ci, u0, u1, w.t, CW_REACH + (d - d_wait) / 3.0):
                        stop_at = max(d_wait, 0.0)
                if stop_at is None and peds.crosswalk_blocked(ci, u0, u1, w.t):
                    stop_at = d - CW_MARGIN
                if stop_at is not None:
                    v_c = stop_profile(stop_at, self.B_COMF)
                    if v_c < target:
                        target, reason = v_c, "crosswalk"

        # --- 止まれ (Town.from_graph: the minor road's approaches to priority crossings, Lane.stop_sign_s). The ego
        # stops with its front bumper STOP_SIGN_MARGIN before the painted line (the line stays in the camera's view),
        # once: after standing there (or up to STOP_SIGN_OVER past the line, if it rolled over it) the line no longer
        # binds, and the 徐行 and give-way rules above take the ego on to its connector's wait point. The memory of
        # the stops made is per route and only the ego itself updates it (not the plan of a displaced camera). The
        # rule depends on the ego speed only through that memory, so it is part of target_nolight.
        # ss_d / ss_go: the pending line and the target without it, recorded for target_for_speed's counterfactuals
        ss_d, ss_go = math.nan, target
        if route.stop_signs:
            if self._ss_route is not route:
                self._ss_route, self._ss_done = route, set()
            s_front = s + hl
            for s_line, lid in route.stop_signs:
                d = s_line - s_front
                if d < -STOP_SIGN_OVER or s_line in self._ss_done:
                    continue
                if d > 45.0:
                    break
                if v < STOP_SIGN_V and d <= STOP_SIGN_MARGIN + STOP_SIGN_ZONE:
                    if real:
                        self._ss_done.add(s_line)
                    continue
                ss_d = d
                v_s = stop_profile(d - STOP_SIGN_MARGIN, self.B_COMF)
                if v_s < target:
                    target, reason = v_s, "stop_sign"
                break

        # --- the box, just past the stop line (label version 7; before the line light_stop does it): a car slower
        # than BOX_V that has rolled over the line stops short of the junction's entry while the exit is full or (box
        # rule) another approach's car is still inside, if it still can. Uses the ego speed, so the counterfactual
        # labels keep it as it is (rare, on-policy frames: a model that set off too early).
        s_front = s + hl
        if v < BOX_V:
            for s_stop, lid in route.stops:
                d = s_stop - s_front
                if d >= 0.0:
                    break
                lane = w.town.lanes[lid]
                if d + lane.length - lane.stop_s - BOX_MARGIN <= v * v / (2 * 6.0) + 0.3:
                    continue            # in the junction, or too close to its entry to stop short of it
                if self.blocked_at(route, s_stop):
                    target, reason = 0.0, "junction_blocked"
                break

        # --- traffic light. The speed-independent inputs of the decision are returned too (lt_*), so recorded
        # frames can be relabelled for other ego speeds (and for later versions of the rule).
        tl = TL_NONE
        target_nolight = target
        lt = {"lt_over": False, "lt_d": math.nan, "lt_st": TL_NONE, "lt_trem": math.nan, "lt_blocked": False,
              "lt_in": math.nan}
        for s_stop, lid in route.stops:
            d = s_stop - s_front
            if d < -3.0:
                continue            # well past this stop line
            if d > 45.0:
                break
            st = w.town.signal_state_for_lane(lid, w.t)
            if d < 0.0:
                # just past the line (overshoot): never roll into the junction on red
                if st == TL_RED:
                    lt["lt_over"] = True
                    if v < 3.0:
                        target, reason = 0.0, "red_light"
                continue
            tl = st
            t_rem = math.nan
            if st == TL_YELLOW:
                J = w.town.junctions[w.town.lanes[lid].signal[0]]
                t_rem = J.remaining(w.town.lanes[lid].signal[1], w.t)
            blocked = d < 12.0 and self.blocked_at(route, s_stop)
            lane = w.town.lanes[lid]
            d_in = lane.length - lane.stop_s                # the line to the junction's entry
            lt.update(lt_d=d, lt_st=st, lt_trem=t_rem, lt_blocked=blocked, lt_in=d_in)
            stop = light_stop(v, d, st, t_rem, blocked, d_in)
            if stop:
                v_tl = stop_profile(stop[1], self.B_COMF)
                if v_tl < target:
                    target, reason = v_tl, stop[0]
            break
        if target < 0.3:
            target = 0.0
        # lead_owner: what the gap is to (>= 0: NPC vehicle index in world.traffic, <= -2: pedestrian, -1: nothing)
        return {"path": path, "target_speed": float(target), "tl_state": int(tl), "reason": reason,
                "gap": g, "lead_owner": int(lead_o[0]), "target_nolight": float(target_nolight),
                "yield_tta": float(yield_tta), "ss_d": float(ss_d), "ss_go": float(ss_go), "ss_creep": ss_creep, **lt}

    def blocked_at(self, route, s_stop):
        """Don't block the box: True if the exit beyond the junction of the route's stop line at `s_stop` has no
        room for the ego, or (box rule) a vehicle from another approach is still inside that junction."""
        w = self.env.world
        k = route.lane_index_at(s_stop)
        if k + 2 >= len(route.lanes):
            return False
        occ = w.traffic.occupancy()
        if w.traffic.exit_blocked(route.lanes[k + 1], route.lanes[k + 2], w.ego.LENGTH + 3.0, occ):
            return True
        return bool(w.traffic.cfg.box_rule and w.traffic.box_busy(route.lanes[k + 1], w.traffic.box_occupants(occ)))

    def act(self, plan=None):
        env = self.env
        plan = plan or self.plan()
        return env.follower(plan["path"], plan["target_speed"], env.world.ego.v, env.cfg.dt * env.cfg.action_repeat)
