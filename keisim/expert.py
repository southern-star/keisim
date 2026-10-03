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
from .geometry import world_to_local
from .traffic import corridor_gaps, vehicle_circles

PATH_S = np.arange(1, 11) * 2.0          # label waypoints every 2 m up to 20 m
LABEL_VERSION = 5                        # 3: yellow per traffic law; 4: at yellow, stop whenever the ego still can;
                                         # 5: and always when creeping (< 1 m/s) with the bumper at the line
TL_MARGIN = 2.5                          # stop this far before the stop line (keeps it in view)


def stop_profile(x, b=2.5):
    """Target speed with `x` metres left to the stopping point.
    sqrt(2bx) far away, a slow linear creep close in -> small distance errors near
    the stopping point only cause small speed errors (easier to learn from vision)."""
    if x <= 0.25:
        return 0.0
    return min(math.sqrt(2 * b * x), 0.8 * x + 0.2)


def light_stop(v, d, st, t_rem, exit_blocked):
    """The expert's decision at the next stop line, `d` >= 0 metres ahead of the front bumper, at ego speed `v`:
    the reason it stops there, or None. This is the only place where the ego speed enters the plan.
    Red and yellow: stop unless even a hard stop (6 m/s^2) is impossible. Japanese traffic law lets a car go on at
    yellow only when it is too close to stop safely. Label version 3 read "safely" as a comfortable 3.5 m/s^2, but
    that per-frame rule could flip to "go" halfway through a stop when the ego braked a little late (it then
    reached the line just after red). With the hard-stop criterion the decision does not flip while the ego brakes,
    and "go" is left only when a stop is physically impossible, i.e. the line is under a second away.
    The remaining yellow time `t_rem` is not used: a camera + speed model could not see it.
    Label version 5: the 0.3 m margin (a moving car whose bumper is about at the line) does not apply below 1 m/s,
    so a car creeping up a queue with its bumper at the line stops there at yellow and red instead of following the
    car ahead across it (seen in queues behind cars waiting to turn right inside two-phase junctions)."""
    can_stop = d > v * v / (2 * 6.0) + (0.3 if v >= 1.0 else 0.0)
    if st in (TL_RED, TL_YELLOW):
        stop = can_stop
    else:
        stop = False
    if stop:
        return "red_light"
    if d < 12.0 and can_stop and exit_blocked:
        return "junction_blocked"            # don't block the box: wait at the stop line if the exit is full
    return None


def target_for_speed(v, lab):
    """The expert's target speed had the ego been driving at `v`, from the speed-independent parts of its plan
    (target_nolight, lt_over, lt_d, lt_st, lt_trem, lt_blocked; see Expert.plan). Relabels recorded frames with
    counterfactual ego speeds, so a speed-input model learns how the speed changes the decision."""
    target = float(lab["target_nolight"])
    if lab["lt_over"] and v < 3.0:
        target = 0.0
    d = float(lab["lt_d"])
    if math.isfinite(d) and light_stop(v, d, int(lab["lt_st"]), float(lab["lt_trem"]), bool(lab["lt_blocked"])):
        target = min(target, stop_profile(d - TL_MARGIN, Expert.B_COMF))
    return 0.0 if target < 0.3 else target


class Expert:
    B_COMF = 2.5          # m/s^2 used for stopping profiles
    GAP0_VEH = 4.0        # standstill gap to vehicles (bumper to bumper)
    GAP0_PED = 5.0        # standstill gap to pedestrians
    A_LAT = 2.0

    def __init__(self, env):
        self.env = env

    def path_local(self, route, s, pose):
        pts = route.poly.interp(np.minimum(s + PATH_S, route.poly.length))
        return world_to_local(pts, *pose)

    def plan(self, pose=None, s=None):
        env = self.env
        w = env.world
        route = env.route
        ego = w.ego
        if pose is None:
            pose = ego.pose
        if s is None:
            s = env.s_ego
        v = ego.v
        hl = ego.LENGTH / 2
        path = self.path_local(route, s, pose)
        reason = "cruise"
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

        # --- give way (two-phase signals): a right turn waits at its connector's wait point, inside the
        # junction, while oncoming traffic would reach the crossing within YIELD_GAP s. Independent of the ego
        # speed, so it is part of target_nolight (the counterfactual labels keep it as it is).
        if tr.has_yields:
            s_front = s + hl
            for jn in route.junctions:
                if jn["s_out"] < s_front:
                    continue
                if jn["s_in"] - s_front > 45.0:
                    break
                conn = w.town.lanes[jn["lane"]]
                if conn.yields:
                    d_wait = jn["s_in"] + conn.wait_s - s_front
                    if d_wait >= -0.5 and tr.must_yield(conn, w.t, tr.yield_tables()):
                        v_y = stop_profile(max(d_wait, 0.0), self.B_COMF)
                        if v_y < target:
                            target, reason = v_y, "yield"
                break

        # --- traffic light. The speed-independent inputs of the decision are returned too (lt_*), so recorded
        # frames can be relabelled for other ego speeds (and for later versions of the rule).
        tl = TL_NONE
        target_nolight = target
        lt = {"lt_over": False, "lt_d": math.nan, "lt_st": TL_NONE, "lt_trem": math.nan, "lt_blocked": False}
        s_front = s + hl
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
            blocked = False
            if d < 12.0:
                k = route.lane_index_at(s_stop)
                if k + 2 < len(route.lanes):
                    occ = w.traffic.occupancy()
                    blocked = w.traffic.exit_blocked(route.lanes[k + 1], route.lanes[k + 2], ego.LENGTH + 3.0, occ)
                    if not blocked and w.traffic.cfg.box_rule:      # another approach's vehicle still inside
                        blocked = w.traffic.box_busy(route.lanes[k + 1], w.traffic.box_occupants(occ))
            lt.update(lt_d=d, lt_st=st, lt_trem=t_rem, lt_blocked=blocked)
            why = light_stop(v, d, st, t_rem, blocked)
            if why:
                v_tl = stop_profile(d - TL_MARGIN, self.B_COMF)
                if v_tl < target:
                    target, reason = v_tl, why
            break
        if target < 0.3:
            target = 0.0
        return {"path": path, "target_speed": float(target), "tl_state": int(tl), "reason": reason,
                "gap": g, "target_nolight": float(target_nolight), **lt}

    def act(self, plan=None):
        env = self.env
        plan = plan or self.plan()
        return env.follower(plan["path"], plan["target_speed"], env.world.ego.v, env.cfg.dt * env.cfg.action_repeat)
