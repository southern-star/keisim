#!/usr/bin/env python3
"""Tables of evaluate.py --osm_town runs: DS / RC / infractions per run, and what happened at the junctions, split by
kind (signalised / unsignalised priority crossing / bend), from the per-junction records the runs keep.

  python scripts/osm_report.py runs/eval/osm_sapporo/*.json

For each run: how many junctions of each kind the ego reached, at how many the expert wanted a stop, at how many
the ego stood, needless stops (the ego stood >= 3 s while the expert would have driven on), missed stops (the
expert wanted a stop, the ego never went below 1 m/s), and the collisions / red lights / blocks located at the
junctions (an infraction within [s_in - 15 m, s_out + 5 m] of a junction; "between" otherwise). The failure
windows (runs/failures/<town>/<tag>/) give the expert's reason in the 12 s before each collision.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

KINDS = ("signalised", "priority", "bend", "between")


def junction_of(passes, s, before=15.0, after=5.0):
    hits = [p for p in passes if p["s_in"] - before <= s <= p["s_out"] + after]
    return min(hits, key=lambda p: abs(s - p["s_in"])) if hits else None


def reason_before(fail, inf, span=12.0):
    """The expert's dominant stop reason in the window before the infraction (from the failure log), or None."""
    if fail is None:
        return None
    for win in fail["windows"]:
        if not win or not (win[0]["t"] <= inf["t"] <= win[-1]["t"] + 0.5):
            continue
        rows = [r for r in win if inf["t"] - span <= r["t"] <= inf["t"]]
        stops = Counter(r["reason"] for r in rows if r["target_expert"] < 0.3)
        if stops:
            return stops.most_common(1)[0][0]
        return Counter(r["reason"] for r in rows).most_common(1)[0][0] if rows else None
    return None


def report(path):
    d = json.load(open(path))
    s, res, a = d["summary"], d["results"], d["args"]
    tag = os.path.splitext(os.path.basename(path))[0]
    fail_dir = os.path.join("runs", "failures", a["osm_town"], tag)
    print(f"== {tag}  ({a['agent']}, renderer {a['renderer']}, {s['routes']} routes, {s['km']:.1f} km)")
    print(f"DS {s['DS']:.3f}  RC {s['RC']:.3f}  success {s['success_rate']:.2f}  status {s['status']}  "
          f"blocked_by {s.get('blocked_by', {})}  infractions {s['infraction_counts']}  "
          f"avg {s['avg_speed_kmh']:.1f} km/h")
    kv = [r["keiview_ms"] for r in res if r.get("keiview_ms") and r["keiview_ms"]["mean"]]
    if kv:
        import numpy as np
        print(f"KeiView render: {np.mean([k['mean'] for k in kv]):.1f} ms/frame mean, p90 "
              f"{np.mean([k['p90'] for k in kv]):.1f} ms, town load {kv[0]['town']:.0f} ms")
    jt = {k: Counter() for k in KINDS}
    inf_at = {k: Counter() for k in KINDS}
    blocked_at = Counter()
    reasons = Counter()
    for r in res:
        passes = r.get("junction_passes", [])
        fail = None
        fp = os.path.join(fail_dir, f"{r['town']}_{r['episode']}.json")
        if os.path.exists(fp):
            fail = json.load(open(fp))
        for p in passes:
            if not p["reached"]:
                continue
            c = jt[p["kind"]]
            c["reached"] += 1
            c["expert_stop"] += p["expert_stop"]
            c["ego_stood"] += p["stood_s"] > 0
            c["needless_stop"] += p["needless_s"] >= 3.0
            c["missed_stop"] += p["expert_stop"] and p["v_min"] > 1.0
            c["yield_turns"] += p["yields"]
        for inf in r["infractions"]:
            p = junction_of(passes, inf["s"])
            kind = p["kind"] if p else "between"
            inf_at[kind][inf["type"]] += 1
            why = reason_before(fail, inf)
            reasons[(kind, inf["type"], why)] += 1
        if r["status"] != "success":
            s_end = r["RC"] * r["route_length"]
            p = junction_of(passes, s_end, before=25.0, after=10.0)
            kind = p["kind"] if p else "between"
            cause = r.get("blocked_by", {}).get("cause", r["status"])
            blocked_at[(kind, r["status"], cause)] += 1
    print(f"{'kind':<11}{'reached':>8}{'exp.stop':>9}{'stood':>7}{'needless':>9}{'missed':>8}{'yield':>7}   infractions")
    for k in KINDS:
        c = jt[k]
        print(f"{k:<11}{c['reached']:>8}{c['expert_stop']:>9}{c['ego_stood']:>7}{c['needless_stop']:>9}"
              f"{c['missed_stop']:>8}{c['yield_turns']:>7}   {dict(inf_at[k])}")
    if reasons:
        print("infractions by (junction kind, type, expert's reason in the 12 s before):")
        for (k, t, why), n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {n:3d}  {k:<11} {t:<11} expert: {why}")
    if blocked_at:
        print("routes not completed, by (junction kind where the ego ended, status, cause):")
        for (k, st, cause), n in blocked_at.items():
            print(f"  {n:3d}  {k:<11} {st:<9} {cause}")
    print()


def main():
    paths = sys.argv[1:] or sorted(glob.glob("runs/eval/osm_sapporo/*.json"))
    for p in paths:
        report(p)


if __name__ == "__main__":
    main()
