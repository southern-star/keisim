"""Full stops at the 止まれ lines of a real town, per model and renderer (evaluate.py --osm_town results).

The stop lines are the junction passes where the expert itself stood: the pass had the expert's "stop_sign" reason
and a minimum speed under 0.4 km/h in the expert's own run (same episode, junction and s_in). The reason alone is not
enough: a stop line just after a junction puts "stop_sign" into that junction's pass too, and the expert drives
through there at 10-20 km/h. For each model run the minimum speed at those passes (if it got there) is counted as a
full stop (< 0.4 km/h), slow (< 3 km/h) or rolling.

usage: python tools/stop_lines.py [--expert expert_ss] keipilot_v18ss_hold_red keipilot_v24_hold_red ...
       (tags as in runs/eval/osm_sapporo/<tag>[_ep200|_ep300][_keiview].json)
"""
import argparse
import json
import os

FULL, SLOW = 0.4 / 3.6, 3.0 / 3.6


def load(d, tag, renderer=""):
    res = []
    for e in ["", "_ep200", "_ep300"]:
        f = os.path.join(d, f"{tag}{e}{renderer}.json")
        if os.path.exists(f):
            res += json.load(open(f))["results"]
    return res


def key(ep, p):
    return ep, p["junction"], round(p["s_in"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--expert", default="expert_ss", help="the expert's runs on the same routes (KeiSim)")
    ap.add_argument("--dir", default="runs/eval/osm_sapporo")
    args = ap.parse_args()
    lines = {key(r["episode"], p) for r in load(args.dir, args.expert) for p in r["junction_passes"]
             if p.get("reached") and "stop_sign" in p.get("expert_reasons", {}) and p["v_min"] < FULL}
    print(f"{len(lines)} stop lines where the expert stood ({args.expert})")
    print(f"{'':32s} {'renderer':8s} passes  full stop  < 3 km/h  rolling")
    for tag in args.tags:
        for renderer in ["", "_keiview"]:
            res = load(args.dir, tag, renderer)
            if not res:
                continue
            got = {key(r["episode"], p): p["v_min"] for r in res for p in r["junction_passes"] if p.get("reached")}
            v = [got[k] for k in lines if k in got]
            n = max(len(v), 1)
            full = sum(x < FULL for x in v)
            slow = sum(FULL <= x < SLOW for x in v)
            print(f"{tag:32s} {'KeiView' if renderer else 'KeiSim':8s} {len(v):6d} {full / n:10.0%} {slow / n:9.0%} "
                  f"{(len(v) - full - slow) / n:8.0%}")


if __name__ == "__main__":
    main()
