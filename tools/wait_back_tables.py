"""Tables of runs/eval/wait_back/ (expert_wb<M>[_delay<S>]_<suite>.json): done, collisions, standoffs, other
blocked / timeouts, mean route time (all routes, and the routes both wait_back 0 and 2 completed) and average speed,
for wait_back 0 and 2 side by side.

    uv run python tools/wait_back_tables.py
"""
import glob
import json
import os
import re

D = os.path.join(os.path.dirname(__file__), "..", "runs", "eval", "wait_back")


def row(path):
    d = json.load(open(path))
    s, r = d["summary"], d["results"]
    st = s["status"]
    coll = sum(v for k, v in st.items() if k.startswith("collision"))
    so = sum(1 for x in r if (x.get("blocked_by") or {}).get("cause") == "standoff")
    other = sum(v for k, v in st.items() if k in ("blocked", "timeout")) - so
    t = sum(x["time"] for x in r) / len(r)
    ok = {(x["town"], x["episode"]): x["time"] for x in r if x["status"] == "success"}
    return f"{st.get('success', 0)}/{len(r)}", coll, so, other, t, s["avg_speed_kmh"], ok


def main():
    runs = {}
    for p in glob.glob(os.path.join(D, "expert_wb*.json")):
        m = re.match(r"expert_wb(\d+(?:\.\d+)?)(?:_delay(\d+(?:\.\d+)?))?_(.+)\.json", os.path.basename(p))
        wb, delay, suite = m.group(1), m.group(2) or "0", m.group(3)
        runs[(delay, suite, wb)] = row(p)
    for delay in sorted({k[0] for k in runs}, key=float):
        print(f"\n### expert_delay {delay} s\n")
        print("| suite | wait_back | done | collisions | standoffs | other blocked / timeouts | mean time (s) | "
              "mean time, routes done with both (s) | avg km/h |")
        print("|---|---|---|---|---|---|---|---|---|")
        for suite in sorted({k[1] for k in runs if k[0] == delay}):
            v0, v2 = runs.get((delay, suite, "0")), runs.get((delay, suite, "2"))
            both = set(v0[6]) & set(v2[6]) if v0 and v2 else set()
            for wb, v in (("0", v0), ("2", v2)):
                if v:
                    pt = f"{sum(v[6][k] for k in both) / len(both):.1f} ({len(both)})" if both else "-"
                    print(f"| {suite} | {wb} | {v[0]} | {v[1]} | {v[2]} | {v[3]} | {v[4]:.1f} | {pt} | {v[5]:.1f} |")


if __name__ == "__main__":
    main()
