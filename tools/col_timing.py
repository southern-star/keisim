"""Timing errors before each vehicle collision in failure-log folders (runs/failures/...): seconds of the last 10 s in
which the expert said stop (< 0.5 m/s) and the model go (> 1.5 m/s), with the ego moving (a late yield, or the rest of
an early start) or standing (v < 0.3: an early start), and in which the expert said go (> 3 m/s) and the model stop;
and how long before the collision.
  python col_timing.py DIR [DIR ...]   (fork dir)"""
import glob, json, sys
for d0 in sys.argv[1:]:
    print("##", d0.split("/")[-1])
    for f in sorted(glob.glob(f"{d0}/*.json")):
        d = json.load(open(f)); r = d["result"]
        if r["status"] != "collision_vehicle":
            continue
        tc = r["infractions"][0]["t"]
        fr = sorted({x["t"]: x for w in d["windows"] for x in w if tc - 10 <= x["t"] <= tc + 1e-6}.values(),
                    key=lambda x: x["t"])
        dt = 0.1
        late = [x["t"] for x in fr if x["target_expert"] < 0.5 and x["target_model"] > 1.5 and x["v"] >= 0.3]
        early = [x["t"] for x in fr if x["target_expert"] < 0.5 and x["target_model"] > 1.5 and x["v"] < 0.3]
        slow = [x["t"] for x in fr if x["target_expert"] > 3 and x["target_model"] < 1.0]
        def span(ts):
            return f"{len(ts) * dt:3.1f} s ({tc - ts[0]:4.1f}..{tc - ts[-1]:4.1f} s before)" if ts else "   -"
        print(f"  ep{r['episode']:5d}  model go / expert stop, moving: {span(late):28s} standing: {span(early):28s}"
              f" model slow / expert go: {span(slow)}")
