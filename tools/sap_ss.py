"""Sapporo 60 routes per tag and renderer: DS, completed, collisions, model-blocked, infractions, and stop-line passes
(junction passes whose window had the expert's "stop_sign" reason): full stop (v_min < 0.4 km/h), slow (< 3 km/h),
rolling (>= 3 km/h)."""
import collections, json, os, sys
R = "runs/eval/osm_sapporo"
tags = sys.argv[1:] or ["keipilot_v20tm", "keipilot_v21", "keipilot_v22"]
for rend in ["", "_keiview"]:
    print("KeiView" if rend else "KeiSim")
    for t in tags:
        res = []
        for e in ["", "_ep200", "_ep300"]:
            f = f"{R}/{t}_hold_red{e}{rend}.json"
            if os.path.exists(f):
                res += json.load(open(f))["results"]
        if not res:
            continue
        st = collections.Counter(r["status"] for r in res)
        blk = collections.Counter(r.get("blocked_by", {}).get("cause") for r in res if "blocked_by" in r)
        inf = collections.Counter(i["type"] for r in res for i in r["infractions"])
        ss = [p for r in res for p in r["junction_passes"] if p.get("reached") and "stop_sign" in p.get("expert_reasons", {})]
        n = len(ss)
        full = sum(p["v_min"] < 0.4 / 3.6 for p in ss)
        slow = sum(0.4 / 3.6 <= p["v_min"] < 3 / 3.6 for p in ss)
        roll = n - full - slow
        ds = sum(r["DS"] for r in res) / len(res)
        col = sum(v for k, v in st.items() if k.startswith("collision"))
        print(f"  {t:16s} n={len(res)} DS={ds:.3f} done={st['success']} col={col} blocked(model)={blk.get('model',0)} "
              f"blocked(other)={sum(blk.values())-blk.get('model',0)} other={ {k:v for k,v in st.items() if k!='success' and not k.startswith('collision') and k!='blocked'} } "
              f"inf={dict(inf)} | stop lines {n}: full {full/n:.0%} slow {slow/n:.0%} roll {roll/n:.0%}")
