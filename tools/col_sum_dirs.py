"""Sapporo failure logs from explicit directories: one line per failed route, with the collision's context
(stood in the last 10 s = out of a 止まれ stop; when the expert's final yield began). Usage: col_sum_dirs.py LABEL=DIR..."""
import glob, json, sys
for arg in sys.argv[1:]:
    label, d0 = arg.split("=", 1)
    for f in sorted(glob.glob(f"{d0}/*.json"), key=lambda p: int(p.rsplit("_", 1)[1][:-5])):
        d = json.load(open(f))
        r = d["result"]
        if not r["status"].startswith("collision"):
            print(f"{label:6s} ep{r['episode']:5d} {r['status']} {r.get('blocked_by', {}).get('cause', '')}")
            continue
        tc = r["infractions"][0]["t"]
        fr = [x for w in d["windows"] for x in w if x["t"] <= tc + 1e-6]
        fr = sorted({x["t"]: x for x in fr}.values(), key=lambda x: x["t"])
        last = [x for x in fr if x["t"] >= tc - 10]
        stood = [x["t"] for x in last if x["v"] < 0.1]
        ty = None
        for x in reversed(last):
            if x["reason"] == "yield":
                ty = x
            elif ty is not None:
                break
        seq = []
        for x in last:
            if x["t"] >= tc - 4 and (not seq or seq[-1] != x["reason"]):
                seq.append(x["reason"])
        pre = min(last, key=lambda x: abs(x["t"] - (tc - 0.5)))
        ys = f"yield {tc - ty['t']:.1f}s before at {ty['v']*3.6:.0f} km/h" if ty else "no yield"
        print(f"{label:6s} ep{r['episode']:5d} {r['infractions'][0]['type'][:3]} v_imp={last[-1]['v']*3.6:4.1f} | {ys} | "
              f"stood {'%.1fs before' % (tc - stood[-1]) if stood else 'no'} | -0.5s m {pre['target_model']*3.6:.0f} e {pre['target_expert']*3.6:.0f} | {'>'.join(seq)}")
