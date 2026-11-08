"""Sapporo collisions out of the wait position after a 止まれ stop, per route: a vehicle collision where the ego stood
(v < 0.1) in the last 10 s and the expert's final yield began at <= 13 km/h (i.e. at the stop, not while slowing from
cruise for a yield-only junction). Paired sign test between two tags. Usage: ss_wait.py TAG_A=DIR[,DIR] TAG_B=DIR"""
import glob, json, math, sys
def kinds(dirs):
    out = {}
    for d0 in dirs:
        for f in glob.glob(f"{d0}/*.json"):
            d = json.load(open(f)); r = d["result"]
            if not r["status"].startswith("collision"):
                continue
            i = r["infractions"][0]; tc = i["t"]
            fr = sorted({x["t"]: x for w in d["windows"] for x in w if x["t"] <= tc + 1e-6}.values(), key=lambda x: x["t"])
            last = [x for x in fr if x["t"] >= tc - 10]
            stood = any(x["v"] < 0.1 for x in last)
            ty = None
            for x in reversed(last):
                if x["reason"] == "yield": ty = x
                elif ty is not None: break
            if i["type"] == "vehicle" and stood and ty is not None and ty["v"] * 3.6 <= 13:
                out[r["episode"]] = "ss_wait"
            elif i["type"] == "vehicle" and ty is not None:
                out[r["episode"]] = "yield"
            else:
                out[r["episode"]] = "other_" + i["type"][:3]
    return out
(la, da), (lb, db) = [a.split("=", 1) for a in sys.argv[1:3]]
A, B = kinds(da.split(",")), kinds(db.split(","))
for lab, K in [(la, A), (lb, B)]:
    c = {}
    for k in K.values(): c[k] = c.get(k, 0) + 1
    print(lab, c, sorted(e for e, k in K.items() if k == "ss_wait"))
a = sum(1 for e in A if A[e] == "ss_wait" and B.get(e) != "ss_wait")
b = sum(1 for e in B if B[e] == "ss_wait" and A.get(e) != "ss_wait")
both = sum(1 for e in A if A[e] == "ss_wait" and B.get(e) == "ss_wait")
m = min(a, b); p = min(1, 2 * sum(math.comb(a + b, i) for i in range(m + 1)) / 2 ** (a + b))
print(f"ss_wait routes: only {la} {a}, only {lb} {b}, both {both}; sign test p={p:.3f}")
