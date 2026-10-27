"""Sapporo, many routes per tag (episode0 100, 200, ... 20 routes each): DS, completed, failures by kind, paired sign
tests between the tags, and where each failure happened (last junction reached -> next junction).
Collision kinds as ss_wait.py, from the failure logs: ss_wait = a vehicle collision where the ego stood in the last
10 s and the expert's final yield began at <= 13 km/h (out of the wait position after a 止まれ stop); yield = the
other vehicle collisions in a yield; veh = the remaining vehicle collisions; ped.
Usage (fork dir): sap300.py [--keiview] [--batches 100-1500] TAG[=FAILDIR,FAILDIR] ...
The default failure-log folder is runs/failures/osm_sapporo/<tag>_hold_red[_keiview]."""
import collections, glob, json, math, os, sys

args = sys.argv[1:]
kv = "--keiview" in args
args = [a for a in args if a != "--keiview"]
batches = list(range(100, 1600, 100)) if not kv else [100, 200, 300]
if "--batches" in args:
    i = args.index("--batches")
    lo, hi = (int(x) for x in args[i + 1].split("-"))
    batches = list(range(lo, hi + 1, 100))
    del args[i:i + 2]
rend = "_keiview" if kv else ""
R = "runs/eval/osm_sapporo"


def kind_of(d):
    r = d["result"]
    i = r["infractions"][0]
    tc = i["t"]
    fr = sorted({x["t"]: x for w in d["windows"] for x in w if x["t"] <= tc + 1e-6}.values(), key=lambda x: x["t"])
    last = [x for x in fr if x["t"] >= tc - 10]
    stood = any(x["v"] < 0.1 for x in last)
    ty = None
    for x in reversed(last):
        if x["reason"] == "yield":
            ty = x
        elif ty is not None:
            break
    if i["type"] == "vehicle" and stood and ty is not None and ty["v"] * 3.6 <= 13:
        return "ss_wait"
    if i["type"] == "vehicle":
        return "yield" if ty is not None else "veh"
    return i["type"][:3]


def where(r):
    """the junction nearest to where the route failed (the first infraction's s, else the distance driven), and the
    failure's place relative to it, as ('j33', 'j33 straight, 46 m after')"""
    jp = r["junction_passes"]
    s = r["infractions"][0]["s"] if r["infractions"] else r.get("distance")
    if s is None or not jp:
        return "?", "?"

    def gap(p):
        return 0.0 if p["s_in"] <= s <= p["s_out"] else min(abs(s - p["s_in"]), abs(s - p["s_out"]))
    p = min(jp, key=gap)
    rel = "in" if gap(p) == 0 else (f"{p['s_in'] - s:.0f} m before" if s < p["s_in"] else f"{s - p['s_out']:.0f} m after")
    return f"j{p['junction']}", f"j{p['junction']} {p['turn']}, {rel}"


tags, kinds, rows = [], {}, {}
for spec in args:
    tag, _, dirs = spec.partition("=")
    dirs = dirs.split(",") if dirs else [f"runs/failures/osm_sapporo/{tag}_hold_red{rend}"]
    res = {}
    for e in batches:
        f = f"{R}/{tag}_hold_red{'' if e == 100 else f'_ep{e}'}{rend}.json"
        if os.path.exists(f):
            for r in json.load(open(f))["results"]:
                res[r["episode"]] = r
    logs = {}
    for d0 in dirs:
        for f in glob.glob(f"{d0}/*.json"):
            d = json.load(open(f))
            logs[d["result"]["episode"]] = d
    K = {}
    for ep, r in res.items():
        if r["status"].startswith("collision"):
            K[ep] = kind_of(logs[ep]) if ep in logs else "col?"
        elif r["status"] == "blocked":
            K[ep] = "blk_" + r.get("blocked_by", {}).get("cause", "?")
        elif r["status"] != "success":
            K[ep] = r["status"]
    tags.append(tag)
    kinds[tag], rows[tag] = K, res

common = set.intersection(*(set(rows[t]) for t in tags))
print(f"{'KeiView' if kv else 'KeiSim'}: {len(common)} routes in common (episode0 {batches[0]}..{batches[-1]})")
cols = ["ss_wait", "yield", "veh", "ped", "blk_model", "blk_traffic", "route_deviation"]
print(f"{'tag':18s} {'DS':>6s} {'done':>5s} {'col':>4s} " + " ".join(f"{c:>9s}" for c in cols) + "  other")
for t in tags:
    res = [rows[t][e] for e in common]
    K = {e: k for e, k in kinds[t].items() if e in common}
    c = collections.Counter(K.values())
    col = sum(1 for r in res if r["status"].startswith("collision"))
    other = {k: v for k, v in c.items() if k not in cols}
    print(f"{t:18s} {sum(r['DS'] for r in res) / len(res):6.3f} {sum(r['status'] == 'success' for r in res):5d} {col:4d} "
          + " ".join(f"{c.get(k, 0):9d}" for k in cols) + f"  {other or ''}")


def sign(a, b):
    m = min(a, b)
    return min(1.0, 2 * sum(math.comb(a + b, i) for i in range(m + 1)) / 2 ** (a + b)) if a + b else 1.0


groups = {"any failure": None, "collision": ("ss_wait", "yield", "veh", "ped", "col?"),
          "yield to side traffic (ss_wait + yield)": ("ss_wait", "yield"), "ss_wait": ("ss_wait",)}
print("\npaired sign tests (routes failing only with A / only with B, p):")
for i, a in enumerate(tags):
    for b in tags[i + 1:]:
        parts = []
        for g, ks in groups.items():
            fa = {e for e, k in kinds[a].items() if e in common and (ks is None or k in ks)}
            fb = {e for e, k in kinds[b].items() if e in common and (ks is None or k in ks)}
            na, nb = len(fa - fb), len(fb - fa)
            parts.append(f"{g}: {na}/{nb} p={sign(na, nb):.3f}")
        print(f"  {a} vs {b}: " + "; ".join(parts))

print("\nfailures by the nearest junction (episode:kind [turn, place]):")
place = collections.defaultdict(lambda: collections.defaultdict(list))
for t in tags:
    for e, k in kinds[t].items():
        if e in common:
            j, rel = where(rows[t][e])
            place[j][t].append(f"{e}:{k} [{rel}]")
for j, by in sorted(place.items(), key=lambda kv: -sum(len(v) for v in kv[1].values())):
    print(f"  {j}")
    for t in tags:
        if by.get(t):
            print(f"      {t:16s} " + "; ".join(by[t]))

# round 1's two new failure spots of mc2 (the curve after j33, the sharp left from a 止まれ at j157): routes
# planned through them, and failures there
print("\nspots (routes through it: failed there / routes):")
for j, turn in [(33, "straight"), (157, "left")]:
    cells = []
    for t in tags:
        thru = [e for e in common if any(p["junction"] == j and p["turn"] == turn for p in rows[t][e]["junction_passes"])]
        bad = [e for e in thru if e in kinds[t] and where(rows[t][e])[0] == f"j{j}"]
        cells.append(f"{t} {len(bad)}/{len(thru)} {sorted(bad) if bad else ''}")
    print(f"  j{j} {turn}: " + "; ".join(cells))
