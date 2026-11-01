"""v24 vs rig models on the grid suites (KeiSim): DS, completed, collisions (vehicle / pedestrian), red lights,
blocked by the model, other failures; then paired sign tests on the routes that fail with only one of two tags.
--keiview: KeiView's long3c and long3c dense instead."""
import collections, json, math, os, sys
R = "runs/eval"
args = sys.argv[1:]
kv = "--keiview" in args
tags = [a for a in args if a != "--keiview"] or ["keipilot_v24", "keipilot_mc2"]
suites = ["long3c", "long3c_dense", "long3d", "long3d_dense", "long3e", "long3e_dense", "long3f", "long3f_dense",
          "test", "test2", "train", "test_dense", "long", "long_dense", "long2", "long2_dense"]
if kv:
    suites = ["long3c_keiview", "long3c_dense_keiview"]
tot = {t: collections.Counter() for t in tags}
failed = {t: {} for t in tags}      # (suite, town, episode) -> the route failed
hit = {t: {} for t in tags}         # ... -> a collision or a red light on the route
print(f"{'suite':14s}" + "".join(f"{t[9:]:>44s}" for t in tags))
for s in suites:
    cells = []
    for t in tags:
        f = f"{R}/{t if '_hold_red' in t else t + '_hold_red'}_{s}.json"     # keipilot_mc8_hold_red_yh: more layers
        if not os.path.exists(f):
            cells.append(f"{'-':>44s}")
            continue
        res = json.load(open(f))["results"]
        st = collections.Counter(r["status"] for r in res)
        inf = collections.Counter(i["type"] for r in res for i in r["infractions"])
        blk = collections.Counter(r.get("blocked_by", {}).get("cause") for r in res if r["status"] in ("blocked", "timeout"))
        ds = sum(r["DS"] for r in res) / len(res)
        c = tot[t]
        c["n"] += len(res); c["ds"] += sum(r["DS"] for r in res); c["done"] += st["success"]
        c["veh"] += inf["vehicle"]; c["ped"] += inf["pedestrian"]; c["red"] += inf["red_light"]; c["model"] += blk["model"]
        c["km"] += sum(r["distance"] for r in res) / 1000
        failed[t].update({(s, r.get("town"), r["episode"]): r["status"] != "success" for r in res})
        hit[t].update({(s, r.get("town"), r["episode"]): any(i["type"] in ("vehicle", "pedestrian", "static", "red_light")
                                                             for i in r["infractions"]) for r in res})
        cells.append(f"{ds:6.3f} {st['success']:3d}/{len(res):<3d} veh {inf['vehicle']} ped {inf['pedestrian']} "
                     f"red {inf['red_light']} blk {blk['model']}".rjust(44))
    print(f"{s:14s}" + "".join(cells))
for t in tags:
    c = tot[t]
    if c["n"]:
        print(f"{t}: {c['n']} routes {c['km']:.0f} km DS {c['ds'] / c['n']:.3f} done {c['done']} veh {c['veh']} "
              f"ped {c['ped']} red {c['red']} model-blocked {c['model']}")


def sign(a, b):
    m = min(a, b)
    return min(1.0, 2 * sum(math.comb(a + b, i) for i in range(m + 1)) / 2 ** (a + b)) if a + b else 1.0


print("paired sign tests on the routes both have (failing only with A / only with B, p):")
for i, a in enumerate(tags):
    for b in tags[i + 1:]:
        common = failed[a].keys() & failed[b].keys()
        parts = []
        for name, f in [("any failure", failed), ("collision or red light", hit)]:
            na = sum(f[a][k] and not f[b][k] for k in common)
            nb = sum(f[b][k] and not f[a][k] for k in common)
            parts.append(f"{name}: {na}/{nb} p={sign(na, nb):.3f}")
        print(f"  {a} vs {b} ({len(common)} routes): " + "; ".join(parts))
