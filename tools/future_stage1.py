"""The future head's closed-loop comparison (docs/future_aux.md, phase 3, stage 1): mc14f (with the head), mc14c
(mc14's training with another random stream) and mc14 on the 440 KeiSim routes the head aims at, from runs/eval.

    python3 tools/future_stage1.py [--tags mc14f mc14c mc14] [--out runs/eval/future_aux/stage1.json]

Per suite and pooled: the routes not done and the routes with a collision or a red light; for every pair of models
the routes where only one of the two failed, with a two-sided sign test. The criteria were set before the run:
one secondary measure (the routes not done, pooled) to be better than mc14's at p < 0.05 or better than both
mc14's and mc14c's.
"""
import argparse
import json
import math
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
H = "hold_red_yh_4_1_3.5_4_0.5_0.5_10"
SUITES = ["parked", "crosswalks", "crosswalks_sub", "bigjunction", "crosswalks_bk8", "crosswalks_sub_bk8",
          "bigjunction_bk8"]
SAP = [f"sap{e}" for e in range(100, 1600, 100)]        # Sapporo, 15 x 20 routes (--ego_patience 3)


def path(tag, suite):
    if suite.startswith("sap"):
        e = int(suite[3:])
        return f"runs/eval/osm_sapporo/keipilot_{tag}_{H}{'' if e == 100 else f'_ep{e}'}_pat3.json"
    return f"runs/eval/keipilot_{tag}_{H}_{suite}.json"


def load(tag, suites):
    return {(s, r["town"], r["episode"]): r for s in suites
            for r in json.load(open(os.path.join(ROOT, path(tag, s))))["results"]}


def red(r):
    return any(i.get("type") == "red_light" for i in r.get("infractions", []))


def bad(r):
    return r["status"].startswith("collision") or red(r)


def failed(r):
    return r["status"] != "success"


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def count(rs):
    out = {"routes": len(rs), "not_done": sum(map(failed, rs)), "coll_or_red": sum(map(bad, rs)),
           "DS": sum(r.get("DS", 0.0) for r in rs) / len(rs), "collisions": {}, "red_only": 0, "blocked": {}}
    for r in rs:
        if r["status"].startswith("collision"):
            k = r["status"][len("collision_"):]
            out["collisions"][k] = out["collisions"].get(k, 0) + 1
        elif red(r):
            out["red_only"] += 1
        if r["status"] in ("blocked", "timeout"):
            c = r.get("blocked_by", {}).get("cause", "?")
            out["blocked"][c] = out["blocked"].get(c, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["mc14f", "mc14c", "mc14"])
    ap.add_argument("--out", default="runs/eval/future_aux/stage1.json")
    args = ap.parse_args()
    groups = [(s, [s]) for s in SUITES] + [("sapporo", SAP), ("pooled", SUITES + SAP)]
    res = {"tags": args.tags, "groups": {}}
    for name, suites in groups:
        rows = {m: load(m, suites) for m in args.tags}
        if any(set(rows[m]) != set(rows[args.tags[0]]) for m in args.tags):
            raise SystemExit(f"{name}: the models' routes are not the same")
        g = {"models": {m: count(list(rows[m].values())) for m in args.tags}, "pairs": {}}
        for i, a in enumerate(args.tags):
            for b in args.tags[i + 1:]:
                A, B = rows[a], rows[b]
                fa, fb = sum(failed(A[k]) and not failed(B[k]) for k in A), sum(failed(B[k]) and not failed(A[k]) for k in A)
                ca, cb = sum(bad(A[k]) and not bad(B[k]) for k in A), sum(bad(B[k]) and not bad(A[k]) for k in A)
                g["pairs"][f"{a} vs {b}"] = {"not_done_only": [fa, fb], "not_done_p": sign_p(fa, fb),
                                             "coll_or_red_only": [ca, cb], "coll_or_red_p": sign_p(ca, cb)}
        res["groups"][name] = g
    print("| suite | routes | " + " | ".join(f"{m} not done" for m in args.tags) + " | "
          + " | ".join(f"{m} coll+red" for m in args.tags) + " |")
    print("|---|---:|" + "---:|" * (2 * len(args.tags)))
    for name, g in res["groups"].items():
        ms = g["models"]
        print(f"| {name} | {ms[args.tags[0]]['routes']} | " + " | ".join(str(ms[m]["not_done"]) for m in args.tags)
              + " | " + " | ".join(str(ms[m]["coll_or_red"]) for m in args.tags) + " |")
    for name in ("sapporo", "pooled"):
        g = res["groups"][name]
        print(f"\n{name}:")
        for m in args.tags:
            c = g["models"][m]
            print(f"  {m:6s} DS {c['DS']:.3f}  collisions {c['collisions']}  red only {c['red_only']}  blocked {c['blocked']}")
        for k, p in g["pairs"].items():
            print(f"  {k}: not done only {p['not_done_only'][0]} / {p['not_done_only'][1]} (p {p['not_done_p']:.3g}); "
                  f"coll-or-red only {p['coll_or_red_only'][0]} / {p['coll_or_red_only'][1]} (p {p['coll_or_red_p']:.3g})")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
