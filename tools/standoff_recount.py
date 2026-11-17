"""Re-run the blocked routes of a model's evaluations with evaluate.py's standoff() (round 13), and count them.

Before round 13, evaluate.py called a route "traffic"-blocked when the expert, where the ego stood, would have waited
too, which includes the ego standing in a junction in the way of a car that waits for it (a standoff: the simulator
breaks none). Each blocked route is re-run as it was (same options, one route per run: --only, Sapporo --episode0 E
--routes 1) into runs/eval/standoff/[osm_sapporo/]<the evaluation's file name>/<town>_<ep>.json; the failure logs
(runs/failures/) are rewritten with the new field.
  python tools/standoff_recount.py TAG --ckpt CKPT [--renderer keisim|keiview]   one evaluate.py command per line
      (the routes not done yet, the longest first: run them a few at a time, e.g. xargs -P)
  python tools/standoff_recount.py TAG --count      the causes of the blocked routes, per group of suites
TAG: the evaluations' tag, e.g. keipilot_mc13_hold_red_yh_4_1_3.5_4_0.5_0.5_10."""
import argparse
import glob
import json
import os
import shlex

GROUPS = {   # the town suites of rounds 10-12 and round 12's new suites
    "grid": ["long3c", "long3d", "long3e", "long3f", "test", "test2", "train", "long", "long2"],
    "lanes": ["suburban", "ramps", "ramp_routes"],
    "new": ["crosswalks", "crosswalks_sub", "bigjunction", "crosswalks_bk8", "crosswalks_sub_bk8", "bigjunction_bk8"],
    "lanes2": ["suburban2", "ramps2", "ramp_routes2"],
}


def eval_files(tag, renderer):
    """(path, suite or None for Sapporo) of the tag's evaluation files with that renderer."""
    out = []
    for f in sorted(glob.glob(f"runs/eval/{tag}_*.json")):
        kv = f.endswith("_keiview.json")
        if kv == (renderer == "keiview"):
            out.append((f, None))
    for f in sorted(glob.glob(f"runs/eval/osm_sapporo/{tag}*_pat3.json")):
        if ("_keiview" in f) == (renderer == "keiview"):
            out.append((f, "osm"))
    return out


def out_path(f, r):
    sub = "osm_sapporo/" if "/osm_sapporo/" in f else ""
    return f"runs/eval/standoff/{sub}{os.path.basename(f)[:-5]}/{r['town']}_{r['episode']}.json"


def commands(f, osm, ckpt, routes):
    """[(wall s of the route's first run, command)] for the routes of f not re-run yet."""
    a = json.load(open(f))["args"]
    c = ["uv", "run", "--offline", "scripts/evaluate.py", "--agent", "model", "--ckpt", ckpt]
    if a.get("brake_hold"):
        c.append("--brake_hold")
    if a.get("red_hold"):
        c.append("--red_hold")
    if a.get("yield_hold") is not None:
        c += ["--yield_hold"] + [f"{x:g}" for x in a["yield_hold"]]
    if a.get("dense"):
        c.append("--dense")
    if a.get("renderer") == "keiview":
        c += ["--renderer", "keiview"]
    c += ["--max_steps", str(a["max_steps"]), "--route_length", f"{a['route_length']:g}", "--workers", "1"]
    if osm:
        c += ["--osm_town", a["osm_town"], "--ego_patience", f"{a['ego_patience']:g}"]
    else:
        c += ["--suite", a["suite"]]
    out = []
    for r in routes:
        p = out_path(f, r)
        if os.path.exists(p):
            continue
        sel = ["--episode0", str(r["episode"]), "--routes", "1"] if osm else ["--only", f"{r['town']}:{r['episode']}"]
        out.append((r.get("wall", 0.0), c + sel + ["--out", p]))
    return out


def group_of(name):
    base = name.replace("_dense", "").replace("_keiview", "")
    for g, suites in GROUPS.items():
        if base in suites:
            return g
    return "other"


def count(tag):
    rows = {}
    for rend in ("keisim", "keiview"):
        for f, osm in eval_files(tag, rend):
            routes = [r for r in json.load(open(f))["results"] if r["status"] in ("blocked", "timeout")]
            if not routes:
                continue
            name = "sapporo" if osm else os.path.basename(f)[len(tag) + 1:-5]
            g = "sapporo" if osm else group_of(name)
            new = {}
            for r in routes:
                p = out_path(f, r)
                if os.path.exists(p):
                    for x in json.load(open(p))["results"]:
                        new[(x["town"], x["episode"])] = x
            for r in routes:
                x = new.get((r["town"], r["episode"]))
                key = (rend, g)
                c = rows.setdefault(key, {})
                if x is None:
                    k = "not re-run"
                elif x["status"] != r["status"] or abs(x["time"] - r["time"]) > 0.15:
                    k = "differs"           # the re-run went otherwise (should not happen)
                else:
                    so = x["blocked_by"].get("standoff")
                    k = x["blocked_by"]["cause"] + ("" if so is None else ("+mutual" if so["mutual"] else "+box"))
                c[k] = c.get(k, 0) + 1
    for (rend, g), c in sorted(rows.items()):
        print(f"{rend:8s} {g:8s} " + "  ".join(f"{k} {v}" for k, v in sorted(c.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--renderer", default="keisim", choices=["keisim", "keiview"])
    ap.add_argument("--count", action="store_true")
    args = ap.parse_args()
    if args.count:
        count(args.tag)
        return
    todo = []
    for f, osm in eval_files(args.tag, args.renderer):
        routes = [r for r in json.load(open(f))["results"] if r["status"] in ("blocked", "timeout")]
        if routes:
            todo += commands(f, osm, args.ckpt, routes)
    for _, c in sorted(todo, key=lambda x: -x[0]):
        print(shlex.join(c))


if __name__ == "__main__":
    main()
