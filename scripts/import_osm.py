#!/usr/bin/env python3
"""Import a real town from OpenStreetMap (keisim/realworld.py), drive the expert through it and export it for KeiView.

    uv run scripts/import_osm.py --bbox 43.088 141.360 43.097 141.372 --name osm_sapporo --routes 4
    cd web && node tools/check.mjs --tdir .towns --town osm_sapporo

--bbox is south west north east (degrees). The Overpass answer is cached under --cache (data/osm/<name>.json) so
later runs are offline. The JSON goes to --out/town_<name>.json (default web/.towns, which the viewer reads with
?tdir=.towns&town=<name>).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from keisim.config import EnvConfig  # noqa: E402
from keisim.realworld import graph_report, osm_town  # noqa: E402


def drive(town, routes, route_length, seed0=100, max_steps=12000):
    from keisim.env import KeiEnv
    cfg = EnvConfig(town=town.cfg, render_rgb=False, render_seg=False)
    cfg.traffic.box_rule = True
    env = KeiEnv(cfg)
    results = []
    for ep in range(routes):
        t0 = time.time()
        env.reset(town, episode_seed=seed0 + ep, route_length=route_length, render=False)
        done, n = False, 0
        while not done and n < max_steps:
            _, _, done, info = env.step(env.expert_action())
            n += 1
        r = env.route
        print(f"route {ep}: {info['status']:14s} DS={info['DS']:.3f} RC={info['RC']:.3f} {info['route_length']:.0f} m "
              f"{info['time']:.0f} s, junctions {len(r.junctions)} (signalised {sum(j['signalized'] for j in r.junctions)}) "
              f"infractions {[(i['type'], i['s']) for i in info['infractions']]} [{time.time() - t0:.0f} s]", flush=True)
        results.append(info)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("S", "W", "N", "E"),
                    default=[43.088, 141.360, 43.097, 141.372], help="Sapporo Higashi-ku's residential grid by default")
    ap.add_argument("--name", default="osm_sapporo")
    ap.add_argument("--cache", default=None, help="Overpass JSON cache (default data/osm/<name>.json)")
    ap.add_argument("--out", default="web/.towns")
    ap.add_argument("--drop", nargs="*", default=["primary"], help="OSM highway classes to leave out")
    ap.add_argument("--seed", type=int, default=0, help="signal timings, trees, building colours")
    ap.add_argument("--routes", type=int, default=0, help="drive the expert through this many routes first")
    ap.add_argument("--route_length", type=float, default=600.0)
    ap.add_argument("--no-buildings", action="store_true")
    ap.add_argument("--oneway", choices=["auto", "on", "off"], default="auto",
                    help="keep one-way roads (auto: on, except the towns built before the importer had them, "
                         "keisim.realworld.TWO_WAY_TOWNS)")
    args = ap.parse_args()

    t0 = time.time()
    town = osm_town(tuple(args.bbox), cache=args.cache or f"data/osm/{args.name}.json", name=args.name, seed=args.seed,
                    drop_kinds=tuple(args.drop), buildings=not args.no_buildings,
                    oneway={"auto": None, "on": True, "off": False}[args.oneway])
    print(town.summary(), f"[{time.time() - t0:.1f} s]")
    print(graph_report(town))
    if args.routes:
        res = drive(town, args.routes, args.route_length)
        print(f"expert: {sum(r['status'] == 'success' for r in res)}/{len(res)} routes completed, "
              f"mean DS {sum(r['DS'] for r in res) / len(res):.3f}")
    from export_town import export_town
    data = export_town(town)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"town_{args.name}.json")
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"{path}  {os.path.getsize(path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
