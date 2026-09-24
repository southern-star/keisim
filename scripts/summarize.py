#!/usr/bin/env python3
"""Print a markdown table from evaluation json files (runs/eval/*.json)."""
from __future__ import annotations

import glob
import json
import os
import sys


def main():
    files = sys.argv[1:] or sorted(glob.glob("runs/eval/*.json"))
    rows = []
    for f in files:
        d = json.load(open(f))
        s = d["summary"]
        name = os.path.basename(f)[:-5]
        inf = s["infractions_per_km"]
        rows.append((name, s["routes"], s["DS"], s["RC"], s["IS"], s["success_rate"], s["km"], s["avg_speed_kmh"],
                     inf.get("vehicle", 0), inf.get("pedestrian", 0), inf.get("static", 0), inf.get("red_light", 0),
                     inf.get("off_road", 0), s["status"]))
    print("| run | routes | DS | RC | IS | success | km | avg km/h | veh/km | ped/km | static/km | red/km | offroad/km | status |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5] * 100:.0f}% | {r[6]:.1f} | {r[7]:.1f} | "
              f"{r[8]:.2f} | {r[9]:.2f} | {r[10]:.2f} | {r[11]:.2f} | {r[12]:.2f} | {r[13]} |")


if __name__ == "__main__":
    main()
