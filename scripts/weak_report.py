#!/usr/bin/env python3
"""Table of the weak-point course (keisim/curve_course.py weak_town) by weak case, from scripts/curve_test.py
results: every case as the driver sees it (turn, lane radius, where the guardrail is), each run (file x direction) a
column. A cell is the offset from the lane centre with the largest size over the piece (m; + = towards the curve's
inside, on a straight towards the left) and a mark: o under 0.4 m, ^ 0.4-0.75 m, x 0.75 m or more (the car's body
leaves its lane) or a collision there. A case met twice in a run (the S-curves) shows the worse.

    python scripts/weak_report.py runs/eval/curve/model_keipilot_v10_keisim_noon_tpahead_w1.json [more.json ...]
    python scripts/weak_report.py --variant weak_lines runs/eval/curve/....json
"""
from __future__ import annotations

import argparse
import json
import os

OK, WARN = 0.4, 0.75          # m

TURN = {"left": "左", "right": "右"}
RAILS = {"none": "なし (対照)", "car_left": "車の左", "car_right": "車の右", "both": "両側",
         "car_left_low": "車の左・低い", "car_right_low": "車の右・低い"}


def case_of(c):
    """(sort key, label) of a piece, as the driver sees it."""
    rails = c["rails"]
    if c["kind"] == "straight":
        return (0, rails), f"直線・ガードレール{RAILS[rails]}"
    inner = (c["turn"] == "left") == rails.startswith("car_left")
    where = "" if rails in ("none", "both") else ("・内側" if inner else "・外側")
    label = f"{TURN[c['turn']]}カーブ R{c['radius']:.0f}{' (180°)' if c['id'].endswith('180') else ''}・{RAILS[rails]}{where}"
    return (1, c["radius"], c["turn"], rails), label


def cell(c):
    """(badness, text) of a piece in a run."""
    if not c.get("reached"):
        return -1.0, "—"
    worst = c["max_in"] if abs(c["max_in"]) >= abs(c["min_in"]) else c["min_in"]
    bad = 9.0 if c.get("collision") else abs(worst)
    mark = "x" if bad >= WARN else "^" if bad >= OK else "o"
    return bad, f"{mark} {worst:+.2f}" + (" 接触" if c.get("collision") else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--variant", default="weak")
    args = ap.parse_args()
    cols, cases = [], {}
    for f in args.files:
        d = json.load(open(f))
        name = os.path.basename(f).replace(".json", "")
        for r in d["results"]:
            if r["variant"] != args.variant or "pieces" not in r:
                continue
            col = f"{name} {'rev' if r['reverse'] else 'fwd'}"
            cols.append(col)
            for c in r["pieces"]:
                key, label = case_of(c)
                cells = cases.setdefault(key, [label, {}])[1]
                new = cell(c)
                if col not in cells or new[0] > cells[col][0]:
                    cells[col] = new
    print("| 場面 | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for key in sorted(cases, key=lambda k: (k[0], str(k[1:]))):
        label, cells = cases[key]
        print(f"| {label} | " + " | ".join(cells[c][1] if c in cells else "" for c in cols) + " |")
    print(f"\no: {OK} m 未満、^: {OK}〜{WARN} m、x: {WARN} m 以上 (車体が車線を出る) かその場面で接触。"
          "値は車線の中心からのずれのいちばん大きいもの (m、＋＝カーブの内側、直線は左)")


if __name__ == "__main__":
    main()
