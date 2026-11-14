"""Count the junction approaches whose exits share a navigation command (Lane.turn, TownConfig.turn_rule).

An approach (a road arm of a junction of 3 or more arms, cars coming in) is ambiguous when two of its exits (other
arms) get the same turn, so a route's command there does not tell which exit it takes.

    uv run python tools/turn_ambiguity.py --style bigjunction --seeds 3000 3020 [--turn_rule heading] [-v]
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from keisim.config import TOWN_STYLES, town_config  # noqa: E402
from keisim.geometry import wrap  # noqa: E402
from keisim.roadnet import Town  # noqa: E402


def ambiguous(T, verbose=False):
    """(ambiguous approaches, approaches) of town T."""
    seen, amb = set(), set()
    for L in T.lanes:
        conns = [T.lanes[c] for c in L.succ if T.lanes[c].kind == "conn"] if L.kind != "conn" else []
        if not conns or len(T.junctions[conns[0].junction].arms) < 3:
            continue
        key = (conns[0].junction, L.road)
        seen.add(key)
        by_turn = defaultdict(set)
        for c in conns:
            by_turn[c.turn].add(T.lanes[c.succ[0]].road)
        if any(len(v) > 1 for v in by_turn.values()) and key not in amb:
            amb.add(key)
            if verbose:
                dh = sorted((round(float(wrap(T.lanes[c.succ[0]].heading[0] - L.heading[-1])), 2), c.turn)
                            for c in conns)
                print(f"  town {T.seed} junction {key[0]} ({len(T.junctions[key[0]].arms)} arms) road {key[1]}: "
                      f"{sorted(set(dh))}")
    return len(amb), len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="bigjunction", choices=list(TOWN_STYLES))
    ap.add_argument("--seeds", type=int, nargs=2, default=(3000, 3020), help="town seeds [a, b)")
    ap.add_argument("--turn_rule", default=None, help="override TownConfig.turn_rule (heading | ordered)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    cfg = town_config(args.style)
    if args.turn_rule:
        cfg.turn_rule = args.turn_rule
    na = nn = 0
    for seed in range(*args.seeds):
        a, n = ambiguous(Town(seed, cfg), args.verbose)
        na, nn = na + a, nn + n
    print(f"{args.style} (turn_rule {cfg.turn_rule}), towns {args.seeds[0]}-{args.seeds[1] - 1}: "
          f"{na} of {nn} approaches ambiguous")


if __name__ == "__main__":
    main()
