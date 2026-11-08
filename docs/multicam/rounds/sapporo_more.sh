#!/bin/bash
# Round 1 follow-up: Sapporo with 240 more routes (episode0 400..1500) for v0.10 (runs/keipilot_v24) and mc2, so the
# stop-line collisions (60 routes: v0.10 6, mc2 3, sign test p = 0.45) can be told from chance. KeiSim, as round 1.
# Every batch skips what is already there. Start: systemd-run --user --unit mc-sapporo-more ... (see HANDOVER.md).
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
for e in 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500; do
  for M in v24 mc2; do
    f=runs/eval/osm_sapporo/keipilot_${M}_hold_red_ep$e
    [ -f $f.json ] || uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_$M/last.pt --brake_hold --red_hold \
        --osm_town osm_sapporo --routes 20 --episode0 $e --workers 6 --out $f.json
  done
done
echo MC_SAPPORO_MORE_DONE
