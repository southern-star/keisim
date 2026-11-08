#!/bin/bash
# mc8 with the narrowed YieldHold (third version, ea7f586): values
# 4 1 3.5 4 0.5 0.5 10 = the second version's yield condition, plus a hold only where the light head sees no light
# (p >= 0.5) and within 10 m of the last stand.
# Picked-route check (runs/mc_work/yh4, 11-04 10:14): Sapporo KeiSim 35: collisions 6 / 2 / 1 (mc8 / 2nd / narrowed),
# standing 138 / 370 / 216 s; grid 15: collisions+red 0 / 7 / 0, standing 125 / 556 / 172 s.
# The same steps as yieldhold.sh (Sapporo with _pat3, then the 16 grid suites beside KeiView's long3c), tag
# keipilot_mc8_hold_red_yh_4_1_3.5_4_0.5_0.5_10. Start / restart: runs/mc_work/start_yieldhold3.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q "self.after_m is None or stand_m" keipilot/agent.py || { echo "agent.py has no narrowed YieldHold"; exit 1; }
M=runs/keipilot_mc8/last.pt
TAG=keipilot_mc8_hold_red_yh_4_1_3.5_4_0.5_0.5_10
E="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold --yield_hold 4 1 3.5 4 0.5 0.5 10"
P=3                     # --ego_patience of Sapporo's evaluations: suffix _pat3

sap() {   # Sapporo with _pat$P: sap keisim|keiview
  local batches="100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500" rend="" opt="--workers 6"
  [ $1 = keiview ] && { rend=_keiview; opt="--renderer keiview --workers 3"; batches="100 200 300"; }
  for e in $batches; do
    local f=runs/eval/osm_sapporo/${TAG}$([ $e = 100 ] || echo _ep$e)${rend}_pat$P
    [ -f $f.json ] || $E --osm_town osm_sapporo --routes 20 --episode0 $e --ego_patience $P $opt --out $f.json
  done
}
ev() { local name=$1; shift; [ -f runs/eval/${TAG}_$name.json ] || $E "$@"; }
kv() {
  sap keiview
  echo YH3_SAPPORO_KV_DONE
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
}
ks() {
  sap keisim
  echo YH3_SAPPORO_KS_DONE
  for a in long3c long3d long3e long3f; do ev $a --suite $a --workers 6; ev ${a}_dense --suite $a --dense --workers 6; done
  ev test --suite test --workers 6
  ev test2 --suite test2 --workers 6
  ev train --suite train --workers 6
  ev test_dense --suite test --dense --max_steps 6000 --workers 6
  ev long --suite long --workers 6
  ev long_dense --suite long --dense --workers 6
  ev long2 --suite long2 --workers 6
  ev long2_dense --suite long2 --dense --workers 6
}
kv &
ks &
wait
echo YH3_DONE
