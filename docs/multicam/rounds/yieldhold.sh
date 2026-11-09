#!/bin/bash
# mc8 with the YieldHold safety layer (keipilot.agent.YieldHold, evaluate.py --yield_hold, tag suffix _yh; runs/mc_work/
# HANDOVER.md, "安全装置"). No training: the evaluations of round 7 again, with the layer on.
# Second version (144e824): a hold starts only from a yield the model itself began (its target 0.5 m/s below the
# speed, or below 0.5 m/s). The first version (1a36bd1, now tag _yh_4_1_3.5_4_-1) held on the head's readings alone:
# Sapporo collisions 6 -> 2 but needless standing ~10x, grid not run. On 33 KeiSim + 7 KeiView picked routes this one
# completed 39 (mc8's 6 KeiSim collisions avoided; fog route 115 blocked), standing 137 / 1472 / 364 s (mc8 / first /
# second). Compare with mc8 without it, route by route (tools/sap300.py, tools/cmp_suites.py: TAG
# keipilot_mc8_hold_red_yh vs keipilot_mc8).
# 1. Sapporo with _pat3: KeiSim 300 (6 workers) beside KeiView 60 (3 workers)
# 2. then the 16 grid suites (no patience) beside KeiView's long3c and long3c dense, as in the rounds
# Every step skips what is already there. Start / restart: runs/mc_work/start_yieldhold.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q "self.brake is None or target" keipilot/agent.py || { echo "agent.py has no YieldHold with the brake condition"; exit 1; }
M=runs/keipilot_mc8/last.pt
TAG=keipilot_mc8_hold_red_yh
E="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold --yield_hold"
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
  echo YH_SAPPORO_KV_DONE
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
}
ks() {
  sap keisim
  echo YH_SAPPORO_KS_DONE
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
echo YH_DONE
