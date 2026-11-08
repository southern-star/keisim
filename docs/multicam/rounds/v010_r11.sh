#!/bin/bash
# v0.10 (runs/keipilot_v24/last.pt, the single 100 deg camera, with its released layers BrakeHold + RedHold) on round
# 11's town suites and the curve courses with this code, for the README's v0.10 vs KeiPilot Mc (mc12) tables.
# Tag keipilot_v24_r11; curve tags v24_r11_*.
# Every step skips what is already there. Start / restart: runs/mc_work/start_v010_r11.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
M=runs/keipilot_v24/last.pt
L="--brake_hold --red_hold"
TAG=keipilot_v24_r11_hold_red
E="uv run scripts/evaluate.py --agent model --ckpt $M $L --tag keipilot_v24_r11"

curve() {   # the curve courses, as round11.sh
  local n=v24_r11 C="uv run scripts/curve_test.py --agent model --ckpt $M $L --tp ahead"
  for ep in 0 1 2 3; do $C --weather noon --episode $ep --variants weak weak_lines --tag ${n}_weak_ep$ep || return 1; done
  $C --weather noon --variants weak weak_lines --guardrail_light flat --tag ${n}_weak_flat || return 1
  for ep in 0 1 2 3; do $C --weather noon --episode $ep --variants nolines_guard nolines_guard_nobld lines_guard \
      --tag ${n}_curve_ep$ep || return 1; done
  $C --renderer keiview --variants weak --tag ${n}_weak_keiview || return 1
}
[ -f runs/eval/curve/v24_r11_weak_keiview.json ] || curve || exit 1
echo V10_CURVE_DONE

P=3                     # --ego_patience of Sapporo's evaluations: suffix _pat3 (as the multi-camera rounds)
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
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
  ev ramp_routes_keiview --suite ramp_routes --renderer keiview --workers 3
  ev suburban_keiview --suite suburban --renderer keiview --workers 3
  sap keiview
  echo V10_KV_DONE
}
ks() {
  for a in long3c long3d long3e long3f; do ev $a --suite $a --workers 6; ev ${a}_dense --suite $a --dense --workers 6; done
  ev ramp_routes --suite ramp_routes --workers 6
  ev ramps --suite ramps --workers 6
  ev suburban --suite suburban --workers 6
  ev ramp_routes_dense --suite ramp_routes --dense --workers 6
  ev suburban_dense --suite suburban --dense --workers 6
  sap keisim
  ev test --suite test --workers 6
  ev test2 --suite test2 --workers 6
  ev train --suite train --workers 6
  ev test_dense --suite test --dense --max_steps 6000 --workers 6
  ev long --suite long --workers 6
  ev long_dense --suite long --dense --workers 6
  ev long2 --suite long2 --workers 6
  ev long2_dense --suite long2 --dense --workers 6
  echo V10_KS_DONE
}
kv &
ks &
wait
echo V10_R11_DONE
