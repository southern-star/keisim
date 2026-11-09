#!/bin/bash
# 片側2車線: where the rig model starts on main's suburban / ramps towns (merged into
# camera-rig at 7502227: two lanes each way, lane changes of the ego and the NPCs, at-grade left-turn ramps).
# mc8 with its standard layers (brake, red, the narrowed YieldHold), and v0.10 (keipilot_v24, one front camera) again
# because main's v0.10 suburban result predates the NPC lane changes. Expert (main's runs/eval): suburban 19/20,
# dense 18/20, ramps 19/20, ramp_routes 30/30, dense 30/30; v0.10: suburban 8/20 (before NPC lane changes),
# ramp_routes 30/30 (but +1.5-2.1 m to the left around the ramps). off_lane_ratio = share of the distance off the
# route's lane (wrong lane, swinging wide). Start / restart: runs/mc_work/start_suburban0.sh. Markers in
# logs/mc_suburban0.log: SUB0_KV_DONE, SUB0_KS_DONE, SUB0_DONE.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q "via_ramp" scripts/evaluate.py || { echo "evaluate.py has no --via_ramp (merge main first)"; exit 1; }
TAG=keipilot_mc8_hold_red_yh_4_1_3.5_4_0.5_0.5_10
E="uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_mc8/last.pt --brake_hold --red_hold --yield_hold 4 1 3.5 4 0.5 0.5 10"
V="uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_v24/last.pt --brake_hold --red_hold"
ev() { local name=$1; shift; [ -f runs/eval/${TAG}_$name.json ] || $E "$@"; }
ev24() { local name=$1; shift; [ -f runs/eval/keipilot_v24_hold_red_$name.json ] || $V "$@"; }
kv() {
  ev suburban_keiview --suite suburban --renderer keiview --workers 3
  ev ramp_routes_keiview --suite ramp_routes --renderer keiview --workers 3
  echo SUB0_KV_DONE
}
ks() {
  ev suburban --suite suburban --workers 6
  ev ramp_routes --suite ramp_routes --workers 6
  ev24 suburban --suite suburban --workers 6
  ev24 ramp_routes --suite ramp_routes --workers 6
  ev suburban_dense --suite suburban --dense --workers 6
  ev ramps --suite ramps --workers 6
  ev ramp_routes_dense --suite ramp_routes --dense --workers 6
  echo SUB0_KS_DONE
}
kv &
ks &
wait
echo SUB0_DONE
