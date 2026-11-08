#!/bin/bash
# Multi-camera KeiPilot, round 8 (runs/mc_work/HANDOVER.md, "8 回目"): the 片側2車線 towns of upstream PR #23
# (suburban main roads with two lanes each way, lane changes, at-grade left-turn ramps), merged into camera-rig
# (7502227). mc8 there (11-05, narrowed YieldHold): suburban 13/20, ramps 6/20 (all 6 route deviations drift into a
# ramp's deceleration / acceleration lane), ramp_routes 27/30 (merges into a car beside / behind: out of view of the
# forward cameras), right turns from the wrong lane or against two-lane oncoming traffic. Nothing of it trained yet.
# Forward cameras as before (wide_narrow); a rear camera is the open question for the round after.
# 1. DAgger with mc8 driving, towns 0-399 (the suites use 2000-2014): KeiView in ramps towns, then KeiSim in ramps
#    towns and dense in suburban towns (the suburban suites' towns have no ramps). 40 % of the episodes in the ramps
#    towns take a ramp (collect.py --via_ramp: about half of these towns have one). Focus as the grid sets, with the
#    expert's lane-change waits and half the other frames kept (lane keeping is most of what is new)
# 2. mc9 = mc8 trained on rounds 1-8, 4 x 200k samples, round 8's sets x4 / x4 / x3 (about a fifth of the samples).
#    For the RAM (load_shards holds it all, 21 GB at most): old sets halved in turn until it fits
# 3. evaluation with mc8's standard layers (--brake_hold --red_hold --yield_hold 4 1 3.5 4 0.5 0.5 10): the 片側2車線
#    suites first, then Sapporo with _pat3, the 16 grid suites beside KeiView's long3c
# Every step skips what is already there. Start / restart: runs/mc_work/start_round8.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q "via_ramp" scripts/collect.py || { echo "collect.py has no --via_ramp"; exit 1; }
R="--rig wide_narrow"
G="--mode dagger --ckpt runs/keipilot_mc8/last.pt --freeze_takeover 4 --freeze_reasons curve cruise crossing"
F="--jam_fixes 1 --focus_yield 0.5 --focus_reasons yield junction_blocked red_light lane_change pedestrian crossing stop_sign"
RAMPS="--town_style ramps --via_ramp 0.4"

# ---- 1. data
[ -f data/mc_sub_kv_dagger_mc8/.done ] || {
  uv run scripts/collect.py $R $G $F $RAMPS --renderer keiview --vehicle_spacing 24 45 --ego_cross_rate 0.3 \
      --episodes_per_town 3 --out data/mc_sub_kv_dagger_mc8 --frames 36000 --workers 3 --seed 4801 || exit 1
  touch data/mc_sub_kv_dagger_mc8/.done; }
echo MC_COLLECT_KV_R8_DONE
[ -f data/mc_sub_dagger_mc8/.done ] || {
  uv run scripts/collect.py $R $G $F $RAMPS --out data/mc_sub_dagger_mc8 --frames 40000 --workers 6 --seed 4802 || exit 1
  touch data/mc_sub_dagger_mc8/.done; }
[ -f data/mc_sub_d_dagger_mc8/.done ] || {
  uv run scripts/collect.py $R $G $F --town_style suburban --vehicle_spacing 20 28 --ego_cross_rate 0.3 \
      --out data/mc_sub_d_dagger_mc8 --frames 30000 --workers 6 --seed 4803 || exit 1
  touch data/mc_sub_d_dagger_mc8/.done; }
echo MC_COLLECT_R8_DONE

# ---- 2. mc9
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
# mc8's sets and weights (round7.sh after its RAM step)
D="data/mc_expert_half_half data/mc_ped_half data/mc_tp_expert_half_half data/mc_tp2_expert_half data/mc_v_expert_half"
D="$D data/mc_osm_expert_half data/mc_osm_y_expert"
D="$D data/mc_ped_dagger_mc2 data/mc_vped_dagger_mc2 data/mc_fl_dagger_mc2_half data/mc_fl_tp_dagger_mc2_half data/mc_tp3_dagger_mc2_half"
D="$D data/mc_ped_dagger_mc3 data/mc_vped_dagger_mc3 data/mc_tp3_dagger_mc3 data/mc_osm_dagger_mc3_half_half data/mc_osm_y_dagger_mc3_half_half"
D="$D data/mc_osm_dagger_mc4_half_half data/mc_osm_y_dagger_mc4_half_half"
D="$D data/mc_osm_dagger_mc5b_half data/mc_osm_y_dagger_mc5b_half"
D="$D data/mc_osm_dagger_mc6 data/mc_osm_y_dagger_mc6"
D="$D data/mc_osm_dagger_mc7 data/mc_osm_y_dagger_mc7 data/mc_tp_dagger_mc7"
D="$D data/mc_kv_expert data/mc_tp_kv_expert data/mc_kv_dagger_mc2_half data/mc_osm_kv_dagger_mc2_half data/mc_osm_kv2_dagger_mc2_half"
D="$D data/mc_kv_dagger_mc3 data/mc_osm_kv_dagger_mc3_half data/mc_osm_kv2_dagger_mc3_half data/mc_osm_kv_dagger_mc4_half_half"
D="$D data/mc_osm_kv_dagger_mc5b_half data/mc_osm_kv_dagger_mc6 data/mc_osm_kv_dagger_mc7"
W="--dir_weight data/mc_ped_half=2 data/mc_tp2_expert_half=2 data/mc_osm_expert_half=1.5 data/mc_osm_y_expert=1.5"
W="$W data/mc_ped_dagger_mc2=2 data/mc_vped_dagger_mc2=2 data/mc_fl_dagger_mc2_half=2 data/mc_fl_tp_dagger_mc2_half=3 data/mc_tp3_dagger_mc2_half=3"
W="$W data/mc_tp_kv_expert=2 data/mc_kv_dagger_mc2_half=2 data/mc_osm_kv_dagger_mc2_half=1.5 data/mc_osm_kv2_dagger_mc2_half=1.5"
W="$W data/mc_ped_dagger_mc3=2 data/mc_vped_dagger_mc3=2 data/mc_tp3_dagger_mc3=3"
W="$W data/mc_osm_dagger_mc3_half_half=1.5 data/mc_osm_y_dagger_mc3_half_half=1.5"
W="$W data/mc_kv_dagger_mc3=2 data/mc_osm_kv_dagger_mc3_half=2 data/mc_osm_kv2_dagger_mc3_half=2"
W="$W data/mc_osm_dagger_mc4_half_half=1.5 data/mc_osm_y_dagger_mc4_half_half=1.5 data/mc_osm_kv_dagger_mc4_half_half=2"
W="$W data/mc_osm_dagger_mc5b_half=1.5 data/mc_osm_y_dagger_mc5b_half=1.5 data/mc_osm_kv_dagger_mc5b_half=2"
W="$W data/mc_osm_dagger_mc6=1.5 data/mc_osm_y_dagger_mc6=1.5 data/mc_osm_kv_dagger_mc6=2"
W="$W data/mc_osm_dagger_mc7=2 data/mc_osm_y_dagger_mc7=2 data/mc_tp_dagger_mc7=2 data/mc_osm_kv_dagger_mc7=3"
# round 8's sets
D="$D data/mc_sub_kv_dagger_mc8 data/mc_sub_dagger_mc8 data/mc_sub_d_dagger_mc8"
W="$W data/mc_sub_kv_dagger_mc8=4 data/mc_sub_dagger_mc8=4 data/mc_sub_d_dagger_mc8=3"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D) GB"
# the RAM: halve old sets (early grid / KeiView expert sets first, round 6's KeiView set last) until under 21.3 GB
for s in mc_kv_expert mc_expert_half_half mc_tp_kv_expert mc_osm_kv_dagger_mc5b_half mc_osm_expert_half mc_tp2_expert_half \
         mc_osm_kv_dagger_mc6; do
  awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.3) }' || break
  half $s
  D=" $D "; D="${D/ data\/$s / data/${s}_half }"; D="${D# }"; D="${D% }"
  W="${W/data\/$s=/data/${s}_half=}"
  echo "training data with $s halved: $(gb $D) GB"
done
[ -f runs/keipilot_mc9/last.pt ] || {
  ls -d $D >/dev/null || exit 1
  T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
  uv run scripts/train.py --data $D --init runs/keipilot_mc8/last.pt --out runs/keipilot_mc9 --epochs 4 \
      --samples_per_epoch 200000 $T $W || exit 1; }
echo MC_TRAIN_MC9_DONE

# ---- 3. evaluation
M=runs/keipilot_mc9/last.pt
TAG=keipilot_mc9_hold_red_yh_4_1_3.5_4_0.5_0.5_10
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
  ev suburban_keiview --suite suburban --renderer keiview --workers 3
  ev ramp_routes_keiview --suite ramp_routes --renderer keiview --workers 3
  echo R8_NEW_KV_DONE
  sap keiview
  echo R8_SAPPORO_KV_DONE
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
}
ks() {
  ev suburban --suite suburban --workers 6
  ev ramp_routes --suite ramp_routes --workers 6
  ev ramps --suite ramps --workers 6
  ev suburban_dense --suite suburban --dense --workers 6
  ev ramp_routes_dense --suite ramp_routes --dense --workers 6
  echo R8_NEW_KS_DONE
  sap keisim
  echo R8_SAPPORO_KS_DONE
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
echo MC_ROUND8_DONE
