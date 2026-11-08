#!/bin/bash
# Multi-camera KeiPilot, round 9 (runs/mc_work/HANDOVER.md, "9 回目"). mc9 (round 8) on the 片側2車線 towns: suburban
# 17/20 (mc8 13), ramps 12/20 (6), KeiView suburban 17/20 (15), but ramp_routes 2/30 (mc8 27) and KeiView 1/30 (26):
# 28 of 30 route deviations, every one where the route changes into the ramp's deceleration lane. mc9 stays in the
# lane beside it. Round 8's labels taught it: mc8 moved over in the taper, before the route's change (which starts
# where the deceleration lane runs beside the lane), and DAgger's labels led back out of the ramp (command left, path
# 4.5-7 m to the right; ~10x as many before the left turns of the ramps towns as of the suburban towns).
# 1. DAgger with mc9 driving in ramps towns, 80 % of the episodes via a ramp (about half of the towns have one), the
#    frames by a ramp the route takes all kept (collect.py --focus_ramp) and those ahead of the route's path into it
#    dropped (--ramp_early_drop); dense traffic in ramps towns (40 % via a ramp) instead of round 8's suburban towns
# 2. round 8's ramps sets without those labels (tools/drop_frames.py: _f), mc10 = mc9 trained on rounds 1-9 with
#    round 9's sets x3 / x3 / x2 and round 8's x3 / x3 / x2 (28 % of the samples; mc9: round 8's 24 %). For the RAM:
#    round 8's dense set, early DAgger sets and round 8's ramps sets halved in turn until under 21.3 GB
# 3. evaluation as round 8's: the 片側2車線 suites first, then Sapporo with _pat3, the 16 grid suites beside KeiView's
#    long3c
# Every step skips what is already there. Start / restart: runs/mc_work/start_round9.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q "ramp_early_drop" scripts/collect.py || { echo "collect.py has no --ramp_early_drop"; exit 1; }
R="--rig wide_narrow"
G="--mode dagger --ckpt runs/keipilot_mc9/last.pt --freeze_takeover 4 --freeze_reasons curve cruise crossing"
F="--jam_fixes 1 --focus_yield 0.5 --focus_reasons yield junction_blocked red_light lane_change pedestrian crossing stop_sign"
RAMPS="--town_style ramps --focus_ramp 1 --ramp_early_drop 1"

# ---- 1. data
for s in mc_sub_dagger_mc8 mc_sub_kv_dagger_mc8; do
  [ -d data/${s}_f ] || uv run python tools/drop_frames.py data/$s data/${s}_f || exit 1
done
[ -f data/mc_ramp_kv_dagger_mc9/.done ] || {
  uv run scripts/collect.py $R $G $F $RAMPS --via_ramp 0.8 --renderer keiview --vehicle_spacing 24 45 --ego_cross_rate 0.3 \
      --episodes_per_town 3 --out data/mc_ramp_kv_dagger_mc9 --frames 36000 --workers 3 --seed 4901 || exit 1
  touch data/mc_ramp_kv_dagger_mc9/.done; }
echo MC_COLLECT_KV_R9_DONE
[ -f data/mc_ramp_dagger_mc9/.done ] || {
  uv run scripts/collect.py $R $G $F $RAMPS --via_ramp 0.8 --out data/mc_ramp_dagger_mc9 --frames 40000 --workers 6 --seed 4902 || exit 1
  touch data/mc_ramp_dagger_mc9/.done; }
[ -f data/mc_ramp_d_dagger_mc9/.done ] || {
  uv run scripts/collect.py $R $G $F $RAMPS --via_ramp 0.4 --vehicle_spacing 20 28 --ego_cross_rate 0.3 \
      --out data/mc_ramp_d_dagger_mc9 --frames 30000 --workers 6 --seed 4903 || exit 1
  touch data/mc_ramp_d_dagger_mc9/.done; }
echo MC_COLLECT_R9_DONE

# ---- 2. mc10
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
# mc9's sets and weights (round8.sh after its RAM step), round 8's ramps sets without the labels back out of a ramp
D="data/mc_expert_half_half_half data/mc_ped_half data/mc_tp_expert_half_half data/mc_tp2_expert_half_half data/mc_v_expert_half"
D="$D data/mc_osm_expert_half_half data/mc_osm_y_expert"
D="$D data/mc_ped_dagger_mc2 data/mc_vped_dagger_mc2 data/mc_fl_dagger_mc2_half data/mc_fl_tp_dagger_mc2_half data/mc_tp3_dagger_mc2_half"
D="$D data/mc_ped_dagger_mc3 data/mc_vped_dagger_mc3 data/mc_tp3_dagger_mc3 data/mc_osm_dagger_mc3_half_half data/mc_osm_y_dagger_mc3_half_half"
D="$D data/mc_osm_dagger_mc4_half_half data/mc_osm_y_dagger_mc4_half_half"
D="$D data/mc_osm_dagger_mc5b_half data/mc_osm_y_dagger_mc5b_half"
D="$D data/mc_osm_dagger_mc6 data/mc_osm_y_dagger_mc6"
D="$D data/mc_osm_dagger_mc7 data/mc_osm_y_dagger_mc7 data/mc_tp_dagger_mc7"
D="$D data/mc_kv_expert_half data/mc_tp_kv_expert_half data/mc_kv_dagger_mc2_half data/mc_osm_kv_dagger_mc2_half data/mc_osm_kv2_dagger_mc2_half"
D="$D data/mc_kv_dagger_mc3 data/mc_osm_kv_dagger_mc3_half data/mc_osm_kv2_dagger_mc3_half data/mc_osm_kv_dagger_mc4_half_half"
D="$D data/mc_osm_kv_dagger_mc5b_half_half data/mc_osm_kv_dagger_mc6_half data/mc_osm_kv_dagger_mc7"
D="$D data/mc_sub_kv_dagger_mc8_f data/mc_sub_dagger_mc8_f data/mc_sub_d_dagger_mc8"
W="--dir_weight data/mc_ped_half=2 data/mc_tp2_expert_half_half=2 data/mc_osm_expert_half_half=1.5 data/mc_osm_y_expert=1.5"
W="$W data/mc_ped_dagger_mc2=2 data/mc_vped_dagger_mc2=2 data/mc_fl_dagger_mc2_half=2 data/mc_fl_tp_dagger_mc2_half=3 data/mc_tp3_dagger_mc2_half=3"
W="$W data/mc_tp_kv_expert_half=2 data/mc_kv_dagger_mc2_half=2 data/mc_osm_kv_dagger_mc2_half=1.5 data/mc_osm_kv2_dagger_mc2_half=1.5"
W="$W data/mc_ped_dagger_mc3=2 data/mc_vped_dagger_mc3=2 data/mc_tp3_dagger_mc3=3"
W="$W data/mc_osm_dagger_mc3_half_half=1.5 data/mc_osm_y_dagger_mc3_half_half=1.5"
W="$W data/mc_kv_dagger_mc3=2 data/mc_osm_kv_dagger_mc3_half=2 data/mc_osm_kv2_dagger_mc3_half=2"
W="$W data/mc_osm_dagger_mc4_half_half=1.5 data/mc_osm_y_dagger_mc4_half_half=1.5 data/mc_osm_kv_dagger_mc4_half_half=2"
W="$W data/mc_osm_dagger_mc5b_half=1.5 data/mc_osm_y_dagger_mc5b_half=1.5 data/mc_osm_kv_dagger_mc5b_half_half=2"
W="$W data/mc_osm_dagger_mc6=1.5 data/mc_osm_y_dagger_mc6=1.5 data/mc_osm_kv_dagger_mc6_half=2"
W="$W data/mc_osm_dagger_mc7=2 data/mc_osm_y_dagger_mc7=2 data/mc_tp_dagger_mc7=2 data/mc_osm_kv_dagger_mc7=3"
W="$W data/mc_sub_kv_dagger_mc8_f=3 data/mc_sub_dagger_mc8_f=3 data/mc_sub_d_dagger_mc8=2"
# round 9's sets
D="$D data/mc_ramp_kv_dagger_mc9 data/mc_ramp_dagger_mc9 data/mc_ramp_d_dagger_mc9"
W="$W data/mc_ramp_kv_dagger_mc9=3 data/mc_ramp_dagger_mc9=3 data/mc_ramp_d_dagger_mc9=2"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D) GB"
# the RAM: halve (round 8's dense set first: round 9's dense set is in ramps towns, then early DAgger sets, round 6's
# KeiView set, round 8's ramps sets) until under 21.3 GB
for s in mc_sub_d_dagger_mc8 mc_ped_dagger_mc2 mc_vped_dagger_mc2 mc_tp3_dagger_mc2_half mc_kv_dagger_mc2_half \
         mc_osm_kv_dagger_mc6_half mc_sub_dagger_mc8_f mc_sub_kv_dagger_mc8_f mc_ped_dagger_mc3 mc_tp3_dagger_mc3; do
  awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.3) }' || break
  half $s
  D=" $D "; D="${D/ data\/$s / data/${s}_half }"; D="${D# }"; D="${D% }"
  W="${W/data\/$s=/data/${s}_half=}"
  echo "training data with $s halved: $(gb $D) GB"
done
[ -f runs/keipilot_mc10/last.pt ] || {
  ls -d $D >/dev/null || exit 1
  T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
  uv run scripts/train.py --data $D --init runs/keipilot_mc9/last.pt --out runs/keipilot_mc10 --epochs 4 \
      --samples_per_epoch 200000 $T $W || exit 1; }
echo MC_TRAIN_MC10_DONE

# ---- 3. evaluation
M=runs/keipilot_mc10/last.pt
TAG=keipilot_mc10_hold_red_yh_4_1_3.5_4_0.5_0.5_10
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
  ev ramp_routes_keiview --suite ramp_routes --renderer keiview --workers 3
  ev suburban_keiview --suite suburban --renderer keiview --workers 3
  echo R9_NEW_KV_DONE
  sap keiview
  echo R9_SAPPORO_KV_DONE
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
}
ks() {
  ev ramp_routes --suite ramp_routes --workers 6
  ev ramps --suite ramps --workers 6
  ev suburban --suite suburban --workers 6
  ev ramp_routes_dense --suite ramp_routes --dense --workers 6
  ev suburban_dense --suite suburban --dense --workers 6
  echo R9_NEW_KS_DONE
  sap keisim
  echo R9_SAPPORO_KS_DONE
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
echo MC_ROUND9_DONE
