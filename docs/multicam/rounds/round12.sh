#!/bin/bash
# Multi-camera KeiPilot, round 12: crosswalks at signals (README, 3 章の「横断歩道を渡る人と歩行者用の信号」, PR #31).
# mc12 (v0.11.0) has never seen people crossing on the zebras with their pedestrian light: on the crosswalks suite's
# towns with 600 m routes (KeiSim, measured on a cloud CPU) it drove onto a zebra in front of people on 3 of 20 routes
# and hit someone on 2 of them, where the expert went 20/20 without infractions (runs/eval/*_crosswalks_600m.json).
# 0. mc12 on the crosswalk suites (2.5 km x 20): crosswalks and crosswalks_sub on KeiSim, crosswalks on KeiView
# 1. data with people on the zebras (--crosswalks 0.7) and 3 corner people per zebra (--crosswalk_spawn 3), thinned to
#    the junction decisions and the zebras at the exits of the route's junctions with someone in the way
#    (--focus_yield 0.15 --focus_reasons yield junction_blocked crosswalk --focus_crosswalk 30). A cloud check of the
#    KeiSim expert (3000 frames, two-phase towns): 2.2 % of the frames with the expert giving way at a zebra and 19 % by
#    an exit zebra with someone in the way, against 0.2 % / none without the thinning and with 1 corner person.
#    Two-phase towns and suburban towns (片側2車線: longer zebras): KeiSim expert 25k + 15k, KeiView expert 10k,
#    DAgger with mc12 20k + 10k
# 2. mc13 = mc12 trained on mc12's sets (round 11) + the 5 new ones, 4 x 200k samples. The new sets at weight 2:
#    round 11's 80k guardrail frames at weight 1 were 2.9 % of the samples, so these should be about 6 %
#    (scripts/train.py also weights the frames giving way at a zebra x3, as the pedestrian ones). Old sets halved
#    until under 21.3 GB, round 11's order first
# 3. mc13 on the crosswalk suites, round 11's curve courses and round 10's town suites (both renderers, Sapporo _pat3)
# Written in the cloud (no GPU) from round11.sh and not run end to end: the three collect.py commands were tried on a
# few hundred frames each (the DAgger one with keipilot_mc.pt = mc12 on the CPU). Every step skips what is already
# there.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q -- "--focus_crosswalk" scripts/collect.py && grep -q '"crosswalks_sub"' scripts/evaluate.py \
    || { echo "not the code with the crosswalks (PR #31) and round 12's collect.py options"; exit 1; }
M12=runs/keipilot_mc12/last.pt
[ -f $M12 ] || { echo "$M12 (round 11) is missing"; exit 1; }
R="--rig wide_narrow"
L="--brake_hold --red_hold --yield_hold 4 1 3.5 4 0.5 0.5 10"

# ---- 0. mc12 on the crosswalk suites
T12=keipilot_mc12_hold_red_yh_4_1_3.5_4_0.5_0.5_10
E12="uv run scripts/evaluate.py --agent model --ckpt $M12 $L"
[ -f runs/eval/${T12}_crosswalks.json ] || $E12 --suite crosswalks --workers 6 || exit 1
[ -f runs/eval/${T12}_crosswalks_sub.json ] || $E12 --suite crosswalks_sub --workers 6 || exit 1
[ -f runs/eval/${T12}_crosswalks_keiview.json ] || $E12 --suite crosswalks --renderer keiview --workers 3 || exit 1
echo R12_BASELINE_DONE

# ---- 1. data
TP="--town_style twophase --jam_fixes 1"
SUB="--town_style suburban --jam_fixes 1"
C="--crosswalks 0.7 --crosswalk_spawn 3 --focus_yield 0.15 --focus_reasons yield junction_blocked crosswalk --focus_crosswalk 30"
col() {   # col NAME ARGS...: collect.py into data/NAME, once
  local n=$1; shift
  [ -f data/$n/.done ] && return
  uv run scripts/collect.py $R $C "$@" --out data/$n || return 1
  touch data/$n/.done
}
col mc_cw_expert $TP --frames 25000 --workers 10 --seed 3121 || exit 1
col mc_cw_sub_expert $SUB --frames 15000 --workers 10 --seed 3124 || exit 1
col mc_cw_dagger_mc12 $TP --mode dagger --ckpt $M12 --frames 20000 --workers 6 --seed 3123 || exit 1
col mc_cw_sub_dagger_mc12 $SUB --mode dagger --ckpt $M12 --frames 10000 --workers 6 --seed 3125 || exit 1
col mc_cw_kv_expert $TP --renderer keiview --episodes_per_town 3 --frames 10000 --workers 3 --seed 3122 || exit 1
echo R12_COLLECT_DONE

# ---- 2. mc13
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
# mc12's sets and weights before round 11's halving (round11.sh); the halving below redoes round 11's first
D="data/mc_expert_half_half_half data/mc_ped_half data/mc_tp_expert_half_half"
D="$D data/mc_tp2_expert_half_half data/mc_v_expert_half data/mc_osm_expert_half_half"
D="$D data/mc_osm_y_expert_half data/mc_ped_dagger_mc2_half data/mc_vped_dagger_mc2_half"
D="$D data/mc_fl_dagger_mc2_half data/mc_fl_tp_dagger_mc2_half_half data/mc_tp3_dagger_mc2_half_half"
D="$D data/mc_ped_dagger_mc3_half data/mc_vped_dagger_mc3_half data/mc_tp3_dagger_mc3_half"
D="$D data/mc_osm_dagger_mc3_half_half data/mc_osm_y_dagger_mc3_half_half data/mc_osm_dagger_mc4_half_half"
D="$D data/mc_osm_y_dagger_mc4_half_half data/mc_osm_dagger_mc5b_half data/mc_osm_y_dagger_mc5b_half"
D="$D data/mc_osm_dagger_mc6_half data/mc_osm_y_dagger_mc6_half data/mc_osm_dagger_mc7"
D="$D data/mc_osm_y_dagger_mc7 data/mc_tp_dagger_mc7_half data/mc_kv_expert_half data/mc_tp_kv_expert_half"
D="$D data/mc_kv_dagger_mc2_half_half data/mc_osm_kv_dagger_mc2_half data/mc_osm_kv2_dagger_mc2_half"
D="$D data/mc_kv_dagger_mc3_half data/mc_osm_kv_dagger_mc3_half_half data/mc_osm_kv2_dagger_mc3_half_half"
D="$D data/mc_osm_kv_dagger_mc4_half_half data/mc_osm_kv_dagger_mc5b_half_half"
D="$D data/mc_osm_kv_dagger_mc6_half_half data/mc_osm_kv_dagger_mc7 data/mc_sub_kv_dagger_mc8_f_half"
D="$D data/mc_sub_dagger_mc8_f_half data/mc_sub_d_dagger_mc8_half data/mc_ramp_kv_dagger_mc9_half"
D="$D data/mc_ramp_dagger_mc9 data/mc_ramp_d_dagger_mc9 data/mc_tp_kv_dagger_mc10 data/mc_tp_dagger_mc10"
D="$D data/mc_tp3_dagger_mc10 data/mc_sub_dagger_mc10"
W="--dir_weight data/mc_ped_half=2 data/mc_tp2_expert_half_half=2 data/mc_osm_expert_half_half=1.5"
W="$W data/mc_osm_y_expert_half=1.5 data/mc_ped_dagger_mc2_half=2 data/mc_vped_dagger_mc2_half=2"
W="$W data/mc_fl_dagger_mc2_half=2 data/mc_fl_tp_dagger_mc2_half_half=3 data/mc_tp3_dagger_mc2_half_half=3"
W="$W data/mc_tp_kv_expert_half=2 data/mc_kv_dagger_mc2_half_half=2 data/mc_osm_kv_dagger_mc2_half=1.5"
W="$W data/mc_osm_kv2_dagger_mc2_half=1.5 data/mc_ped_dagger_mc3_half=2 data/mc_vped_dagger_mc3_half=2"
W="$W data/mc_tp3_dagger_mc3_half=3 data/mc_osm_dagger_mc3_half_half=1.5"
W="$W data/mc_osm_y_dagger_mc3_half_half=1.5 data/mc_kv_dagger_mc3_half=2"
W="$W data/mc_osm_kv_dagger_mc3_half_half=2 data/mc_osm_kv2_dagger_mc3_half_half=2"
W="$W data/mc_osm_dagger_mc4_half_half=1.5 data/mc_osm_y_dagger_mc4_half_half=1.5"
W="$W data/mc_osm_kv_dagger_mc4_half_half=2 data/mc_osm_dagger_mc5b_half=1.5"
W="$W data/mc_osm_y_dagger_mc5b_half=1.5 data/mc_osm_kv_dagger_mc5b_half_half=2"
W="$W data/mc_osm_dagger_mc6_half=1.5 data/mc_osm_y_dagger_mc6_half=1.5"
W="$W data/mc_osm_kv_dagger_mc6_half_half=2 data/mc_osm_dagger_mc7=2 data/mc_osm_y_dagger_mc7=2"
W="$W data/mc_tp_dagger_mc7_half=2 data/mc_osm_kv_dagger_mc7=3 data/mc_sub_kv_dagger_mc8_f_half=3"
W="$W data/mc_sub_dagger_mc8_f_half=3 data/mc_sub_d_dagger_mc8_half=2 data/mc_ramp_kv_dagger_mc9_half=3"
W="$W data/mc_ramp_dagger_mc9=3 data/mc_ramp_d_dagger_mc9=2 data/mc_tp_kv_dagger_mc10=3"
W="$W data/mc_tp_dagger_mc10=3 data/mc_tp3_dagger_mc10=2 data/mc_sub_dagger_mc10=2"
# round 11's sets
D="$D data/mc_guard_expert data/mc_guard_kv_expert data/mc_guard_dagger_mc11"
W="$W data/mc_guard_expert=1 data/mc_guard_kv_expert=1 data/mc_guard_dagger_mc11=1"
# round 12's sets
D="$D data/mc_cw_expert data/mc_cw_sub_expert data/mc_cw_dagger_mc12 data/mc_cw_sub_dagger_mc12 data/mc_cw_kv_expert"
W="$W data/mc_cw_expert=2 data/mc_cw_sub_expert=2 data/mc_cw_dagger_mc12=2 data/mc_cw_sub_dagger_mc12=2"
W="$W data/mc_cw_kv_expert=2"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D) GB"
# the RAM: halve until under 21.3 GB, round 11's order first (the older and more general sets), then older DAgger sets
for s in mc_osm_dagger_mc7 mc_kv_expert_half mc_tp_kv_expert_half mc_tp_expert_half_half mc_osm_kv_dagger_mc4_half_half \
         mc_osm_kv_dagger_mc5b_half_half mc_osm_y_dagger_mc7 mc_osm_kv_dagger_mc7 \
         mc_osm_dagger_mc6_half mc_osm_y_dagger_mc6_half mc_osm_kv_dagger_mc6_half_half mc_tp_dagger_mc7_half \
         mc_ped_dagger_mc3_half mc_vped_dagger_mc3_half mc_kv_dagger_mc3_half mc_tp3_dagger_mc3_half; do
  awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.3) }' || break
  half $s
  D=" $D "; D="${D/ data\/$s / data/${s}_half }"; D="${D# }"; D="${D% }"
  W="${W/data\/$s=/data/${s}_half=}"
  echo "training data with $s halved: $(gb $D) GB"
done
awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.5) }' && { echo "training data still $(gb $D) GB"; exit 1; }
[ -f runs/keipilot_mc13/last.pt ] || {
  ls -d $D >/dev/null || exit 1
  T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
  uv run scripts/train.py --data $D --init $M12 --out runs/keipilot_mc13 --epochs 4 \
      --samples_per_epoch 200000 $T $W || exit 1; }
echo R12_TRAIN_MC13_DONE

# ---- 3. evaluation
M=runs/keipilot_mc13/last.pt
TAG=keipilot_mc13_hold_red_yh_4_1_3.5_4_0.5_0.5_10
E="uv run scripts/evaluate.py --agent model --ckpt $M $L"
ev() { local name=$1; shift; [ -f runs/eval/${TAG}_$name.json ] || $E "$@"; }
ev crosswalks --suite crosswalks --workers 6
ev crosswalks_sub --suite crosswalks_sub --workers 6
ev crosswalks_keiview --suite crosswalks --renderer keiview --workers 3
echo R12_CROSSWALKS_MC13_DONE
curve() {   # curve NAME CKPT: round 11's curve courses
  local n=$1 C="uv run scripts/curve_test.py --agent model --ckpt $2 $L --tp ahead"
  for ep in 0 1 2 3; do $C --weather noon --episode $ep --variants weak weak_lines --tag ${n}_weak_ep$ep || return 1; done
  $C --weather noon --variants weak weak_lines --guardrail_light flat --tag ${n}_weak_flat || return 1
  for ep in 0 1 2 3; do $C --weather noon --episode $ep --variants nolines_guard nolines_guard_nobld lines_guard \
      --tag ${n}_curve_ep$ep || return 1; done
  $C --renderer keiview --variants weak --tag ${n}_weak_keiview || return 1
}
[ -f runs/eval/curve/mc13_weak_keiview.json ] || curve mc13 $M || exit 1
echo R12_CURVE_MC13_DONE
P=3                     # --ego_patience of Sapporo's evaluations: suffix _pat3
sap() {   # Sapporo with _pat$P: sap keisim|keiview
  local batches="100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500" rend="" opt="--workers 6"
  [ $1 = keiview ] && { rend=_keiview; opt="--renderer keiview --workers 3"; batches="100 200 300"; }
  for e in $batches; do
    local f=runs/eval/osm_sapporo/${TAG}$([ $e = 100 ] || echo _ep$e)${rend}_pat$P
    [ -f $f.json ] || $E --osm_town osm_sapporo --routes 20 --episode0 $e --ego_patience $P $opt --out $f.json
  done
}
kv() {
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
  echo R12_TP_KV_DONE
  ev ramp_routes_keiview --suite ramp_routes --renderer keiview --workers 3
  ev suburban_keiview --suite suburban --renderer keiview --workers 3
  echo R12_NEW_KV_DONE
  sap keiview
  echo R12_SAPPORO_KV_DONE
}
ks() {
  for a in long3c long3d long3e long3f; do ev $a --suite $a --workers 6; ev ${a}_dense --suite $a --dense --workers 6; done
  echo R12_TP_KS_DONE
  ev ramp_routes --suite ramp_routes --workers 6
  ev ramps --suite ramps --workers 6
  ev suburban --suite suburban --workers 6
  ev ramp_routes_dense --suite ramp_routes --dense --workers 6
  ev suburban_dense --suite suburban --dense --workers 6
  echo R12_NEW_KS_DONE
  sap keisim
  echo R12_SAPPORO_KS_DONE
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
echo MC_ROUND12_DONE
