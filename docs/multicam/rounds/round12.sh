#!/bin/bash
# Multi-camera KeiPilot, round 12: crosswalks, big junctions and bicycles in one round (mc12 -> mc13).
# mc12 (v0.11.0) has never seen any of the three. Cloud checks on 600 m routes (KeiSim, the expert went 20/20 on each):
#   crosswalks (PR #31): 17/20, drove onto a zebra in front of people on 3 routes, hit someone on 2
#   bigjunction (PR #33, with PR #39's turn commands): 16/20 (zebras 4, red light 1, a car 1, stuck in a left turn 1)
#   bicycles (PR #35, long3c 2.5 km, --bicycles 8): 18/20, never hit a rider, mostly follows them
# The labels of the bicycle data count only the riders the wide camera sees (--bike_visible 1, PR #40): the 巻き込み
# from behind on the left that no camera of the rig sees is left as it is for now.
# 0. mc12 on the new suites: crosswalks, crosswalks_sub, bigjunction and the three with bicycles (_bk8) on KeiSim,
#    crosswalks and bigjunction on KeiView (2.5 km x 20 each)
# 1. data, each as KeiSim expert, mc12 DAgger and KeiView expert:
#    a. people on the zebras (as before): --crosswalks 0.7 --crosswalk_spawn 3, thinned to the junction decisions and
#       the exit zebras with someone in the way (--focus_crosswalk 30). Two-phase and suburban towns.
#    b. big junctions (bigjunction towns): the changes into the right-turn lanes (reason lane_change kept), scramble
#       crossings and their diagonal zebras (the same zebra focus)
#    c. bicycles: --bicycles 8 --bike_visible 1 --focus_bicycle 15 (frames with the expert's reason bicycle or a rider
#       the camera sees riding along within 15 m), with people on the zebras too. Two-phase and suburban towns. A
#       cloud check (3000 frames, --focus_bicycle 20): 31 % of the frames with the reason bicycle, half with a rider
#       about; 15 m keeps somewhat fewer
#    b and c take the settings of the _bk8 suites (README, 3 章の「自転車」): --npc_past_line 2 (an NPC standing a little
#    past its stop line obeys only its light: with the old NPCs one set off at red and locked with the crossing road's
#    right turn in the junction, and with --npc_past_line 1 one held for the ego waiting in the junction while the ego
#    waited for it), and for c --bike_sweep 1 (the expert's room beside a rider ahead counts its front corner off the
#    path: coming out of a right turn it clipped a rider standing at the start of the exit road)
# 2. mc13 = mc12 trained on mc12's sets (round 11) + the new ones, 4 x 200k samples. Weights (--dir_weight):
#    crosswalk sets 2 (as before), big junction sets 2, bicycle sets 1 (train.py weights their reason-bicycle frames
#    x3, about a quarter of them: on average ~1.5). Share of the samples, estimated as round12.sh did before (round 11's
#    80k guardrail frames at weight 1 were 2.9 %): crosswalks 80k x 2 ~ 5.5 %, big junctions 48k x 2 ~ 3 %,
#    bicycles 53k (~x 1.5) ~ 2.5 %; all new sets ~ 11 % (round 11: 2.9 %, the crosswalks alone: ~ 6 %).
#    Memory: the new sets are ~181k frames (~4 GB). MMAP=1 (train.py --mmap, PR #38) leaves the shards' images on disk
#    and keeps every old set whole; without it the old sets are halved until under 21.3 GB as before (round 11's
#    order first), and more of them than before
# 3. mc13 on the new suites, round 11's curve courses and round 10's town suites (both renderers, Sapporo _pat3)
# Written in the cloud (no GPU) and not run end to end: every collect.py command was tried on a few hundred frames (the
# DAgger ones with keipilot_mc.pt = mc12 on the CPU, the KeiView ones not: no browser in the cloud). Every step skips
# what is already there.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q -- "--focus_crosswalk" scripts/collect.py && grep -q -- "--focus_bicycle" scripts/collect.py \
    && grep -q '"bigjunction_bk8"' scripts/evaluate.py && grep -q 'turn_rule' keisim/config.py \
    || { echo "not the code with the crosswalks (PR #31), big junctions (#33, #39) and bicycles (#35, #40)"; exit 1; }
MMAP=${MMAP:-0}
[ $MMAP = 1 ] && { grep -q -- "--mmap" scripts/train.py || { echo "MMAP=1: train.py has no --mmap (PR #38)"; exit 1; }; }
M12=runs/keipilot_mc12/last.pt
[ -f $M12 ] || { echo "$M12 (round 11) is missing"; exit 1; }
R="--rig wide_narrow"
L="--brake_hold --red_hold --yield_hold 4 1 3.5 4 0.5 0.5 10"

# ---- 0. mc12 on the new suites
T12=keipilot_mc12_hold_red_yh_4_1_3.5_4_0.5_0.5_10
E12="uv run scripts/evaluate.py --agent model --ckpt $M12 $L"
for s in crosswalks crosswalks_sub bigjunction crosswalks_bk8 crosswalks_sub_bk8 bigjunction_bk8; do
  [ -f runs/eval/${T12}_$s.json ] || $E12 --suite $s --workers 6 || exit 1
done
for s in crosswalks bigjunction; do
  [ -f runs/eval/${T12}_${s}_keiview.json ] || $E12 --suite $s --renderer keiview --workers 3 || exit 1
done
echo R12_BASELINE_DONE

# ---- 1. data
TP="--town_style twophase --jam_fixes 1"
SUB="--town_style suburban --jam_fixes 1"
BJ="--town_style bigjunction --jam_fixes 1"
CW="--crosswalks 0.7 --crosswalk_spawn 3 --focus_yield 0.15 --focus_crosswalk 30"
C="$CW --focus_reasons yield junction_blocked crosswalk"
CB="$CW --focus_reasons yield junction_blocked crosswalk lane_change --npc_past_line 2"
CK="$CW --focus_reasons yield junction_blocked crosswalk bicycle --bicycles 8 --bike_visible 1 --focus_bicycle 15 \
    --npc_past_line 2 --bike_sweep 1"
col() {   # col NAME ARGS...: collect.py into data/NAME, once
  local n=$1; shift
  [ -f data/$n/.done ] && return
  uv run scripts/collect.py $R "$@" --out data/$n || return 1
  touch data/$n/.done
}
# a. crosswalks (as in the first round12.sh)
col mc_cw_expert $TP $C --frames 25000 --workers 10 --seed 3121 || exit 1
col mc_cw_sub_expert $SUB $C --frames 15000 --workers 10 --seed 3124 || exit 1
col mc_cw_dagger_mc12 $TP $C --mode dagger --ckpt $M12 --frames 20000 --workers 6 --seed 3123 || exit 1
col mc_cw_sub_dagger_mc12 $SUB $C --mode dagger --ckpt $M12 --frames 10000 --workers 6 --seed 3125 || exit 1
col mc_cw_kv_expert $TP $C --renderer keiview --episodes_per_town 3 --frames 10000 --workers 3 --seed 3122 || exit 1
# b. big junctions (collect.py's town seeds 0-399: the bigjunction suites use 3000-3009)
col mc_bj_expert $BJ $CB --frames 25000 --workers 10 --seed 3131 || exit 1
col mc_bj_dagger_mc12 $BJ $CB --mode dagger --ckpt $M12 --frames 15000 --workers 6 --seed 3133 || exit 1
col mc_bj_kv_expert $BJ $CB --renderer keiview --episodes_per_town 3 --frames 8000 --workers 3 --seed 3132 || exit 1
# c. bicycles
col mc_bk_expert $TP $CK --frames 20000 --workers 10 --seed 3141 || exit 1
col mc_bk_sub_expert $SUB $CK --frames 10000 --workers 10 --seed 3144 || exit 1
col mc_bk_dagger_mc12 $TP $CK --mode dagger --ckpt $M12 --frames 10000 --workers 6 --seed 3143 || exit 1
col mc_bk_sub_dagger_mc12 $SUB $CK --mode dagger --ckpt $M12 --frames 5000 --workers 6 --seed 3145 || exit 1
col mc_bk_kv_expert $TP $CK --renderer keiview --episodes_per_town 3 --frames 8000 --workers 3 --seed 3142 || exit 1
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
D="$D data/mc_bj_expert data/mc_bj_dagger_mc12 data/mc_bj_kv_expert"
W="$W data/mc_bj_expert=2 data/mc_bj_dagger_mc12=2 data/mc_bj_kv_expert=2"
D="$D data/mc_bk_expert data/mc_bk_sub_expert data/mc_bk_dagger_mc12 data/mc_bk_sub_dagger_mc12 data/mc_bk_kv_expert"
W="$W data/mc_bk_expert=1 data/mc_bk_sub_expert=1 data/mc_bk_dagger_mc12=1 data/mc_bk_sub_dagger_mc12=1"
W="$W data/mc_bk_kv_expert=1"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D) GB"
if [ $MMAP = 0 ]; then
# the RAM (MMAP=0): halve until under 21.3 GB, round 11's order first (the older and more general sets), then older
# DAgger sets; the new sets are ~4 GB more than the first round12.sh had, so the list goes on to more old DAgger sets
for s in mc_osm_dagger_mc7 mc_kv_expert_half mc_tp_kv_expert_half mc_tp_expert_half_half mc_osm_kv_dagger_mc4_half_half \
         mc_osm_kv_dagger_mc5b_half_half mc_osm_y_dagger_mc7 mc_osm_kv_dagger_mc7 \
         mc_osm_dagger_mc6_half mc_osm_y_dagger_mc6_half mc_osm_kv_dagger_mc6_half_half mc_tp_dagger_mc7_half \
         mc_ped_dagger_mc3_half mc_vped_dagger_mc3_half mc_kv_dagger_mc3_half mc_tp3_dagger_mc3_half \
         mc_osm_dagger_mc5b_half mc_osm_y_dagger_mc5b_half mc_osm_dagger_mc4_half_half mc_osm_y_dagger_mc4_half_half \
         mc_osm_dagger_mc3_half_half mc_osm_y_dagger_mc3_half_half mc_fl_dagger_mc2_half mc_ped_dagger_mc2_half; do
  awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.3) }' || break
  half $s
  D=" $D "; D="${D/ data\/$s / data/${s}_half }"; D="${D# }"; D="${D% }"
  W="${W/data\/$s=/data/${s}_half=}"
  echo "training data with $s halved: $(gb $D) GB"
done
awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.5) }' && { echo "training data still $(gb $D) GB"; exit 1; }
fi
[ -f runs/keipilot_mc13/last.pt ] || {
  ls -d $D >/dev/null || exit 1
  T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
  [ $MMAP = 1 ] && T="$T --mmap"
  uv run scripts/train.py --data $D --init $M12 --out runs/keipilot_mc13 --epochs 4 \
      --samples_per_epoch 200000 $T $W || exit 1; }
echo R12_TRAIN_MC13_DONE

# ---- 3. evaluation
M=runs/keipilot_mc13/last.pt
TAG=keipilot_mc13_hold_red_yh_4_1_3.5_4_0.5_0.5_10
E="uv run scripts/evaluate.py --agent model --ckpt $M $L"
ev() { local name=$1; shift; [ -f runs/eval/${TAG}_$name.json ] || $E "$@"; }
for s in crosswalks crosswalks_sub bigjunction crosswalks_bk8 crosswalks_sub_bk8 bigjunction_bk8; do
  ev $s --suite $s --workers 6
done
ev crosswalks_keiview --suite crosswalks --renderer keiview --workers 3
ev bigjunction_keiview --suite bigjunction --renderer keiview --workers 3
echo R12_NEW_SUITES_MC13_DONE
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
