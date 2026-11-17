#!/bin/bash
# Multi-camera KeiPilot, round 13: parked cars and children darting out from in front of them (mc13 -> mc14).
# Needs the code of PR #44 (parked cars, --parked/--dart_out, reason 11) and #45 (convoys past them).
# mc12 (keipilot_mc.pt, v0.11.0) has never seen a parked car. Cloud checks (KeiSim, parked suite, 600 m x 20):
#   mc12 6/20 (the expert 20/20; the same routes without parked cars: mc12 20/20). It never passes a parked car: it stops
#   behind it and waits while the expert says go (11 routes blocked), and it does not give way to an oncoming car that
#   comes out round parked cars on its side (2 collisions; the NPC set off when mc12 could still stop at 0.6 and
#   1.6 m/s^2). It stopped for all 6 children who ran out. mc13 (round 12) has not seen parked cars either.
# 0. mc13 on the parked suite (2.5 km x 20, KeiSim and KeiView; KeiView draws the parked cars as ordinary cars) and
#    on its 600 m routes (the cloud numbers above)
# 1. data, all with --parked 4 --dart_out 0.6 (the parked suite's), thinned to the frames by parked cars:
#    --focus_yield 0.15 --focus_parked 30 --focus_reasons yield junction_blocked parked (frames with the expert's
#    reason parked, a group of parked cars on the route within 30 m, a child running out). Two-phase towns (2.4 groups
#    of parked cars on a 600 m route), road-types towns (1.4, the most alleys: passing over the centre line) and
#    suburban ones (1.6). Cloud tries (KeiSim, 300-600 frames each): reason parked 22-27 % of the expert's frames,
#    48 % (two-phase) and 29 % (road types) of mc12's DAgger frames; by parked cars or a child 39-64 %.
#    The DAgger sets hand a freeze behind a parked car to the expert after 4 s (--freeze_takeover 4 --freeze_reasons
#    ... parked): mc12 stood there for the whole blocked_timeout. collect.py's NPC caps are twice the suite's (non-classic
#    towns): the expert went 19/20 (road types, one timeout at 95 %) and 20/20 (two-phase) on 1 km routes with them.
# 2. mc14 = mc13 trained on round 12's sets + the new ones, 4 x 200k samples. Weights (--dir_weight): the parked
#    sets 2, as round 12's crosswalk and big junction sets. Every weight of train.py on the cloud tries: a frame of
#    the parked expert sets weighs 1.81 (two-phase) and 2.02 (road types) against 1.44 of an ordinary set (reason 11
#    x2, the children's reason 3 x3), mc12's DAgger frames 4.98 (with --dagger_weight 2; moving off behind a parked
#    car x3; an ordinary DAgger frame 1.44 x 2). Share of the samples, from round 12's estimate (~3.03M frame-weights
#    with its sets): the new sets 79k frames x 2 x ~1.4 (expert 1.26-1.40, DAgger 1.73 of an ordinary frame) ~ 7 %
#    (round 12's crosswalks: ~6 %), about 45 % of it on reason-11 frames (~3 % of all the samples).
#    Memory: ~80k frames, ~2 GB (25 KB a KeiSim frame, 42 KB a KeiView one). MMAP=1 (train.py --mmap) is the way:
#    without it the old sets are halved as in round 12 until under 21.3 GB.
# 3. mc14 on the parked suite, round 12's new suites, round 11's curve courses and round 10's town suites
# Written in the cloud (no GPU) and not run end to end: every collect.py command was tried on 200-600 frames (the
# DAgger ones with keipilot_mc.pt = mc12 on the CPU, the KeiView one on 30 frames with SwiftShader). Every step skips
# what is already there.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q -- "--focus_parked" scripts/collect.py && grep -q '"parked"' scripts/evaluate.py \
    && grep -q 'PK_CONVOY' keisim/parked.py \
    || { echo "not the code with the parked cars (PR #44) and their convoys (#45)"; exit 1; }
MMAP=${MMAP:-0}
[ $MMAP = 1 ] && { grep -q -- "--mmap" scripts/train.py || { echo "MMAP=1: train.py has no --mmap (PR #38)"; exit 1; }; }
M13=runs/keipilot_mc13/last.pt
[ -f $M13 ] || { echo "$M13 (round 12) is missing"; exit 1; }
R="--rig wide_narrow"
L="--brake_hold --red_hold --yield_hold 4 1 3.5 4 0.5 0.5 10"

# ---- 0. mc13 on the parked suite
T13=keipilot_mc13_hold_red_yh_4_1_3.5_4_0.5_0.5_10
E13="uv run scripts/evaluate.py --agent model --ckpt $M13 $L"
[ -f runs/eval/${T13}_parked.json ] || $E13 --suite parked --workers 6 || exit 1
[ -f runs/eval/${T13}_parked_600m.json ] || $E13 --suite parked --route_length 600 --max_steps 4000 --workers 6 \
    --out runs/eval/${T13}_parked_600m.json || exit 1
[ -f runs/eval/${T13}_parked_keiview.json ] || $E13 --suite parked --renderer keiview --workers 3 || exit 1
echo R13_BASELINE_DONE

# ---- 1. data
TP="--town_style twophase --jam_fixes 1"
RT="--town_style roadtypes --jam_fixes 1"
SUB="--town_style suburban --jam_fixes 1"
C="--parked 4 --dart_out 0.6 --focus_yield 0.15 --focus_parked 30 --focus_reasons yield junction_blocked parked"
G="--mode dagger --ckpt $M13 --freeze_takeover 4 --freeze_reasons curve cruise crossing parked"
col() {   # col NAME ARGS...: collect.py into data/NAME, once
  local n=$1; shift
  [ -f data/$n/.done ] && return
  uv run scripts/collect.py $R "$@" --out data/$n || return 1
  touch data/$n/.done
}
col mc_pk_expert $TP $C --frames 25000 --workers 10 --seed 3151 || exit 1
col mc_pk_rt_expert $RT $C --frames 15000 --workers 10 --seed 3154 || exit 1
col mc_pk_sub_expert $SUB $C --frames 8000 --workers 10 --seed 3156 || exit 1
col mc_pk_dagger_mc13 $TP $C $G --frames 15000 --workers 6 --seed 3153 || exit 1
col mc_pk_rt_dagger_mc13 $RT $C $G --frames 8000 --workers 6 --seed 3155 || exit 1
col mc_pk_kv_expert $TP $C --renderer keiview --episodes_per_town 3 --frames 8000 --workers 3 --seed 3152 || exit 1
echo R13_COLLECT_DONE

# ---- 2. mc14
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
# mc13's sets and weights (round12.sh, before its halving)
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
# round 13's sets
D="$D data/mc_pk_expert data/mc_pk_rt_expert data/mc_pk_sub_expert data/mc_pk_dagger_mc13 data/mc_pk_rt_dagger_mc13"
D="$D data/mc_pk_kv_expert"
W="$W data/mc_pk_expert=2 data/mc_pk_rt_expert=2 data/mc_pk_sub_expert=2 data/mc_pk_dagger_mc13=2"
W="$W data/mc_pk_rt_dagger_mc13=2 data/mc_pk_kv_expert=2"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D) GB"
if [ $MMAP = 0 ]; then
# the RAM (MMAP=0): halve until under 21.3 GB, round 12's order (the older and more general sets, then older DAgger
# sets); round 13's ~2 GB more may take a few more of them than round 12 did
for s in mc_osm_dagger_mc7 mc_kv_expert_half mc_tp_kv_expert_half mc_tp_expert_half_half mc_osm_kv_dagger_mc4_half_half \
         mc_osm_kv_dagger_mc5b_half_half mc_osm_y_dagger_mc7 mc_osm_kv_dagger_mc7 \
         mc_osm_dagger_mc6_half mc_osm_y_dagger_mc6_half mc_osm_kv_dagger_mc6_half_half mc_tp_dagger_mc7_half \
         mc_ped_dagger_mc3_half mc_vped_dagger_mc3_half mc_kv_dagger_mc3_half mc_tp3_dagger_mc3_half \
         mc_osm_dagger_mc5b_half mc_osm_y_dagger_mc5b_half mc_osm_dagger_mc4_half_half mc_osm_y_dagger_mc4_half_half \
         mc_osm_dagger_mc3_half_half mc_osm_y_dagger_mc3_half_half mc_fl_dagger_mc2_half mc_ped_dagger_mc2_half \
         mc_vped_dagger_mc2_half mc_osm_kv_dagger_mc2_half mc_osm_kv2_dagger_mc2_half mc_kv_dagger_mc2_half_half; do
  awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.3) }' || break
  half $s
  D=" $D "; D="${D/ data\/$s / data/${s}_half }"; D="${D# }"; D="${D% }"
  W="${W/data\/$s=/data/${s}_half=}"
  echo "training data with $s halved: $(gb $D) GB"
done
awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.5) }' && { echo "training data still $(gb $D) GB: use MMAP=1"; exit 1; }
fi
[ -f runs/keipilot_mc14/last.pt ] || {
  ls -d $D >/dev/null || exit 1
  T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
  [ $MMAP = 1 ] && T="$T --mmap"
  uv run scripts/train.py --data $D --init $M13 --out runs/keipilot_mc14 --epochs 4 \
      --samples_per_epoch 200000 $T $W || exit 1; }
echo R13_TRAIN_MC14_DONE

# ---- 3. evaluation
M=runs/keipilot_mc14/last.pt
TAG=keipilot_mc14_hold_red_yh_4_1_3.5_4_0.5_0.5_10
E="uv run scripts/evaluate.py --agent model --ckpt $M $L"
ev() { local name=$1; shift; [ -f runs/eval/${TAG}_$name.json ] || $E "$@"; }
ev parked --suite parked --workers 6
ev parked_600m --suite parked --route_length 600 --max_steps 4000 --workers 6 --out runs/eval/${TAG}_parked_600m.json
ev parked_keiview --suite parked --renderer keiview --workers 3
echo R13_PARKED_MC14_DONE
for s in crosswalks crosswalks_sub bigjunction crosswalks_bk8 crosswalks_sub_bk8 bigjunction_bk8; do
  ev $s --suite $s --workers 6
done
ev crosswalks_keiview --suite crosswalks --renderer keiview --workers 3
ev bigjunction_keiview --suite bigjunction --renderer keiview --workers 3
echo R13_R12_SUITES_MC14_DONE
curve() {   # curve NAME CKPT: round 11's curve courses
  local n=$1 C="uv run scripts/curve_test.py --agent model --ckpt $2 $L --tp ahead"
  for ep in 0 1 2 3; do $C --weather noon --episode $ep --variants weak weak_lines --tag ${n}_weak_ep$ep || return 1; done
  $C --weather noon --variants weak weak_lines --guardrail_light flat --tag ${n}_weak_flat || return 1
  for ep in 0 1 2 3; do $C --weather noon --episode $ep --variants nolines_guard nolines_guard_nobld lines_guard \
      --tag ${n}_curve_ep$ep || return 1; done
  $C --renderer keiview --variants weak --tag ${n}_weak_keiview || return 1
}
[ -f runs/eval/curve/mc14_weak_keiview.json ] || curve mc14 $M || exit 1
echo R13_CURVE_MC14_DONE
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
  echo R13_TP_KV_DONE
  ev ramp_routes_keiview --suite ramp_routes --renderer keiview --workers 3
  ev suburban_keiview --suite suburban --renderer keiview --workers 3
  echo R13_NEW_KV_DONE
  sap keiview
  echo R13_SAPPORO_KV_DONE
}
ks() {
  for a in long3c long3d long3e long3f; do ev $a --suite $a --workers 6; ev ${a}_dense --suite $a --dense --workers 6; done
  echo R13_TP_KS_DONE
  ev ramp_routes --suite ramp_routes --workers 6
  ev ramps --suite ramps --workers 6
  ev suburban --suite suburban --workers 6
  ev ramp_routes_dense --suite ramp_routes --dense --workers 6
  ev suburban_dense --suite suburban --dense --workers 6
  echo R13_NEW_KS_DONE
  sap keisim
  echo R13_SAPPORO_KS_DONE
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
echo MC_ROUND13_DONE
