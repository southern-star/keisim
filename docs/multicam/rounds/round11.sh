#!/bin/bash
# Multi-camera KeiPilot, round 11: the guardrail pull on sharp curves without white lines (README, 2 章の「カーブの
# 試験コース」). The same recipe, tried first on a single-camera model (not released), went from contacts on all 10
# runs of the weak-point course to none, and from 12 of 16 to 0 contacts on the curve courses. Applied here to the
# multi-camera model with mc8's standard layers.
# 0. mc11 on the test suite with the merged code (f46dc2c): KeiSim must give the round 10 routes again (KeiView keeps
#    its exported towns in web/.towns, so the regular towns keep their pre-merge lots)
# 1. mc11 on the curve courses: weak / weak_lines x sunny episodes 0-3 + flat guardrails, both ways;
#    nolines_guard / nolines_guard_nobld / lines_guard x episodes 0-3; weak on KeiView; --tp ahead
# 2. the guardrail scenes (--curve_scenes 1) with the rig: KeiSim expert 40k, KeiView expert 10k, DAgger with mc11 30k
# 3. mc12 = mc11 trained on mc11's sets + the 3 new ones, 4 x 200k samples; the new sets at weight 1 = 2.9 % of the
#    samples, as the 3.0 % of v26 (weight 2 in a 2.6M-frame set); old sets halved until under 21.3 GB
# 4. mc12 on the curve courses, then round 10's town suites (both renderers, Sapporo _pat3)
# Every step skips what is already there. Start / restart: runs/mc_work/start_round11.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
[ -f keisim/curve_course.py ] && grep -q -- "--curve_scenes" scripts/collect.py && grep -q -- '"--rig"' scripts/curve_test.py \
    || { echo "not the merged code (f46dc2c + 7199e4c)"; exit 1; }
# the single-camera v26 evaluations (main checkout's worktree guardrail-v26) use the CPU and RAM until about 23:00
while systemctl --user is-active --quiet guard-phase2; do sleep 60; done
R="--rig wide_narrow"
L="--brake_hold --red_hold --yield_hold 4 1 3.5 4 0.5 0.5 10"

# ---- 0. the merge and KeiSim
T11=keipilot_mc11_hold_red_yh_4_1_3.5_4_0.5_0.5_10
[ -f runs/eval/postmerge_${T11}_test.json ] || uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_mc11/last.pt $L \
    --suite test --workers 6 --out runs/eval/postmerge_${T11}_test.json
echo R11_POSTMERGE_DONE

curve() {   # curve NAME CKPT: the curve courses
  local n=$1 C="uv run scripts/curve_test.py --agent model --ckpt $2 $L --tp ahead"
  for ep in 0 1 2 3; do $C --weather noon --episode $ep --variants weak weak_lines --tag ${n}_weak_ep$ep || return 1; done
  $C --weather noon --variants weak weak_lines --guardrail_light flat --tag ${n}_weak_flat || return 1
  for ep in 0 1 2 3; do $C --weather noon --episode $ep --variants nolines_guard nolines_guard_nobld lines_guard \
      --tag ${n}_curve_ep$ep || return 1; done
  $C --renderer keiview --variants weak --tag ${n}_weak_keiview || return 1
}
# ---- 1. mc11 on the curve courses
[ -f runs/eval/curve/mc11_weak_keiview.json ] || curve mc11 runs/keipilot_mc11/last.pt || exit 1
echo R11_CURVE_MC11_DONE

# ---- 2. data
S="--curve_scenes 1"
[ -f data/mc_guard_expert/.done ] || {
  uv run scripts/collect.py $R $S --out data/mc_guard_expert --frames 40000 --workers 10 --seed 3111 || exit 1
  touch data/mc_guard_expert/.done; }
[ -f data/mc_guard_dagger_mc11/.done ] || {
  uv run scripts/collect.py $R $S --mode dagger --ckpt runs/keipilot_mc11/last.pt --out data/mc_guard_dagger_mc11 \
      --frames 30000 --workers 6 --seed 3113 || exit 1
  touch data/mc_guard_dagger_mc11/.done; }
[ -f data/mc_guard_kv_expert/.done ] || {
  uv run scripts/collect.py $R $S --renderer keiview --episodes_per_town 3 --out data/mc_guard_kv_expert --frames 10000 \
      --workers 3 --seed 3112 || exit 1
  touch data/mc_guard_kv_expert/.done; }
echo R11_COLLECT_DONE

# ---- 3. mc12
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
# mc11's sets and weights (its train.py command in logs/mc_round10.log)
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
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D) GB"
# the RAM: halve until under 21.3 GB, the older and more general sets first (round 7's Sapporo KeiView set last)
for s in mc_osm_dagger_mc7 mc_kv_expert_half mc_tp_kv_expert_half mc_tp_expert_half_half mc_osm_kv_dagger_mc4_half_half \
         mc_osm_kv_dagger_mc5b_half_half mc_osm_y_dagger_mc7 mc_osm_kv_dagger_mc7; do
  awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.3) }' || break
  half $s
  D=" $D "; D="${D/ data\/$s / data/${s}_half }"; D="${D# }"; D="${D% }"
  W="${W/data\/$s=/data/${s}_half=}"
  echo "training data with $s halved: $(gb $D) GB"
done
awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.5) }' && { echo "training data still $(gb $D) GB"; exit 1; }
[ -f runs/keipilot_mc12/last.pt ] || {
  ls -d $D >/dev/null || exit 1
  T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
  uv run scripts/train.py --data $D --init runs/keipilot_mc11/last.pt --out runs/keipilot_mc12 --epochs 4 \
      --samples_per_epoch 200000 $T $W || exit 1; }
echo R11_TRAIN_MC12_DONE

# ---- 4. evaluation
[ -f runs/eval/curve/mc12_weak_keiview.json ] || curve mc12 runs/keipilot_mc12/last.pt || exit 1
echo R11_CURVE_MC12_DONE
M=runs/keipilot_mc12/last.pt
TAG=keipilot_mc12_hold_red_yh_4_1_3.5_4_0.5_0.5_10
E="uv run scripts/evaluate.py --agent model --ckpt $M $L"
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
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
  echo R11_TP_KV_DONE
  ev ramp_routes_keiview --suite ramp_routes --renderer keiview --workers 3
  ev suburban_keiview --suite suburban --renderer keiview --workers 3
  echo R11_NEW_KV_DONE
  sap keiview
  echo R11_SAPPORO_KV_DONE
}
ks() {
  for a in long3c long3d long3e long3f; do ev $a --suite $a --workers 6; ev ${a}_dense --suite $a --dense --workers 6; done
  echo R11_TP_KS_DONE
  ev ramp_routes --suite ramp_routes --workers 6
  ev ramps --suite ramps --workers 6
  ev suburban --suite suburban --workers 6
  ev ramp_routes_dense --suite ramp_routes --dense --workers 6
  ev suburban_dense --suite suburban --dense --workers 6
  echo R11_NEW_KS_DONE
  sap keisim
  echo R11_SAPPORO_KS_DONE
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
echo MC_ROUND11_DONE
