#!/bin/bash
# Multi-camera KeiPilot, round 13rt: right-turn yields (mc13 -> mc14rt). Run 11-16 08:26 to 11-17 13:11 as round13.sh
# with mc14 and renamed afterwards (mc14 -> mc14rt, R13_ -> R13RT_): round13.sh of PR #46 (parked cars) trains its
# own mc14 and skips what is already there. Not adopted: mc14rt equals mc13 on every suite group (no significant
# difference) and stands in as many right-turn standoffs (片側2車線 7 = 7, the fresh suites 4 = 4); see the rounds README.
# mc13 (round 12) stands too deep into its right turns. At green it sets off into the turn; the expert at once says
# yield and stops at its wait point, mc13 about 0.5 s later and 0.6-3 m further on, its nose in the oncoming lane. The
# oncoming car stops in front of it and the two wait for each other until blocked_timeout: the simulator breaks no
# standoff, and evaluate.py counted the route as blocked by "traffic" (where the ego stood, the expert would have
# waited too). With the expert driving from the green on, all 5 of the 片側2車線 jams only mc13 had went through.
# From the failure logs: mc13 19 such jams, mc12 15, mc11 4; 片側2車線 98/120 routes done (mc12 106, mc11 109).
# 0. evaluate.py's standoff(): the cause "standoff" where the ego stands inside a junction in the corridor of a car
#    that waits for it. The blocked routes of mc13, mc12 and mc11 re-run with it (tools/standoff_recount.py, into
#    runs/eval/standoff/), KeiSim 7 and KeiView 3 at a time
# 1. DAgger with mc13 driving, KeiSim and KeiView at once. The focus on yields and on mc13's target speed more than
#    1 m/s off the expert's (the late stops, and the cars it drove on to while the expert stopped: 2 of its 片側2車線
#    collisions); --freeze_takeover as in round 10:
#    a. 片側2車線: suburban towns, normal and dense, and the ramps towns
#    b. big junctions (people on the zebras and --npc_past_line 2 as in round 12)
#    c. two-phase grid towns, dense
#    d. Sapporo's neighbours as round 7's yield set (priority crossings, 止まれ; --ego_patience 3): Sapporo's 2 timing
#       standoffs and its collisions after a 止まれ
#    e. KeiView: suburban and big junction towns
# 2. mc14rt = mc13 trained on round 12's sets and these (x2), 4 x 200k samples, MMAP=1 (train.py --mmap)
# 3. evaluation, KeiSim and KeiView at once from the start. KeiSim: the 片側2車線 suites first (R13RT_LANES_DONE: the
#    first read), round 12's new suites (big junctions first), the curve courses, the grid suites, Sapporo _pat3 and
#    the release suites. KeiView: round 12's new suites and the town suites, then (KeiSim, 4 workers) the fresh
#    片側2車線 suites suburban2, ramps2 and ramp_routes2 (episodes no model was picked on) for mc14rt and mc13
# Every step skips what is already there. Start / restart: runs/mc_work/start_round13rt.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q "def standoff" scripts/evaluate.py && grep -q '"suburban2"' scripts/evaluate.py && [ -f tools/standoff_recount.py ] \
    || { echo "not the code with evaluate.py's standoff() and the fresh 片側2車線 suites (round 13)"; exit 1; }
[ "${MMAP:-1}" = 1 ] || { echo "round 13 needs MMAP=1 (train.py --mmap): its sets are ~29 GB"; exit 1; }
M13=runs/keipilot_mc13/last.pt
[ -f $M13 ] || { echo "$M13 (round 12) is missing"; exit 1; }
R="--rig wide_narrow"
L="--brake_hold --red_hold --yield_hold 4 1 3.5 4 0.5 0.5 10"
H=hold_red_yh_4_1_3.5_4_0.5_0.5_10
mkdir -p logs

# ---- 0. the blocked routes of mc13, mc12 and mc11 with standoff()
for r in keisim keiview; do
  for m in mc13 mc12 mc11; do
    uv run python tools/standoff_recount.py keipilot_${m}_$H --ckpt runs/keipilot_$m/last.pt --renderer $r || exit 1
  done > logs/r13rt_recount_$r.cmds
done
xargs -d '\n' -n 1 -P 7 bash -c 'eval "$1"' _ < logs/r13rt_recount_keisim.cmds &
xargs -d '\n' -n 1 -P 3 bash -c 'eval "$1"' _ < logs/r13rt_recount_keiview.cmds &
wait
for m in mc13 mc12 mc11; do echo "== $m"; uv run python tools/standoff_recount.py keipilot_${m}_$H --count; done
echo R13RT_RECOUNT_DONE

# ---- 1. data
col() {   # col NAME ARGS...: collect.py into data/NAME, once
  local n=$1; shift
  [ -f data/$n/.done ] && return
  uv run scripts/collect.py $R "$@" --out data/$n || return 1
  touch data/$n/.done
}
G="--mode dagger --ckpt $M13 --freeze_takeover 4 --freeze_reasons curve cruise crossing"
F="--focus_yield 0.15 --focus_reasons yield junction_blocked --focus_disagree 1.0"
SUB="--town_style suburban --jam_fixes 1 --npc_past_line 1"
DN="--vehicle_spacing 20 28 --ego_cross_rate 0.3"
RMP="--town_style ramps --jam_fixes 1 --npc_past_line 1 --via_ramp 0.3"
BJ="--town_style bigjunction --jam_fixes 1 --crosswalks 0.7 --crosswalk_spawn 3 --npc_past_line 2"
TPD="--town_style twophase --jam_fixes 1 --npc_past_line 1 $DN"
OSM="--osm_towns osm_sapporo_n osm_sapporo_e osm_sapporo_s --jam_fixes 1 $DN --ego_patience 3 --episodes_per_town 2"
OSM="$OSM --osm_seed0 6300"
FO="--focus_yield 0.15 --focus_reasons yield crossing stop_sign junction_blocked --focus_disagree 1.0"
ks_data() {
  col mc_rt_sub_dagger_mc13 $G $SUB $F --frames 20000 --workers 6 --seed 3301 || return 1
  col mc_rt_subd_dagger_mc13 $G $SUB $DN $F --frames 15000 --workers 6 --seed 3302 || return 1
  col mc_rt_ramp_dagger_mc13 $G $RMP $F --frames 10000 --workers 6 --seed 3303 || return 1
  col mc_rt_bj_dagger_mc13 $G $BJ $F --frames 12000 --workers 6 --seed 3304 || return 1
  col mc_rt_tp_dagger_mc13 $G $TPD $F --frames 8000 --workers 6 --seed 3305 || return 1
  col mc_rt_osm_dagger_mc13 $G $OSM $FO --frames 8000 --workers 6 --seed 3306 || return 1
}
kv_data() {
  col mc_rt_sub_kv_dagger_mc13 $G $SUB $F --renderer keiview --episodes_per_town 3 --frames 6000 --workers 3 \
      --seed 3307 || return 1
  col mc_rt_bj_kv_dagger_mc13 $G $BJ $F --renderer keiview --episodes_per_town 3 --frames 4000 --workers 3 \
      --seed 3308 || return 1
}
ks_data & P1=$!
kv_data & P2=$!
wait $P1 || exit 1
wait $P2 || exit 1
echo R13RT_COLLECT_DONE

# ---- 2. mc14rt
# round 12's training sets and weights (round12.sh; with MMAP=1 none is halved)
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
D="$D data/mc_rt_sub_dagger_mc13 data/mc_rt_subd_dagger_mc13 data/mc_rt_ramp_dagger_mc13 data/mc_rt_bj_dagger_mc13"
D="$D data/mc_rt_tp_dagger_mc13 data/mc_rt_osm_dagger_mc13 data/mc_rt_sub_kv_dagger_mc13 data/mc_rt_bj_kv_dagger_mc13"
W="$W data/mc_rt_sub_dagger_mc13=2 data/mc_rt_subd_dagger_mc13=2 data/mc_rt_ramp_dagger_mc13=2"
W="$W data/mc_rt_bj_dagger_mc13=2 data/mc_rt_tp_dagger_mc13=2 data/mc_rt_osm_dagger_mc13=2"
W="$W data/mc_rt_sub_kv_dagger_mc13=2 data/mc_rt_bj_kv_dagger_mc13=2"
echo "training data: $(du -sLc $D | tail -1 | awk '{printf "%.1f", $1 / 1048576}') GB"
[ -f runs/keipilot_mc14rt/last.pt ] || {
  ls -d $D >/dev/null || exit 1
  T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8"
  T="$T --kv_weight 2 --dagger_weight 2 --mmap"
  uv run scripts/train.py --data $D --init $M13 --out runs/keipilot_mc14rt --epochs 4 \
      --samples_per_epoch 200000 $T $W || exit 1; }
echo R13RT_TRAIN_MC14RT_DONE

# ---- 3. evaluation
M=runs/keipilot_mc14rt/last.pt
TAG=keipilot_mc14rt_$H
E="uv run scripts/evaluate.py --agent model --ckpt $M $L"
ev() { local name=$1; shift; [ -f runs/eval/${TAG}_$name.json ] || $E "$@"; }
curve() {   # curve NAME CKPT: round 11's curve courses
  local n=$1 C="uv run scripts/curve_test.py --agent model --ckpt $2 $L --tp ahead"
  for ep in 0 1 2 3; do $C --weather noon --episode $ep --variants weak weak_lines --tag ${n}_weak_ep$ep || return 1; done
  $C --weather noon --variants weak weak_lines --guardrail_light flat --tag ${n}_weak_flat || return 1
  for ep in 0 1 2 3; do $C --weather noon --episode $ep --variants nolines_guard nolines_guard_nobld lines_guard \
      --tag ${n}_curve_ep$ep || return 1; done
  $C --renderer keiview --variants weak --tag ${n}_weak_keiview || return 1
}
P=3                     # --ego_patience of Sapporo's evaluations: suffix _pat3
sap() {   # Sapporo with _pat$P: sap keisim|keiview
  local batches="100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500" rend="" opt="--workers 6"
  [ $1 = keiview ] && { rend=_keiview; opt="--renderer keiview --workers 3"; batches="100 200 300"; }
  for e in $batches; do
    local f=runs/eval/osm_sapporo/${TAG}$([ $e = 100 ] || echo _ep$e)${rend}_pat$P
    [ -f $f.json ] || $E --osm_town osm_sapporo --routes 20 --episode0 $e --ego_patience $P $opt --out $f.json
  done
}
lanes2() {   # lanes2 MODEL: the fresh 片側2車線 suites (KeiSim, 4 workers beside the KeiSim branch's 6)
  local t=keipilot_$1_$H e="uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_$1/last.pt $L"
  for s in suburban2 ramps2 ramp_routes2; do
    [ -f runs/eval/${t}_$s.json ] || $e --suite $s --workers 4
    [ $s = ramps2 ] || [ -f runs/eval/${t}_${s}_dense.json ] || $e --suite $s --dense --workers 4
  done
}
ks() {
  ev suburban --suite suburban --workers 6
  ev suburban_dense --suite suburban --dense --workers 6
  ev ramps --suite ramps --workers 6
  ev ramp_routes --suite ramp_routes --workers 6
  ev ramp_routes_dense --suite ramp_routes --dense --workers 6
  echo R13RT_LANES_DONE
  for s in bigjunction bigjunction_bk8 crosswalks crosswalks_sub crosswalks_bk8 crosswalks_sub_bk8; do
    ev $s --suite $s --workers 6
  done
  echo R13RT_NEW_KS_DONE
  [ -f runs/eval/curve/mc14rt_weak_keiview.json ] || curve mc14rt $M
  echo R13RT_CURVE_DONE
  for a in long3c long3d long3e long3f; do ev $a --suite $a --workers 6; ev ${a}_dense --suite $a --dense --workers 6; done
  echo R13RT_TP_KS_DONE
  sap keisim
  echo R13RT_SAPPORO_KS_DONE
  ev test --suite test --workers 6
  ev test2 --suite test2 --workers 6
  ev train --suite train --workers 6
  ev test_dense --suite test --dense --max_steps 6000 --workers 6
  ev long --suite long --workers 6
  ev long_dense --suite long --dense --workers 6
  ev long2 --suite long2 --workers 6
  ev long2_dense --suite long2 --dense --workers 6
  echo R13RT_GRID_KS_DONE
}
kv() {
  ev bigjunction_keiview --suite bigjunction --renderer keiview --workers 3
  ev crosswalks_keiview --suite crosswalks --renderer keiview --workers 3
  echo R13RT_NEW_KV_DONE
  ev suburban_keiview --suite suburban --renderer keiview --workers 3
  ev ramp_routes_keiview --suite ramp_routes --renderer keiview --workers 3
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
  sap keiview
  echo R13RT_KV_DONE
  lanes2 mc14rt
  lanes2 mc13
  echo R13RT_LANES2_DONE
}
kv &
ks &
wait
echo MC_ROUND13RT_DONE
