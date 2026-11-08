#!/bin/bash
# Multi-camera KeiPilot, round 10 (runs/mc_work/HANDOVER.md, "10 回目"). mc10 (round 9) on the 16 grid suites: 9
# vehicle collisions (mc8 3, p 0.057), none from behind. Replays: 5 = a right turn rolling through its wait point into
# the oncoming left turn that merges into the same exit, where the expert's late check only looked at the first
# crossing and said "go"; 3 = setting off at green before the box was clear (a car of the other phase still turning
# out of it): the expert said "wait" at the line, but from 0.4 m before it at 4.5 km/h, where it could no longer stop
# there, "go"; 1 = an NPC that had stopped 1.1 m over its line at green drove in at red.
# 0. label version 7 (keisim/expert.py): at signals every crossing still ahead counts past the wait point; a car slower
#    than BOX_V that cannot stop at the line for a full exit or a busy box stops short of the junction's entry (old
#    shards get the part before the line through target_for_speed); TrafficConfig.light_past_line (collect.py
#    --npc_past_line 1, collection only: the evaluations keep the old NPCs, so they compare with mc8 / mc10)
# 1. DAgger with mc10 driving, towns 0-399: KeiView in two-phase grid towns (dense), then KeiSim two-phase dense and
#    normal and the 片側2車線 suburban towns (their signalised right turns: half of mc10's jams there), the focus on
#    yields, the box and red lights
# 2. mc11 = mc10 trained on rounds 1-10, 4 x 200k samples, round 10's sets x3 / x3 / x2 / x2; old sets halved in turn
#    until under 21.3 GB (round 9's order first, as for mc10)
# 3. evaluation with mc8's standard layers: the two-phase grid suites first (both renderers), then the 片側2車線
#    suites, Sapporo with _pat3 and the rest of the grid suites
# Every step skips what is already there. Start / restart: runs/mc_work/start_round10.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q "LABEL_VERSION = 7" keisim/expert.py || { echo "keisim/expert.py is not label version 7"; exit 1; }
grep -q "npc_past_line" scripts/collect.py || { echo "collect.py has no --npc_past_line"; exit 1; }
R="--rig wide_narrow"
G="--mode dagger --ckpt runs/keipilot_mc10/last.pt --freeze_takeover 4 --freeze_reasons curve cruise crossing --npc_past_line 1"
TP="--town_style twophase --jam_fixes 1 --ego_cross_rate 0.3 --focus_yield 0.3 --focus_reasons yield junction_blocked red_light"

# ---- 1. data
[ -f data/mc_tp_kv_dagger_mc10/.done ] || {
  uv run scripts/collect.py $R $G $TP --vehicle_spacing 20 28 --renderer keiview --episodes_per_town 3 \
      --out data/mc_tp_kv_dagger_mc10 --frames 30000 --workers 3 --seed 5001 || exit 1
  touch data/mc_tp_kv_dagger_mc10/.done; }
echo MC_COLLECT_KV_R10_DONE
[ -f data/mc_tp_dagger_mc10/.done ] || {
  uv run scripts/collect.py $R $G $TP --vehicle_spacing 20 28 --out data/mc_tp_dagger_mc10 --frames 40000 --workers 6 \
      --seed 5002 || exit 1
  touch data/mc_tp_dagger_mc10/.done; }
[ -f data/mc_tp3_dagger_mc10/.done ] || {
  uv run scripts/collect.py $R $G $TP --vehicle_spacing 24 45 --out data/mc_tp3_dagger_mc10 --frames 30000 --workers 6 \
      --seed 5003 || exit 1
  touch data/mc_tp3_dagger_mc10/.done; }
[ -f data/mc_sub_dagger_mc10/.done ] || {
  uv run scripts/collect.py $R $G --town_style suburban --jam_fixes 1 --focus_yield 0.4 \
      --focus_reasons yield junction_blocked red_light lane_change pedestrian --out data/mc_sub_dagger_mc10 \
      --frames 30000 --workers 6 --seed 5004 || exit 1
  touch data/mc_sub_dagger_mc10/.done; }
echo MC_COLLECT_R10_DONE

# ---- 2. mc11
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
# mc10's sets and weights (round9.sh before its RAM step)
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
D="$D data/mc_ramp_kv_dagger_mc9 data/mc_ramp_dagger_mc9 data/mc_ramp_d_dagger_mc9"
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
W="$W data/mc_ramp_kv_dagger_mc9=3 data/mc_ramp_dagger_mc9=3 data/mc_ramp_d_dagger_mc9=2"
# round 10's sets
D="$D data/mc_tp_kv_dagger_mc10 data/mc_tp_dagger_mc10 data/mc_tp3_dagger_mc10 data/mc_sub_dagger_mc10"
W="$W data/mc_tp_kv_dagger_mc10=3 data/mc_tp_dagger_mc10=3 data/mc_tp3_dagger_mc10=2 data/mc_sub_dagger_mc10=2"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D) GB"
# the RAM: halve until under 21.3 GB. Round 9's order first (mc10's own halved sets), then the older two-phase and
# Sapporo DAgger sets (round 10's and rounds 6-7's newer ones cover the same), round 3's KeiView sets, round 9's
# KeiView ramps set; then Sapporo's give-way expert set (the DAgger sets of rounds 3-7 cover it, still 21.6 GB without
# it) and round 7's KeiSim Sapporo set
for s in mc_sub_d_dagger_mc8 mc_ped_dagger_mc2 mc_vped_dagger_mc2 mc_tp3_dagger_mc2_half mc_kv_dagger_mc2_half \
         mc_osm_kv_dagger_mc6_half mc_sub_dagger_mc8_f mc_sub_kv_dagger_mc8_f mc_ped_dagger_mc3 mc_tp3_dagger_mc3 \
         mc_fl_tp_dagger_mc2_half mc_tp_dagger_mc7 mc_osm_y_dagger_mc6 mc_osm_dagger_mc6 mc_kv_dagger_mc3 \
         mc_osm_kv2_dagger_mc3_half mc_osm_kv_dagger_mc3_half mc_ramp_kv_dagger_mc9 mc_vped_dagger_mc3 \
         mc_osm_y_expert mc_osm_dagger_mc7; do
  awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.3) }' || break
  half $s
  D=" $D "; D="${D/ data\/$s / data/${s}_half }"; D="${D# }"; D="${D% }"
  W="${W/data\/$s=/data/${s}_half=}"
  echo "training data with $s halved: $(gb $D) GB"
done
awk -v g="$(gb $D)" 'BEGIN { exit !(g >= 21.5) }' && { echo "training data still $(gb $D) GB"; exit 1; }
[ -f runs/keipilot_mc11/last.pt ] || {
  ls -d $D >/dev/null || exit 1
  T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
  uv run scripts/train.py --data $D --init runs/keipilot_mc10/last.pt --out runs/keipilot_mc11 --epochs 4 \
      --samples_per_epoch 200000 $T $W || exit 1; }
echo MC_TRAIN_MC11_DONE

# ---- 3. evaluation (the old NPCs, as mc8's and mc10's)
M=runs/keipilot_mc11/last.pt
TAG=keipilot_mc11_hold_red_yh_4_1_3.5_4_0.5_0.5_10
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
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
  echo R10_TP_KV_DONE
  ev ramp_routes_keiview --suite ramp_routes --renderer keiview --workers 3
  ev suburban_keiview --suite suburban --renderer keiview --workers 3
  echo R10_NEW_KV_DONE
  sap keiview
  echo R10_SAPPORO_KV_DONE
}
ks() {
  for a in long3c long3d long3e long3f; do ev $a --suite $a --workers 6; ev ${a}_dense --suite $a --dense --workers 6; done
  echo R10_TP_KS_DONE
  ev ramp_routes --suite ramp_routes --workers 6
  ev ramps --suite ramps --workers 6
  ev suburban --suite suburban --workers 6
  ev ramp_routes_dense --suite ramp_routes --dense --workers 6
  ev suburban_dense --suite suburban --dense --workers 6
  echo R10_NEW_KS_DONE
  sap keisim
  echo R10_SAPPORO_KS_DONE
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
echo MC_ROUND10_DONE
