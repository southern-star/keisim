#!/bin/bash
# Multi-camera KeiPilot, round 7 (runs/mc_work/HANDOVER.md, "7 回目"), from mc7 (round 6). Round 6's --stand_pert removed
# KeiView's phantom stands (Sapporo: none stood still, needless standing 227 -> 62 s), but mc7 now sets off from a stand
# on weak cues where mc5b and mc6, given the very same inputs, keep standing: a small side car in KeiView (Sapporo
# ep116, 0.9 s before the expert), a far light turning green with a car standing 5 m ahead (grid long3c town 1014,
# a rear-end collision after a 74 s standoff). New:
#  - DAgger with mc7 driving: its early starts are frames where its target is far off the expert's (in focus), so
#    they get the expert's "stand" (the usual round; nothing new in collect.py)
#  - a grid DAgger set as well (KeiSim, two-phase towns, dense; the last grid DAgger was round 3's), as the standoff
#    start happened in a grid town
#  - --stand_pert 0.2 -> 0.1 (round 6's stand frames stay in the mix)
# 1. KeiView 5 x 9000 frames on new Sapporo variants (--osm_seed0 5000+), then KeiSim's N and Y sets, then the grid set
# 2. mc8 = mc7 trained on rounds 1-7, 4 x 200k samples. For the RAM: round 5's Sapporo sets halved, round 4's at a
#    quarter, round 2's two-phase grid DAgger halved; round 2's following sets halved too if still 21 GB or more.
#    Round 7's sets weigh KeiSim x2 / KeiView x3 / grid x2, round 6's x1.5 / x2
# 3. evaluation as round 6: mc8 on Sapporo with _pat3 (KeiView 60 and KeiSim 300 at once), then KeiView's long3c
#    beside the 16 grid suites (no patience)
# Every step skips what is already there. Start / restart: runs/mc_work/start_round7.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q "class StandPert" scripts/collect.py || { echo "collect.py has no --stand_pert"; exit 1; }
R="--rig wide_narrow"
G="--mode dagger --ckpt runs/keipilot_mc7/last.pt --freeze_takeover 4 --freeze_reasons curve cruise crossing --ego_patience 3"
GT="${G/ --ego_patience 3/}"      # the grid set: no patience, as the grid suites (their standoffs are the case)
G="$G --stand_pert 0.1 --stand_hold 1 4"
OSM="--osm_towns osm_sapporo_n osm_sapporo_e osm_sapporo_s --jam_fixes 1 --ego_cross_rate 0.3"
N="$OSM --vehicle_spacing 24 45 --focus_yield 0.3 --focus_reasons yield crossing stop_sign junction_blocked pedestrian"
Y="$OSM --vehicle_spacing 20 28 --focus_yield 0.15 --focus_reasons yield crossing stop_sign"
KVN="$OSM --vehicle_spacing 20 30 --focus_yield 0.3 --focus_reasons yield crossing stop_sign pedestrian"
TPD="--town_style twophase --jam_fixes 1 --vehicle_spacing 20 28 --ego_cross_rate 0.3 --focus_yield 0.3"
TPD="$TPD --focus_reasons yield junction_blocked red_light"     # the grid set: round 3's tp3, dense
P=3                     # --ego_patience of Sapporo's evaluations: suffix _pat3

sap() {   # Sapporo with _pat$P: sap CKPT keisim|keiview (tag from the checkpoint's folder, as evaluate.py's)
  local m=$1 tag=$(basename $(dirname $1)) batches="100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500"
  local rend="" opt="--workers 6"
  [ $2 = keiview ] && { rend=_keiview; opt="--renderer keiview --workers 3"; batches="100 200 300"; }
  for e in $batches; do
    local f=runs/eval/osm_sapporo/${tag}_hold_red$([ $e = 100 ] || echo _ep$e)${rend}_pat$P
    [ -f $f.json ] || uv run scripts/evaluate.py --agent model --ckpt $m --brake_hold --red_hold --osm_town osm_sapporo \
        --routes 20 --episode0 $e --ego_patience $P $opt --out $f.json
  done
}

# ---- 1. data
kv_data() {
  for i in 0 1 2 3 4; do    # collect.py resumes its workers: every chunk adds 3000 frames each, on its own variants
    uv run scripts/collect.py $R $G $KVN --renderer keiview --episodes_per_town 5 --osm_seed0 $((5000 + 10 * i)) \
        --out data/mc_osm_kv_dagger_mc7 --frames $((9000 * (i + 1))) --workers 3 --seed 4711 || return 1
  done
}
kv_data || exit 1
echo MC_COLLECT_KV_R7_DONE
uv run scripts/collect.py $R $G $N --episodes_per_town 2 --osm_seed0 5000 --out data/mc_osm_dagger_mc7 --frames 40000 \
    --workers 6 --seed 4701 || exit 1
uv run scripts/collect.py $R $G $Y --episodes_per_town 2 --osm_seed0 5000 --out data/mc_osm_y_dagger_mc7 --frames 25000 \
    --workers 6 --seed 4702 || exit 1
uv run scripts/collect.py $R $GT $TPD --out data/mc_tp_dagger_mc7 --frames 30000 --workers 6 --seed 4703 || exit 1
echo MC_COLLECT_R7_DONE

# ---- 2. mc8
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
for d in mc_osm_dagger_mc4 mc_osm_y_dagger_mc4 mc_osm_kv_dagger_mc4 mc_osm_dagger_mc3_half mc_osm_y_dagger_mc3_half \
         mc_kv_dagger_mc2 mc_expert_half mc_osm_dagger_mc4_half mc_osm_y_dagger_mc4_half mc_osm_kv_dagger_mc4_half \
         mc_osm_dagger_mc5b mc_osm_y_dagger_mc5b mc_osm_kv_dagger_mc5b mc_tp3_dagger_mc2; do half $d; done
D1="data/mc_expert_half_half data/mc_ped_half data/mc_tp_expert_half_half data/mc_tp2_expert_half data/mc_v_expert_half"
D1="$D1 data/mc_osm_expert_half data/mc_osm_y_expert"
D2="data/mc_ped_dagger_mc2 data/mc_vped_dagger_mc2 data/mc_fl_dagger_mc2 data/mc_fl_tp_dagger_mc2 data/mc_tp3_dagger_mc2_half"
D3="data/mc_ped_dagger_mc3 data/mc_vped_dagger_mc3 data/mc_tp3_dagger_mc3 data/mc_osm_dagger_mc3_half_half data/mc_osm_y_dagger_mc3_half_half"
D4="data/mc_osm_dagger_mc4_half_half data/mc_osm_y_dagger_mc4_half_half"
D5="data/mc_osm_dagger_mc5b_half data/mc_osm_y_dagger_mc5b_half"
D6="data/mc_osm_dagger_mc6 data/mc_osm_y_dagger_mc6"
D7="data/mc_osm_dagger_mc7 data/mc_osm_y_dagger_mc7 data/mc_tp_dagger_mc7"
DK="data/mc_kv_expert data/mc_tp_kv_expert data/mc_kv_dagger_mc2_half data/mc_osm_kv_dagger_mc2_half data/mc_osm_kv2_dagger_mc2_half"
DK="$DK data/mc_kv_dagger_mc3 data/mc_osm_kv_dagger_mc3_half data/mc_osm_kv2_dagger_mc3_half data/mc_osm_kv_dagger_mc4_half_half"
DK="$DK data/mc_osm_kv_dagger_mc5b_half data/mc_osm_kv_dagger_mc6 data/mc_osm_kv_dagger_mc7"
T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
W="--dir_weight data/mc_ped_half=2 data/mc_tp2_expert_half=2 data/mc_osm_expert_half=1.5 data/mc_osm_y_expert=1.5"
W="$W data/mc_ped_dagger_mc2=2 data/mc_vped_dagger_mc2=2 data/mc_fl_dagger_mc2=2 data/mc_fl_tp_dagger_mc2=3 data/mc_tp3_dagger_mc2_half=3"
W="$W data/mc_tp_kv_expert=2 data/mc_kv_dagger_mc2_half=2 data/mc_osm_kv_dagger_mc2_half=1.5 data/mc_osm_kv2_dagger_mc2_half=1.5"
W="$W data/mc_ped_dagger_mc3=2 data/mc_vped_dagger_mc3=2 data/mc_tp3_dagger_mc3=3"
W="$W data/mc_osm_dagger_mc3_half_half=1.5 data/mc_osm_y_dagger_mc3_half_half=1.5"
W="$W data/mc_kv_dagger_mc3=2 data/mc_osm_kv_dagger_mc3_half=2 data/mc_osm_kv2_dagger_mc3_half=2"
W="$W data/mc_osm_dagger_mc4_half_half=1.5 data/mc_osm_y_dagger_mc4_half_half=1.5 data/mc_osm_kv_dagger_mc4_half_half=2"
W="$W data/mc_osm_dagger_mc5b_half=1.5 data/mc_osm_y_dagger_mc5b_half=1.5 data/mc_osm_kv_dagger_mc5b_half=2"
W="$W data/mc_osm_dagger_mc6=1.5 data/mc_osm_y_dagger_mc6=1.5 data/mc_osm_kv_dagger_mc6=2"
W="$W data/mc_osm_dagger_mc7=2 data/mc_osm_y_dagger_mc7=2 data/mc_tp_dagger_mc7=2 data/mc_osm_kv_dagger_mc7=3"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D1 $D2 $D3 $D4 $D5 $D6 $D7 $DK) GB"
if [ "$(gb $D1 $D2 $D3 $D4 $D5 $D6 $D7 $DK | cut -d. -f1)" -ge 21 ]; then   # load_shards holds it all in RAM (32 GB)
  half mc_fl_dagger_mc2
  half mc_fl_tp_dagger_mc2
  D2="${D2/data\/mc_fl_dagger_mc2 /data/mc_fl_dagger_mc2_half }"
  D2="${D2/data\/mc_fl_tp_dagger_mc2 /data/mc_fl_tp_dagger_mc2_half }"
  W="${W/data\/mc_fl_dagger_mc2=/data/mc_fl_dagger_mc2_half=}"
  W="${W/data\/mc_fl_tp_dagger_mc2=/data/mc_fl_tp_dagger_mc2_half=}"
  echo "training data with round 2's following sets halved: $(gb $D1 $D2 $D3 $D4 $D5 $D6 $D7 $DK) GB"
fi
[ -f runs/keipilot_mc8/last.pt ] || uv run scripts/train.py --data $D1 $D2 $D3 $D4 $D5 $D6 $D7 $DK --init runs/keipilot_mc7/last.pt \
    --out runs/keipilot_mc8 --epochs 4 --samples_per_epoch 200000 $T $W || exit 1
echo MC_TRAIN_MC8_DONE

# ---- 3. evaluation
M=runs/keipilot_mc8/last.pt
E="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold"
ev() { local name=$1; shift; [ -f runs/eval/keipilot_mc8_hold_red_$name.json ] || $E "$@"; }
kv() {
  sap $M keiview
  echo MC_SAPPORO_KV_MC8_DONE
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
}
ks() {
  sap $M keisim
  echo MC_SAPPORO_MC8_DONE
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
echo MC_ROUND7_DONE
