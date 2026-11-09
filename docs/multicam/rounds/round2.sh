#!/bin/bash
# Multi-camera KeiPilot, round 2: DAgger with mc2 driving, for what round 1 showed (runs/mc_work/HANDOVER.md):
#  - grid towns, 320 routes, mc2 worse than v0.10 (p = 0.02): pedestrians at ~36 km/h (braking 1.5-2 s late), lights
#    (3 misread, 3 starts from standstill on yellow / red), a standing car ahead, two-phase right turns
#  - Sapporo, 300 routes: yields to side traffic 21 -> 8 collisions, but new failures at two spots of the evaluation
#    box (a freeze in a curve; sharp left turns after a 止まれ leaving the road)
#  - KeiView: mc2 has never seen it (the rig renders there since 0d2671d)
# 1. KeiSim DAgger (220k frames, 6 workers) next to KeiView's grid sets (expert 50k + DAgger 20k, 3 workers); then
#    KeiView's Sapporo DAgger (20k) alone: its headless Chrome takes ~6 GB per worker in the real towns
# 2. mc3 = mc2 trained on round 1 (mc_expert and mc_tp_expert halved for RAM: ~20 GB in all) + round 2, 4 x 200k
# 3. evaluation: Sapporo 300 routes (KeiSim), then KeiView's Sapporo 60 routes alone, then the 16 grid suites (KeiSim)
#    next to KeiView's long3c and long3c dense
# Every step skips what is already there. Restart: runs/mc_work/start_round2.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
R="--rig wide_narrow"
G="--mode dagger --ckpt runs/keipilot_mc2/last.pt"
OSM="--osm_towns osm_sapporo_n osm_sapporo_e osm_sapporo_s --jam_fixes 1 --ego_cross_rate 0.3"
N="$OSM --vehicle_spacing 24 45 --focus_yield 0.3 --focus_reasons yield crossing stop_sign junction_blocked pedestrian"
Y="$OSM --vehicle_spacing 20 28 --focus_yield 0.15 --focus_reasons yield crossing stop_sign"
TP="--town_style twophase --jam_fixes 1"
PD="--ego_cross_rate 0.5 --focus_yield 0.3 --focus_reasons pedestrian"
FE="--vehicle_spacing 20 28 --ego_cross_rate 0.3 --focus_yield 0.15 --focus_lead_edge 12 --focus_reasons red_light"
TP3="--vehicle_spacing 24 45 --ego_cross_rate 0.3 --focus_yield 0.3 --focus_reasons yield junction_blocked red_light"
KV="--renderer keiview --episodes_per_town 3"

# ---- 1. data
keisim_sets() {
  uv run scripts/collect.py $R $G $PD --out data/mc_ped_dagger_mc2 --frames 25000 --workers 6 --seed 4201 || return 1
  uv run scripts/collect.py $R $G $PD --town_style varied --jam_fixes 1 --out data/mc_vped_dagger_mc2 --frames 20000 --workers 6 --seed 4202 || return 1
  uv run scripts/collect.py $R $G $FE --out data/mc_fl_dagger_mc2 --frames 30000 --workers 6 --seed 4203 || return 1
  uv run scripts/collect.py $R $G $FE $TP --out data/mc_fl_tp_dagger_mc2 --frames 30000 --workers 6 --seed 4204 || return 1
  uv run scripts/collect.py $R $G $TP3 $TP --out data/mc_tp3_dagger_mc2 --frames 40000 --workers 6 --seed 4205 || return 1
  uv run scripts/collect.py $R $G $N --episodes_per_town 2 --out data/mc_osm_dagger_mc2 --frames 50000 --workers 6 --seed 4206 || return 1
  uv run scripts/collect.py $R $G $Y --episodes_per_town 2 --out data/mc_osm_y_dagger_mc2 --frames 25000 --workers 6 --seed 4207 || return 1
}
keisim_sets > logs/mc_round2_keisim.log 2>&1 &
KS=$!
uv run scripts/collect.py $R $KV --out data/mc_kv_expert --frames 30000 --workers 3 --seed 4211 --ego_cross_rate 0.3 || exit 1
uv run scripts/collect.py $R $KV $TP --vehicle_spacing 24 45 --ego_cross_rate 0.2 --focus_yield 0.3 --out data/mc_tp_kv_expert \
    --frames 20000 --workers 3 --seed 4212 || exit 1
uv run scripts/collect.py $R $KV $G --vehicle_spacing 20 28 --ego_cross_rate 0.3 --focus_yield 0.3 \
    --focus_reasons pedestrian red_light yield junction_blocked --out data/mc_kv_dagger_mc2 --frames 20000 --workers 3 --seed 4213 || exit 1
# KeiView's Sapporo DAgger, 20k frames. Changed on 10-26 at 12:40 while this script ran (the bytes above untouched):
# the KeiSim sets keep ~1.2 frames/s per worker with their focus filters, a third of round 1's rate, so they run until
# the evening. One KeiView worker drives next to them while the RAM allows (its headless Chrome takes ~8 GB here),
# 2000 frames a step (collect.py resumes a worker); the rest with 3 workers in data/mc_osm_kv2_dagger_mc2 after KeiSim
n=2000
while kill -0 $KS 2>/dev/null && [ $n -le 16000 ] && [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 11 ]; do
  uv run scripts/collect.py $R $G $N --renderer keiview --episodes_per_town 5 --out data/mc_osm_kv_dagger_mc2 --frames $n --workers 1 --seed 4214 || exit 1
  n=$((n + 2000))
done
wait $KS || exit 1
echo MC_COLLECT_KEISIM_DONE
nb=$((20000 - 2000 * $(ls data/mc_osm_kv_dagger_mc2/*.npz 2>/dev/null | wc -l)))   # every shard holds 2000 frames
[ $nb -le 0 ] || uv run scripts/collect.py $R $G $N --renderer keiview --episodes_per_town 5 --out data/mc_osm_kv2_dagger_mc2 \
    --frames $nb --workers 3 --seed 4215 || exit 1
echo MC_COLLECT_R2_DONE

# ---- 2. mc3
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
half mc_expert
half mc_tp_expert
D1="data/mc_expert_half data/mc_ped data/mc_tp_expert_half data/mc_tp2_expert data/mc_v_expert data/mc_osm_expert data/mc_osm_y_expert"
D1="$D1 data/mc_dagger_mc1 data/mc_tp_dagger_mc1 data/mc_osm_dagger_mc1 data/mc_osm_y_dagger_mc1"
D2="data/mc_ped_dagger_mc2 data/mc_vped_dagger_mc2 data/mc_fl_dagger_mc2 data/mc_fl_tp_dagger_mc2 data/mc_tp3_dagger_mc2"
D2="$D2 data/mc_osm_dagger_mc2 data/mc_osm_y_dagger_mc2"
DK="data/mc_kv_expert data/mc_tp_kv_expert data/mc_kv_dagger_mc2"
for d in data/mc_osm_kv_dagger_mc2 data/mc_osm_kv2_dagger_mc2; do [ -d $d ] && DK="$DK $d"; done
T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
W="--dir_weight data/mc_ped=2 data/mc_tp2_expert=2 data/mc_osm_expert=1.5 data/mc_osm_y_expert=1.5 data/mc_osm_dagger_mc1=1.5 data/mc_osm_y_dagger_mc1=1.5"
W="$W data/mc_ped_dagger_mc2=2 data/mc_vped_dagger_mc2=2 data/mc_fl_dagger_mc2=2 data/mc_fl_tp_dagger_mc2=3 data/mc_tp3_dagger_mc2=4"
W="$W data/mc_osm_dagger_mc2=1.5 data/mc_osm_y_dagger_mc2=1.5 data/mc_tp_kv_expert=2 data/mc_kv_dagger_mc2=2 data/mc_osm_kv_dagger_mc2=1.5"
W="$W data/mc_osm_kv2_dagger_mc2=1.5"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%d", $1 / 1048576}'; }
echo "training data: $(gb $D1 $D2 $DK) GB"
if [ "$(gb $D1 $D2 $DK)" -gt 22 ]; then     # load_shards holds it all in RAM (32 GB, with the desktop apps)
  half mc_osm_expert
  half mc_v_expert
  D1="${D1/data\/mc_osm_expert /data/mc_osm_expert_half }"
  D1="${D1/data\/mc_v_expert /data/mc_v_expert_half }"
  W="${W/data\/mc_osm_expert=/data/mc_osm_expert_half=}"
  echo "training data with more halves: $(gb $D1 $D2 $DK) GB"
fi
[ -f runs/keipilot_mc3/last.pt ] || uv run scripts/train.py --data $D1 $D2 $DK --init runs/keipilot_mc2/last.pt --out runs/keipilot_mc3 \
    --epochs 4 --samples_per_epoch 200000 $T $W || exit 1
echo MC_TRAIN_MC3_DONE

# ---- 3. evaluation
M=runs/keipilot_mc3/last.pt
S="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold --osm_town osm_sapporo --routes 20"
for e in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500; do
  f=runs/eval/osm_sapporo/keipilot_mc3_hold_red$([ $e = 100 ] || echo _ep$e)
  [ -f $f.json ] || $S --episode0 $e --workers 6 --out $f.json
done
echo MC_SAPPORO_MC3_DONE
for e in 100 200 300; do
  f=runs/eval/osm_sapporo/keipilot_mc3_hold_red$([ $e = 100 ] || echo _ep$e)_keiview
  [ -f $f.json ] || $S --episode0 $e --renderer keiview --workers 3 --out $f.json
done
echo MC_SAPPORO_KV_MC3_DONE
E="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold"
ev() { local name=$1; shift; [ -f runs/eval/keipilot_mc3_hold_red_$name.json ] || $E "$@"; }
ks() {
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
kv() {
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
}
ks &
kv &
wait
echo MC_ROUND2_DONE
