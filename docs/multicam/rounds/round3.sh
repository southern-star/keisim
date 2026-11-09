#!/bin/bash
# Multi-camera KeiPilot, round 3: DAgger with mc3 driving, for what round 2 showed (runs/mc_work/HANDOVER.md):
#  - KeiView, Sapporo (60 routes): mc3 DS 0.910 vs v0.10 0.977. In 4 of its 5 freezes it stops in the middle of a left
#    turn and never starts again. The stop depends on the place under the episode's light (ep300: 3 of 3 with its own
#    light, none with two others)
#  - pedestrians: braking ~1 s late (3 collisions on the 16 grid suites); starting out while the expert still gives way
#    (3 Sapporo collisions out of the wait position after a 止まれ, 2 on the grid)
# New: collect.py --freeze_takeover 4. Once the model has asked to stand for 4 s while the expert drives on (curve /
# cruise), the expert drives for 3-5 s. In round 2's KeiView Sapporo DAgger, 24.5 % of the frames were such a stand
# (39 freezes, ~18 s of the same picture each, up to blocked_timeout's 40 s); now a freeze adds ~20 frames and the
# episode goes on.
# 1. KeiSim grid sets (pedestrians, two-phase right turns; 6 workers) with one KeiView Sapporo worker next to them
#    while the RAM allows (~7.6 GB each); then KeiSim's Sapporo sets (6 workers, ~4.2 GB each) alone; then the rest of
#    KeiView's Sapporo DAgger and KeiView's grid DAgger (3 workers)
# 2. mc4 = mc3 trained on rounds 1-3, 4 x 200k samples. For the RAM (~20 GB of data): round 1's mc1 DAgger of the grid
#    towns is left out, the Sapporo mc1 DAgger and three expert sets are halved
# 3. evaluation as round 2: Sapporo 300 routes (KeiSim), KeiView's Sapporo 60, the 16 grid suites, KeiView's long3c.
#    Done files are skipped, so results made elsewhere (a cloud session) and copied into runs/eval count as done
# Every step skips what is already there. Start / restart: runs/mc_work/start_round3.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
R="--rig wide_narrow"
G="--mode dagger --ckpt runs/keipilot_mc3/last.pt --freeze_takeover 4"
OSM="--osm_towns osm_sapporo_n osm_sapporo_e osm_sapporo_s --jam_fixes 1 --ego_cross_rate 0.3"
N="$OSM --vehicle_spacing 24 45 --focus_yield 0.3 --focus_reasons yield crossing stop_sign junction_blocked pedestrian"
Y="$OSM --vehicle_spacing 20 28 --focus_yield 0.15 --focus_reasons yield crossing stop_sign"
TP="--town_style twophase --jam_fixes 1"
PD="--ego_cross_rate 0.5 --focus_yield 0.3 --focus_reasons pedestrian"
TP3="--vehicle_spacing 24 45 --ego_cross_rate 0.3 --focus_yield 0.3 --focus_reasons yield junction_blocked red_light"
KV="--renderer keiview --episodes_per_town 3"
KVOSM=30000                     # KeiView's Sapporo DAgger frames in all (the two folders below)

# ---- 1. data
keisim_grid() {
  uv run scripts/collect.py $R $G $PD --out data/mc_ped_dagger_mc3 --frames 25000 --workers 6 --seed 4301 || return 1
  uv run scripts/collect.py $R $G $PD --town_style varied --jam_fixes 1 --out data/mc_vped_dagger_mc3 --frames 15000 --workers 6 --seed 4302 || return 1
  uv run scripts/collect.py $R $G $TP3 $TP --out data/mc_tp3_dagger_mc3 --frames 30000 --workers 6 --seed 4303 || return 1
}
keisim_grid > logs/mc_round3_keisim.log 2>&1 &
KS=$!
# one KeiView Sapporo worker next to the grid sets, 2000 frames a step (collect.py resumes a worker), while the RAM allows
n=2000
while kill -0 $KS 2>/dev/null && [ $n -le 20000 ] && [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 11 ]; do
  uv run scripts/collect.py $R $G $N --renderer keiview --episodes_per_town 5 --out data/mc_osm_kv_dagger_mc3 --frames $n --workers 1 --seed 4311 || exit 1
  n=$((n + 2000))
done
wait $KS || exit 1
echo MC_COLLECT_GRID_DONE
uv run scripts/collect.py $R $G $N --episodes_per_town 2 --out data/mc_osm_dagger_mc3 --frames 40000 --workers 6 --seed 4304 || exit 1
uv run scripts/collect.py $R $G $Y --episodes_per_town 2 --out data/mc_osm_y_dagger_mc3 --frames 25000 --workers 6 --seed 4305 || exit 1
echo MC_COLLECT_KEISIM_DONE
nb=$((KVOSM - 2000 * $(ls data/mc_osm_kv_dagger_mc3/*.npz 2>/dev/null | wc -l)))   # every shard holds 2000 frames
[ $nb -le 0 ] || uv run scripts/collect.py $R $G $N --renderer keiview --episodes_per_town 5 --out data/mc_osm_kv2_dagger_mc3 \
    --frames $nb --workers 3 --seed 4312 || exit 1
uv run scripts/collect.py $R $KV $G --vehicle_spacing 20 28 --ego_cross_rate 0.3 --focus_yield 0.3 \
    --focus_reasons pedestrian red_light yield junction_blocked --out data/mc_kv_dagger_mc3 --frames 10000 --workers 3 --seed 4313 || exit 1
echo MC_COLLECT_R3_DONE

# ---- 2. mc4
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
for d in mc_expert mc_tp_expert mc_tp2_expert mc_osm_expert mc_v_expert mc_osm_dagger_mc1 mc_osm_y_dagger_mc1; do half $d; done
D1="data/mc_expert_half data/mc_ped data/mc_tp_expert_half data/mc_tp2_expert_half data/mc_v_expert_half data/mc_osm_expert_half"
D1="$D1 data/mc_osm_y_expert data/mc_osm_dagger_mc1_half data/mc_osm_y_dagger_mc1_half"
D2="data/mc_ped_dagger_mc2 data/mc_vped_dagger_mc2 data/mc_fl_dagger_mc2 data/mc_fl_tp_dagger_mc2 data/mc_tp3_dagger_mc2"
D2="$D2 data/mc_osm_dagger_mc2 data/mc_osm_y_dagger_mc2"
D3="data/mc_ped_dagger_mc3 data/mc_vped_dagger_mc3 data/mc_tp3_dagger_mc3 data/mc_osm_dagger_mc3 data/mc_osm_y_dagger_mc3"
DK="data/mc_kv_expert data/mc_tp_kv_expert data/mc_kv_dagger_mc2 data/mc_osm_kv_dagger_mc2 data/mc_osm_kv2_dagger_mc2"
DK="$DK data/mc_kv_dagger_mc3 data/mc_osm_kv_dagger_mc3"
[ -d data/mc_osm_kv2_dagger_mc3 ] && DK="$DK data/mc_osm_kv2_dagger_mc3"
T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
W="--dir_weight data/mc_ped=2 data/mc_tp2_expert_half=2 data/mc_osm_expert_half=1.5 data/mc_osm_y_expert=1.5"
W="$W data/mc_osm_dagger_mc1_half=1.5 data/mc_osm_y_dagger_mc1_half=1.5"
W="$W data/mc_ped_dagger_mc2=2 data/mc_vped_dagger_mc2=2 data/mc_fl_dagger_mc2=2 data/mc_fl_tp_dagger_mc2=3 data/mc_tp3_dagger_mc2=3"
W="$W data/mc_osm_dagger_mc2=1.5 data/mc_osm_y_dagger_mc2=1.5 data/mc_tp_kv_expert=2 data/mc_kv_dagger_mc2=2 data/mc_osm_kv_dagger_mc2=1.5"
W="$W data/mc_osm_kv2_dagger_mc2=1.5"
W="$W data/mc_ped_dagger_mc3=2 data/mc_vped_dagger_mc3=2 data/mc_tp3_dagger_mc3=3 data/mc_osm_dagger_mc3=1.5 data/mc_osm_y_dagger_mc3=1.5"
W="$W data/mc_kv_dagger_mc3=2 data/mc_osm_kv_dagger_mc3=2 data/mc_osm_kv2_dagger_mc3=2"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D1 $D2 $D3 $DK) GB"
if [ "$(gb $D1 $D2 $D3 $DK | cut -d. -f1)" -ge 21 ]; then   # load_shards holds it all in RAM (32 GB, with the desktop apps)
  half mc_ped
  D1="${D1/data\/mc_ped /data/mc_ped_half }"
  W="${W/data\/mc_ped=/data/mc_ped_half=}"
  echo "training data with mc_ped halved: $(gb $D1 $D2 $D3 $DK) GB"
fi
[ -f runs/keipilot_mc4/last.pt ] || uv run scripts/train.py --data $D1 $D2 $D3 $DK --init runs/keipilot_mc3/last.pt --out runs/keipilot_mc4 \
    --epochs 4 --samples_per_epoch 200000 $T $W || exit 1
echo MC_TRAIN_MC4_DONE

# ---- 3. evaluation
M=runs/keipilot_mc4/last.pt
S="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold --osm_town osm_sapporo --routes 20"
for e in 100 200 300; do      # KeiView's Sapporo first: the point of this round
  f=runs/eval/osm_sapporo/keipilot_mc4_hold_red$([ $e = 100 ] || echo _ep$e)_keiview
  [ -f $f.json ] || $S --episode0 $e --renderer keiview --workers 3 --out $f.json
done
echo MC_SAPPORO_KV_MC4_DONE
for e in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500; do
  f=runs/eval/osm_sapporo/keipilot_mc4_hold_red$([ $e = 100 ] || echo _ep$e)
  [ -f $f.json ] || $S --episode0 $e --workers 6 --out $f.json
done
echo MC_SAPPORO_MC4_DONE
E="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold"
ev() { local name=$1; shift; [ -f runs/eval/keipilot_mc4_hold_red_$name.json ] || $E "$@"; }
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
echo MC_ROUND3_DONE
