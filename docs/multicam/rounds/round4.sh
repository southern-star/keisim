#!/bin/bash
# Multi-camera KeiPilot, round 4: what the replays of mc4's KeiView pull-outs showed (runs/mc_work/HANDOVER.md):
#  - in KeiView the cars with priority coming from the sides are seen late or not at all (yield_tta about 8 s with the
#    car in view; on new episodes KeiView missed more than KeiSim in 13 events, KeiSim more in 5, in the training boxes
#    too): the training frames are remembered, not generalised. Two of the four pull-outs: the car hidden behind
#    KeiView's corner walls and hedges at the stop line
#  - the expert sped up to 16-21 km/h between its 止まれ stop and the wait point
# New: label version 6 (keisim/expert.py): 徐行 (CROSSING_V, 10 km/h) from the 止まれ line to the wait point. Older
# Sapporo frames there are dropped in training (keipilot/data.py stale_creep: 11-14 % of round 3's Sapporo DAgger)
# 1. DAgger with mc4 driving, in Sapporo's training boxes only (the grid towns have no 止まれ lines: their labels did
#    not change), on new town variants (--osm_seed0 2000+; KeiView's houses, walls, hedges and trees follow the seed):
#    KeiSim (6 workers): the N and Y sets of round 3. KeiView (3 workers): 5 chunks of 9000 frames, every chunk on 9
#    new variants, denser traffic on the priority roads (more cars coming from the sides)
# 2. mc5 = mc4 trained on rounds 1-4, 4 x 200k samples, the new KeiView set x3. For the RAM: round 1's Sapporo DAgger
#    left out, mc_ped and round 2's Sapporo DAgger halved
# 3. evaluation as round 3, KeiView's Sapporo first, then tools/kv_tta_survey.py on the evaluation box (episodes
#    3000-3009, as for mc4: runs/mc_work/survey)
# Every step skips what is already there. Start / restart: runs/mc_work/start_round4.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
R="--rig wide_narrow"
G="--mode dagger --ckpt runs/keipilot_mc4/last.pt --freeze_takeover 4"
OSM="--osm_towns osm_sapporo_n osm_sapporo_e osm_sapporo_s --jam_fixes 1 --ego_cross_rate 0.3"
N="$OSM --vehicle_spacing 24 45 --focus_yield 0.3 --focus_reasons yield crossing stop_sign junction_blocked pedestrian"
Y="$OSM --vehicle_spacing 20 28 --focus_yield 0.15 --focus_reasons yield crossing stop_sign"
KVN="$OSM --vehicle_spacing 20 30 --focus_yield 0.3 --focus_reasons yield crossing stop_sign pedestrian"
grep -q "^LABEL_VERSION = 6" keisim/expert.py || { echo "keisim/expert.py is not label version 6"; exit 1; }

# ---- 1. data
uv run scripts/collect.py $R $G $N --episodes_per_town 2 --osm_seed0 2000 --out data/mc_osm_dagger_mc4 --frames 40000 \
    --workers 6 --seed 4401 || exit 1
uv run scripts/collect.py $R $G $Y --episodes_per_town 2 --osm_seed0 2000 --out data/mc_osm_y_dagger_mc4 --frames 25000 \
    --workers 6 --seed 4402 || exit 1
echo MC_COLLECT_KEISIM_DONE
for i in 0 1 2 3 4; do      # collect.py resumes its workers: every chunk adds 3000 frames each, on its own variants
  uv run scripts/collect.py $R $G $KVN --renderer keiview --episodes_per_town 5 --osm_seed0 $((2000 + 10 * i)) \
      --out data/mc_osm_kv_dagger_mc4 --frames $((9000 * (i + 1))) --workers 3 --seed 4411 || exit 1
done
echo MC_COLLECT_R4_DONE

# ---- 2. mc5
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
for d in mc_ped mc_osm_dagger_mc2 mc_osm_y_dagger_mc2; do half $d; done
D1="data/mc_expert_half data/mc_ped_half data/mc_tp_expert_half data/mc_tp2_expert_half data/mc_v_expert_half"
D1="$D1 data/mc_osm_expert_half data/mc_osm_y_expert"
D2="data/mc_ped_dagger_mc2 data/mc_vped_dagger_mc2 data/mc_fl_dagger_mc2 data/mc_fl_tp_dagger_mc2 data/mc_tp3_dagger_mc2"
D2="$D2 data/mc_osm_dagger_mc2_half data/mc_osm_y_dagger_mc2_half"
D3="data/mc_ped_dagger_mc3 data/mc_vped_dagger_mc3 data/mc_tp3_dagger_mc3 data/mc_osm_dagger_mc3 data/mc_osm_y_dagger_mc3"
D4="data/mc_osm_dagger_mc4 data/mc_osm_y_dagger_mc4"
DK="data/mc_kv_expert data/mc_tp_kv_expert data/mc_kv_dagger_mc2 data/mc_osm_kv_dagger_mc2 data/mc_osm_kv2_dagger_mc2"
DK="$DK data/mc_kv_dagger_mc3 data/mc_osm_kv_dagger_mc3 data/mc_osm_kv2_dagger_mc3 data/mc_osm_kv_dagger_mc4"
T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
W="--dir_weight data/mc_ped_half=2 data/mc_tp2_expert_half=2 data/mc_osm_expert_half=1.5 data/mc_osm_y_expert=1.5"
W="$W data/mc_ped_dagger_mc2=2 data/mc_vped_dagger_mc2=2 data/mc_fl_dagger_mc2=2 data/mc_fl_tp_dagger_mc2=3 data/mc_tp3_dagger_mc2=3"
W="$W data/mc_osm_dagger_mc2_half=1.5 data/mc_osm_y_dagger_mc2_half=1.5 data/mc_tp_kv_expert=2 data/mc_kv_dagger_mc2=2"
W="$W data/mc_osm_kv_dagger_mc2=1.5 data/mc_osm_kv2_dagger_mc2=1.5"
W="$W data/mc_ped_dagger_mc3=2 data/mc_vped_dagger_mc3=2 data/mc_tp3_dagger_mc3=3 data/mc_osm_dagger_mc3=1.5 data/mc_osm_y_dagger_mc3=1.5"
W="$W data/mc_kv_dagger_mc3=2 data/mc_osm_kv_dagger_mc3=2 data/mc_osm_kv2_dagger_mc3=2"
W="$W data/mc_osm_dagger_mc4=2 data/mc_osm_y_dagger_mc4=2 data/mc_osm_kv_dagger_mc4=3"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D1 $D2 $D3 $D4 $DK) GB"
if [ "$(gb $D1 $D2 $D3 $D4 $DK | cut -d. -f1)" -ge 21 ]; then   # load_shards holds it all in RAM (32 GB, with the desktop apps)
  half mc_tp_expert_half
  D1="${D1/data\/mc_tp_expert_half /data/mc_tp_expert_half_half }"
  echo "training data with mc_tp_expert at a quarter: $(gb $D1 $D2 $D3 $D4 $DK) GB"
fi
[ -f runs/keipilot_mc5/last.pt ] || uv run scripts/train.py --data $D1 $D2 $D3 $D4 $DK --init runs/keipilot_mc4/last.pt \
    --out runs/keipilot_mc5 --epochs 4 --samples_per_epoch 200000 $T $W || exit 1
echo MC_TRAIN_MC5_DONE

# ---- 3. evaluation
M=runs/keipilot_mc5/last.pt
S="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold --osm_town osm_sapporo --routes 20"
for e in 100 200 300; do      # KeiView's Sapporo first: the point of this round
  f=runs/eval/osm_sapporo/keipilot_mc5_hold_red$([ $e = 100 ] || echo _ep$e)_keiview
  [ -f $f.json ] || $S --episode0 $e --renderer keiview --workers 3 --out $f.json
done
echo MC_SAPPORO_KV_MC5_DONE
f=runs/mc_work/survey/mc5_eval_ep3000
[ -f $f.json ] || uv run tools/kv_tta_survey.py $M osm_sapporo 3000 10 $f.json > $f.log 2>&1
echo MC_SURVEY_MC5_DONE
for e in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500; do
  f=runs/eval/osm_sapporo/keipilot_mc5_hold_red$([ $e = 100 ] || echo _ep$e)
  [ -f $f.json ] || $S --episode0 $e --workers 6 --out $f.json
done
echo MC_SAPPORO_MC5_DONE
E="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold"
ev() { local name=$1; shift; [ -f runs/eval/keipilot_mc5_hold_red_$name.json ] || $E "$@"; }
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
echo MC_ROUND4_DONE
