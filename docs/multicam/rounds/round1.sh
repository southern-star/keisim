#!/bin/bash
# Multi-camera KeiPilot, round 1: the wide_narrow rig (keisim CAMERA_RIGS: a cylindrical 180 deg camera that sees the
# cars with priority from the 止まれ lines, and a 50 deg narrow camera), KeiSim only (KeiView has no rigs yet).
# 1. expert data with the rig: the grid towns (classic, varied, two-phase) and Sapporo's three training boxes, in the
#    mix of v0.10's KeiSim data (the flags of README's dir_expert / hist_ped / tp_expert / tp2_expert / osm rounds)
# 2. mc1 = v0.10 (runs/keipilot_v24) with two cameras (shared trunk, position embeddings per camera), 6 x 200k samples
# 3. DAgger with mc1 driving; mc2 = mc1 trained on all of it, 3 x 200k
# 4. Sapporo, 60 routes (v0.10 in KeiSim: DS 0.928, 6 collisions, 7 of v0.10's 8 Sapporo collisions came out of the
#    stop at a 止まれ line), then long3c-f and the release suites, all in KeiSim
# Every step skips what is already there (collect.py resumes per worker). Restart: runs/mc_work/start_round1.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
R="--rig wide_narrow"
TP="--town_style twophase --jam_fixes 1 --vehicle_spacing 24 45"
OSM="--osm_towns osm_sapporo_n osm_sapporo_e osm_sapporo_s --jam_fixes 1 --ego_cross_rate 0.3 --episodes_per_town 2"
N="$OSM --vehicle_spacing 24 45 --focus_yield 0.3 --focus_reasons yield crossing stop_sign junction_blocked pedestrian"
Y="$OSM --vehicle_spacing 20 28 --focus_yield 0.15 --focus_reasons yield crossing stop_sign"

# ---- 1. expert data (about 525k frames)
uv run scripts/collect.py $R --out data/mc_expert --frames 150000 --workers 10 --seed 4001 --ego_cross_rate 0.3 || exit 1
uv run scripts/collect.py $R --out data/mc_ped --frames 40000 --workers 10 --seed 4002 --ego_cross_rate 0.45 --ped_spacing 14 26 || exit 1
uv run scripts/collect.py $R $TP --out data/mc_tp_expert --frames 100000 --workers 10 --seed 4003 --ego_cross_rate 0.2 || exit 1
uv run scripts/collect.py $R $TP --out data/mc_tp2_expert --frames 60000 --workers 10 --seed 4004 --ego_cross_rate 0.2 --focus_yield 0.3 || exit 1
uv run scripts/collect.py $R --town_style varied --jam_fixes 1 --out data/mc_v_expert --frames 40000 --workers 10 --seed 4005 --ego_cross_rate 0.3 || exit 1
uv run scripts/collect.py $R $N --out data/mc_osm_expert --frames 90000 --workers 10 --seed 4006 || exit 1
uv run scripts/collect.py $R $Y --out data/mc_osm_y_expert --frames 45000 --workers 10 --seed 4007 || exit 1
echo MC_COLLECT_EXPERT_DONE

# ---- 2. mc1
D="data/mc_expert data/mc_ped data/mc_tp_expert data/mc_tp2_expert data/mc_v_expert data/mc_osm_expert data/mc_osm_y_expert"
T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 10"
W="--dir_weight data/mc_ped=2 data/mc_tp2_expert=2 data/mc_osm_expert=1.5 data/mc_osm_y_expert=1.5"
[ -f runs/keipilot_mc1/last.pt ] || uv run scripts/train.py --data $D --init runs/keipilot_v24/last.pt --out runs/keipilot_mc1 \
    --epochs 6 --samples_per_epoch 200000 $T $W || exit 1
echo MC_TRAIN_MC1_DONE

# ---- 3. DAgger with mc1, mc2
G="--mode dagger --ckpt runs/keipilot_mc1/last.pt --workers 6"
uv run scripts/collect.py $R $G --out data/mc_dagger_mc1 --frames 40000 --seed 4101 --ego_cross_rate 0.3 || exit 1
uv run scripts/collect.py $R $G $TP --out data/mc_tp_dagger_mc1 --frames 40000 --seed 4102 --ego_cross_rate 0.3 --focus_yield 0.3 || exit 1
uv run scripts/collect.py $R $G $N --out data/mc_osm_dagger_mc1 --frames 60000 --seed 4103 || exit 1
uv run scripts/collect.py $R $G $Y --out data/mc_osm_y_dagger_mc1 --frames 30000 --seed 4104 || exit 1
echo MC_COLLECT_DAGGER_MC1_DONE
DG="data/mc_dagger_mc1 data/mc_tp_dagger_mc1 data/mc_osm_dagger_mc1 data/mc_osm_y_dagger_mc1"
W2="$W data/mc_osm_dagger_mc1=1.5 data/mc_osm_y_dagger_mc1=1.5"
[ -f runs/keipilot_mc2/last.pt ] || uv run scripts/train.py --data $D $DG --init runs/keipilot_mc1/last.pt --out runs/keipilot_mc2 \
    --epochs 3 --samples_per_epoch 200000 $T $W2 || exit 1
echo MC_TRAIN_MC2_DONE

# ---- 4. evaluation (KeiSim; the rig comes from the checkpoint)
for M in mc2 mc1; do
  for e in 100 200 300; do
    f=runs/eval/osm_sapporo/keipilot_${M}_hold_red$([ $e = 100 ] || echo _ep$e)
    [ -f $f.json ] || uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_$M/last.pt --brake_hold --red_hold \
        --osm_town osm_sapporo --routes 20 --episode0 $e --workers 6 --out $f.json
  done
done
echo MC_SAPPORO_DONE
M=runs/keipilot_mc2/last.pt
E="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold"
ev() { local name=$1; shift; [ -f runs/eval/keipilot_mc2_hold_red_$name.json ] || $E "$@"; }
for a in long3c long3d long3e long3f; do ev $a --suite $a --workers 6; ev ${a}_dense --suite $a --dense --workers 6; done
echo MC_LONG3_DONE
ev test --suite test --workers 6
ev test2 --suite test2 --workers 6
ev train --suite train --workers 6
ev test_dense --suite test --dense --max_steps 6000 --workers 6
ev long --suite long --workers 6
ev long_dense --suite long --dense --workers 6
ev long2 --suite long2 --workers 6
ev long2_dense --suite long2 --dense --workers 6
echo MC_ROUND1_DONE
