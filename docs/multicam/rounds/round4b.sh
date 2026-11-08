#!/bin/bash
# Multi-camera KeiPilot, round 4b: mc5 hesitated after its 止まれ stops in KeiSim (Sapporo 300: needless standing at
# 止まれ junctions 533 s, 12 passes > 5 s, against mc4's 145 s and 1; 2 of its 3 freezes there). load_shards' stale_creep
# dropped only the older frames past a stop whose label was faster than 徐行 and kept the waits: 43 % waits in what
# remained of rounds 1-3's KeiSim frames there, 13 % in label version 6's. Now every such frame goes (keipilot/data.py).
# mc5b = mc4 trained exactly as mc5 otherwise (same data, weights, 4 x 200k samples). Evaluation as round 4, Sapporo
# first; the grid suites last (they have no 止まれ lines). Every step skips what is already there.
# Start / restart: runs/mc_work/start_round4b.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q "whatever their label" keipilot/data.py || { echo "keipilot/data.py still drops by label"; exit 1; }

# ---- 1. mc5b
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
[ -f runs/keipilot_mc5b/last.pt ] || uv run scripts/train.py --data $D1 $D2 $D3 $D4 $DK --init runs/keipilot_mc4/last.pt \
    --out runs/keipilot_mc5b --epochs 4 --samples_per_epoch 200000 $T $W || exit 1
echo MC_TRAIN_MC5B_DONE

# ---- 2. evaluation
M=runs/keipilot_mc5b/last.pt
S="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold --osm_town osm_sapporo --routes 20"
for e in 100 200 300; do      # KeiView's Sapporo first: the point of this round
  f=runs/eval/osm_sapporo/keipilot_mc5b_hold_red$([ $e = 100 ] || echo _ep$e)_keiview
  [ -f $f.json ] || $S --episode0 $e --renderer keiview --workers 3 --out $f.json
done
echo MC_SAPPORO_KV_MC5B_DONE
f=runs/mc_work/survey/mc5b_eval_ep3000
[ -f $f.json ] || uv run tools/kv_tta_survey.py $M osm_sapporo 3000 10 $f.json > $f.log 2>&1
echo MC_SURVEY_MC5B_DONE
for e in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500; do
  f=runs/eval/osm_sapporo/keipilot_mc5b_hold_red$([ $e = 100 ] || echo _ep$e)
  [ -f $f.json ] || $S --episode0 $e --workers 6 --out $f.json
done
echo MC_SAPPORO_MC5B_DONE
E="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold"
ev() { local name=$1; shift; [ -f runs/eval/keipilot_mc5b_hold_red_$name.json ] || $E "$@"; }
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
echo MC_ROUND4B_DONE
