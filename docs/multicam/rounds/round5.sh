#!/bin/bash
# Multi-camera KeiPilot, round 5 (runs/mc_work/HANDOVER.md, "5 回目"), from mc5b (round 4b). On Sapporo, 4 of mc5b's
# 5 KeiSim collisions are late yields while it creeps after a 止まれ (it slows about 0.5 s after the expert and stops
# past the wait point), and about three quarters of its long stands are mutual waits: an NPC that gives way to the
# standing ego waits for it, and the model waits for the standing NPC. New:
#  - TrafficConfig.ego_patience_s (collect.py / evaluate.py --ego_patience): an NPC stops waiting for an ego that has
#    stood still 3 s. Sapporo's evaluations with it get the suffix _pat3, and v24 and mc5b are evaluated again with it
#    (tools/sap300.py keipilot_v24:_pat3 ...). The grid suites, KeiView's long3c and the survey stay without it, so
#    they compare with rounds 1-4
#  - collect.py --freeze_reasons curve cruise crossing: a model that freezes after a 止まれ with nobody coming is
#    handed to the expert too
# 1. DAgger with mc5b driving, Sapporo's training boxes, new town variants (--osm_seed0 3000+): KeiSim's N and Y sets
#    as round 4, then KeiView 5 x 9000 frames; meanwhile v24 and mc5b on KeiSim's Sapporo 300 with _pat3
# 2. mc6 = mc5b trained on rounds 1-5, 4 x 200k samples. For the RAM: round 2's Sapporo KeiSim DAgger left out,
#    round 3's halved, rounds 2-3's KeiView Sapporo sets halved, mc_tp_expert at a quarter. Round 4's Sapporo sets
#    weigh less than round 5's (KeiSim x1.5 / x2, KeiView x2 / x3), so KeiView's share stays near round 4's 0.43
# 3. evaluation: mc6 on Sapporo with _pat3 (KeiView 60 and KeiSim 300 at once), the survey; then KeiView's Sapporo
#    for mc5b and v24 with _pat3 and KeiView's long3c for mc6, beside the 16 grid suites for mc6
# Every step skips what is already there. Start / restart: runs/mc_work/start_round5.sh.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q "ego_patience_s" keisim/world.py || { echo "keisim has no TrafficConfig.ego_patience_s"; exit 1; }
R="--rig wide_narrow"
G="--mode dagger --ckpt runs/keipilot_mc5b/last.pt --freeze_takeover 4 --freeze_reasons curve cruise crossing --ego_patience 3"
OSM="--osm_towns osm_sapporo_n osm_sapporo_e osm_sapporo_s --jam_fixes 1 --ego_cross_rate 0.3"
N="$OSM --vehicle_spacing 24 45 --focus_yield 0.3 --focus_reasons yield crossing stop_sign junction_blocked pedestrian"
Y="$OSM --vehicle_spacing 20 28 --focus_yield 0.15 --focus_reasons yield crossing stop_sign"
KVN="$OSM --vehicle_spacing 20 30 --focus_yield 0.3 --focus_reasons yield crossing stop_sign pedestrian"
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

# ---- 1. data, and meanwhile v24 and mc5b on KeiSim's Sapporo with _pat3
uv run scripts/collect.py $R $G $N --episodes_per_town 2 --osm_seed0 3000 --out data/mc_osm_dagger_mc5b --frames 40000 \
    --workers 6 --seed 4501 || exit 1
uv run scripts/collect.py $R $G $Y --episodes_per_town 2 --osm_seed0 3000 --out data/mc_osm_y_dagger_mc5b --frames 25000 \
    --workers 6 --seed 4502 || exit 1
echo MC_COLLECT_KEISIM_DONE
kv_data() {
  for i in 0 1 2 3 4; do    # collect.py resumes its workers: every chunk adds 3000 frames each, on its own variants
    uv run scripts/collect.py $R $G $KVN --renderer keiview --episodes_per_town 5 --osm_seed0 $((3000 + 10 * i)) \
        --out data/mc_osm_kv_dagger_mc5b --frames $((9000 * (i + 1))) --workers 3 --seed 4511 || return 1
  done
}
rerun_ks() { sap runs/keipilot_v24/last.pt keisim; sap runs/keipilot_mc5b/last.pt keisim; echo MC_SAPPORO_RERUN_DONE; }
kv_data & K=$!
rerun_ks & E=$!
wait $K || exit 1
wait $E
echo MC_COLLECT_R5_DONE

# ---- 2. mc6
half() {    # data/<name>_half: every other shard of data/<name> (symlinks), to stay within the RAM
  local src=data/$1 dst=data/$1_half
  [ -d $dst ] && return
  mkdir -p $dst && cp $src/meta.json $dst/
  ls $src/*.npz | awk 'NR % 2 == 1' | while read f; do ln -s ../$1/$(basename $f) $dst/; done
}
for d in mc_ped mc_tp_expert_half mc_osm_dagger_mc3 mc_osm_y_dagger_mc3 mc_osm_kv_dagger_mc2 mc_osm_kv2_dagger_mc2 \
         mc_osm_kv_dagger_mc3 mc_osm_kv2_dagger_mc3; do half $d; done
D1="data/mc_expert_half data/mc_ped_half data/mc_tp_expert_half_half data/mc_tp2_expert_half data/mc_v_expert_half"
D1="$D1 data/mc_osm_expert_half data/mc_osm_y_expert"
D2="data/mc_ped_dagger_mc2 data/mc_vped_dagger_mc2 data/mc_fl_dagger_mc2 data/mc_fl_tp_dagger_mc2 data/mc_tp3_dagger_mc2"
D3="data/mc_ped_dagger_mc3 data/mc_vped_dagger_mc3 data/mc_tp3_dagger_mc3 data/mc_osm_dagger_mc3_half data/mc_osm_y_dagger_mc3_half"
D4="data/mc_osm_dagger_mc4 data/mc_osm_y_dagger_mc4"
D5="data/mc_osm_dagger_mc5b data/mc_osm_y_dagger_mc5b"
DK="data/mc_kv_expert data/mc_tp_kv_expert data/mc_kv_dagger_mc2 data/mc_osm_kv_dagger_mc2_half data/mc_osm_kv2_dagger_mc2_half"
DK="$DK data/mc_kv_dagger_mc3 data/mc_osm_kv_dagger_mc3_half data/mc_osm_kv2_dagger_mc3_half data/mc_osm_kv_dagger_mc4"
DK="$DK data/mc_osm_kv_dagger_mc5b"
T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8 --kv_weight 2 --dagger_weight 2"
W="--dir_weight data/mc_ped_half=2 data/mc_tp2_expert_half=2 data/mc_osm_expert_half=1.5 data/mc_osm_y_expert=1.5"
W="$W data/mc_ped_dagger_mc2=2 data/mc_vped_dagger_mc2=2 data/mc_fl_dagger_mc2=2 data/mc_fl_tp_dagger_mc2=3 data/mc_tp3_dagger_mc2=3"
W="$W data/mc_tp_kv_expert=2 data/mc_kv_dagger_mc2=2 data/mc_osm_kv_dagger_mc2_half=1.5 data/mc_osm_kv2_dagger_mc2_half=1.5"
W="$W data/mc_ped_dagger_mc3=2 data/mc_vped_dagger_mc3=2 data/mc_tp3_dagger_mc3=3"
W="$W data/mc_osm_dagger_mc3_half=1.5 data/mc_osm_y_dagger_mc3_half=1.5"
W="$W data/mc_kv_dagger_mc3=2 data/mc_osm_kv_dagger_mc3_half=2 data/mc_osm_kv2_dagger_mc3_half=2"
W="$W data/mc_osm_dagger_mc4=1.5 data/mc_osm_y_dagger_mc4=1.5 data/mc_osm_kv_dagger_mc4=2"
W="$W data/mc_osm_dagger_mc5b=2 data/mc_osm_y_dagger_mc5b=2 data/mc_osm_kv_dagger_mc5b=3"
gb() { du -sLc "$@" | tail -1 | awk '{printf "%.1f", $1 / 1048576}'; }
echo "training data: $(gb $D1 $D2 $D3 $D4 $D5 $DK) GB"
if [ "$(gb $D1 $D2 $D3 $D4 $D5 $DK | cut -d. -f1)" -ge 21 ]; then   # load_shards holds it all in RAM (32 GB)
  half mc_expert_half
  D1="${D1/data\/mc_expert_half /data/mc_expert_half_half }"
  echo "training data with mc_expert at a quarter: $(gb $D1 $D2 $D3 $D4 $D5 $DK) GB"
fi
[ -f runs/keipilot_mc6/last.pt ] || uv run scripts/train.py --data $D1 $D2 $D3 $D4 $D5 $DK --init runs/keipilot_mc5b/last.pt \
    --out runs/keipilot_mc6 --epochs 4 --samples_per_epoch 200000 $T $W || exit 1
echo MC_TRAIN_MC6_DONE

# ---- 3. evaluation
M=runs/keipilot_mc6/last.pt
E="uv run scripts/evaluate.py --agent model --ckpt $M --brake_hold --red_hold"
ev() { local name=$1; shift; [ -f runs/eval/keipilot_mc6_hold_red_$name.json ] || $E "$@"; }
kv() {
  sap $M keiview
  echo MC_SAPPORO_KV_MC6_DONE
  local f=runs/mc_work/survey/mc6_eval_ep3000
  [ -f $f.json ] || uv run tools/kv_tta_survey.py $M osm_sapporo 3000 10 $f.json > $f.log 2>&1
  echo MC_SURVEY_MC6_DONE
  sap runs/keipilot_mc5b/last.pt keiview
  sap runs/keipilot_v24/last.pt keiview
  echo MC_SAPPORO_KV_RERUN_DONE
  ev long3c_keiview --suite long3c --renderer keiview --workers 3
  ev long3c_dense_keiview --suite long3c --dense --renderer keiview --workers 3
}
ks() {
  sap $M keisim
  echo MC_SAPPORO_MC6_DONE
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
echo MC_ROUND5_DONE
