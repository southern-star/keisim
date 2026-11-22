#!/bin/bash
# ReconView S2 (docs/recon_view.md): depth data from KeiView, the depth head on the frozen mc14, its accuracy, the ONNX.
#   bash docs/recon/s2.sh            (from the repository root; every step is skipped once it is done)
# Needs KeiView (npm install in web/, a Chrome, a GPU), runs/keipilot_mc14/last.pt, and for the ONNX step onnx and
# onnxruntime (uv run --with onnx --with onnxruntime). Data: about 6.5 GB under data/recon_depth_*.
set -o pipefail
PY=${PY:-.venv/bin/python}
M=runs/keipilot_mc14/last.pt
OUT=runs/keipilot_mc14d
# the expert on KeiView's wide_narrow rig, the rig's own cameras only (no displaced ones: the head's lines of sight
# are theirs), every frame with its depth
R="--mode expert --renderer keiview --rig wide_narrow --virtual_prob 0 --depth 1 --workers ${WORKERS:-3}"
col() {   # col NAME ARGS...: collect.py into data/NAME, once
  local n=$1; shift
  [ -f data/$n/.done ] && return
  $PY scripts/collect.py $R "$@" --out data/$n || return 1
  touch data/$n/.done
}
# ---- 1. data: 100,000 frames in the towns of the training range (seeds 0-399; the Sapporo boxes n / e / s)
J="--jam_fixes 1"
col recon_depth_classic --episodes_per_town 3 --frames 16000 --seed 4101 || exit 1
col recon_depth_twophase --town_style twophase $J --episodes_per_town 3 --frames 16000 --seed 4102 || exit 1
col recon_depth_roadtypes --town_style roadtypes $J --episodes_per_town 3 --frames 14000 --seed 4103 || exit 1
col recon_depth_suburban --town_style suburban $J --episodes_per_town 3 --frames 10000 --seed 4104 || exit 1
col recon_depth_bigjunction --town_style bigjunction $J --crosswalks 0.7 --crosswalk_spawn 3.0 --episodes_per_town 3 --frames 10000 --seed 4105 || exit 1
col recon_depth_bicycles --town_style twophase $J --bicycles 8 --bike_visible 1 --episodes_per_town 3 --frames 11000 --seed 4106 || exit 1
col recon_depth_parked --town_style twophase $J --parked 4 --dart_out 0.6 --episodes_per_town 3 --frames 11000 --seed 4107 || exit 1
col recon_depth_osm --osm_towns osm_sapporo_n osm_sapporo_e osm_sapporo_s $J --episodes_per_town 2 --frames 12000 --seed 4108 || exit 1
# the evaluation town (never trained on): 2,000 frames, for tools/depth_eval.py only
col recon_depth_eval_sapporo --osm_towns osm_sapporo --allow_eval_town $J --episodes_per_town 2 --frames 2000 --seed 4199 || exit 1
echo S2_COLLECT_DONE

# ---- 2. the head (the driving model frozen)
TRAIN="data/recon_depth_classic data/recon_depth_twophase data/recon_depth_roadtypes data/recon_depth_suburban data/recon_depth_bigjunction data/recon_depth_bicycles data/recon_depth_parked data/recon_depth_osm"
[ -f $OUT/.done ] || { $PY scripts/train_depth.py --init $M --data $TRAIN --out $OUT --epochs ${EPOCHS:-15} ${WIDTH:+--width $WIDTH} && touch $OUT/.done; } || exit 1
echo S2_TRAIN_DONE

# ---- 3. accuracy on the evaluation town, and the ONNX for the browser
mkdir -p runs/eval/recon
$PY tools/depth_eval.py $OUT/last.pt --data data/recon_depth_eval_sapporo --out runs/eval/recon/g2_depth_sapporo.json || exit 1
PYONNX=${PYONNX:-uv run --offline --with onnx --with onnxruntime python}
[ -f web/models/keipilot_mc_depth.onnx ] || $PYONNX scripts/export_onnx.py $OUT/last.pt web/models/keipilot_mc_depth.onnx || exit 1
echo S2_DONE
