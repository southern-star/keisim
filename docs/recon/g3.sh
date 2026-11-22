#!/bin/bash
# ReconView G3 (docs/recon_view.md 7.2): the model's depth beside the oracle's, the drive with and without ReconView,
# the page's speed; without the keyframes of S4 (--accum 0), as they were measured. Headless Chrome with a GPU; needs web/models/keipilot_mc_depth.onnx (docs/recon/s2.sh).
#   bash docs/recon/g3.sh [fidelity] [drive] [speed]       (from the repository root; all three without arguments)
# DRIVE_RUNS="3 4" bash docs/recon/g3.sh drive             (more runs of every setting, numbered 3 and 4; SPEED_RUNS likewise)
set -o pipefail
cd "$(dirname "$0")/../../web" || exit 1
D=../runs/eval/recon
M=models/keipilot_mc_depth.onnx
STEPS=${*:-fidelity drive speed}
run() { node tools/recon_check.mjs "$@" 2>&1 | grep -v '^\[warn\]\|^\[error\] Failed to load resource'; }

# 1. fidelity: 60 s in three towns, every 2 s the same frame with the model's depth and with the town's
if [[ " $STEPS " == *" fidelity "* ]]; then
  mkdir -p $D/g3
  for town in 1000 1015-98782ab7 osm_sapporo; do
    run --town $town --sec 60 --every 2 --recon model --accum 0 --out $D/g3 --shotdir ../runs/recon/g3/shots --shots all > $D/g3/$town.log
    echo "fidelity $town: $(grep '^digest' $D/g3/$town.log)"
  done
fi

# 2. the drive: 60 s, the car's state, the plan and the input images every second; without ReconView (the same
#    model), with it, and with the split view. Runs of the same digest are the same to the bit.
if [[ " $STEPS " == *" drive "* ]]; then
  mkdir -p $D/g3_drive
  for town in 1000 osm_sapporo; do
    for i in ${DRIVE_RUNS:-1 2}; do
      for cfg in off model split; do
        case $cfg in
          off) a="--recon off --model $M" ;;
          model) a="--recon model --accum 0 --measure 0" ;;
          split) a="--recon model --accum 0 --layout split --measure 0" ;;
        esac
        run --town $town --sec 60 --every 1 $a --out $D/g3_drive --name ${town}_${cfg}_$i > $D/g3_drive/${town}_${cfg}_$i.log
        echo "drive $town $cfg $i: $(grep '^digest' $D/g3_drive/${town}_${cfg}_$i.log) $(grep -c 'FAIL' $D/g3_drive/${town}_${cfg}_$i.log) FAIL"
      done
    done
  done
fi

# 3. speed: 12 s in real time, as a visitor's page; the released model, the model with the depth head, ReconView
if [[ " $STEPS " == *" speed "* ]]; then
  mkdir -p $D/g3_speed
  for i in ${SPEED_RUNS:-1 2 3 4 5}; do
    for town in 1000 osm_sapporo; do
      for cfg in public off model split; do
        case $cfg in
          public) a="--recon off" ;;
          off) a="--recon off --model $M" ;;
          model) a="--recon model --accum 0 --measure 0" ;;
          split) a="--recon model --accum 0 --layout split --measure 0" ;;
        esac
        run --town $town --realtime 12 --sec 2 --every 2 $a > $D/g3_speed/${town}_${cfg}_$i.log
        echo "speed $town $cfg $i: $(grep '^real time' $D/g3_speed/${town}_${cfg}_$i.log)"
      done
    done
  done
fi
echo G3_DONE
