#!/bin/bash
# ReconView G4 (docs/recon_view.md 8): the keyframes of passed places. The same frames with the current frame alone
# and with the keyframes; the oracle's depth with keyframes; the drive with and without; the page's speed.
# Headless Chrome with a GPU; needs web/models/keipilot_mc_depth.onnx (docs/recon/s2.sh).
#   bash docs/recon/g4.sh [fidelity] [oracle] [drive] [speed]       (from the repository root; all four without arguments)
#   bash docs/recon/g4.sh keep       (8.6: what of the keyframes is drawn, every variant on the same frames and keyframes)
set -o pipefail
cd "$(dirname "$0")/../../web" || exit 1
D=../runs/eval/recon
M=models/keipilot_mc_depth.onnx
STEPS=${*:-fidelity oracle drive speed}
run() { node tools/recon_check.mjs "$@" 2>&1 | grep -v '^\[warn\]\|^\[error\] Failed to load resource'; }

# 1. the model's depth: 60 s in three towns, every 2 s the frame alone (A), with the keyframes (A+), and the oracle's
#    frame alone; once more with riders (the ghost count)
if [[ " $STEPS " == *" fidelity "* ]]; then
  mkdir -p $D/g4 $D/g4_bikes
  for town in 1000 1015-98782ab7 osm_sapporo; do
    run --town $town --sec 60 --every 2 --recon model --accum 1 --out $D/g4 --shotdir ../runs/recon/g4/shots --shots all > $D/g4/$town.log
    echo "fidelity $town: $(grep '^digest' $D/g4/$town.log)"
  done
  run --town 1015-98782ab7 --bikes 8 --sec 60 --every 2 --recon model --accum 1 --out $D/g4_bikes --shotdir ../runs/recon/g4_bikes/shots --shots all > $D/g4_bikes/1015-98782ab7.log
  echo "fidelity with riders: $(grep '^digest' $D/g4_bikes/1015-98782ab7.log)"
fi

# 2. the town's own depth with keyframes: what accumulation gives when the depth is right
if [[ " $STEPS " == *" oracle "* ]]; then
  mkdir -p $D/g4_oracle
  for town in 1000 1015-98782ab7 osm_sapporo; do
    run --town $town --sec 60 --every 2 --recon oracle --accum 1 --out $D/g4_oracle > $D/g4_oracle/$town.log
    echo "oracle $town: $(grep '^digest' $D/g4_oracle/$town.log)"
  done
fi

# 3. the drive: 60 s without ReconView (the same model) and with it and its keyframes, as a visitor's page draws it
if [[ " $STEPS " == *" drive "* ]]; then
  mkdir -p $D/g4_drive
  for town in 1000 osm_sapporo; do
    for i in ${DRIVE_RUNS:-1 2}; do
      for cfg in off accum; do
        case $cfg in
          off) a="--recon off --model $M" ;;
          accum) a="--recon model --accum 1 --measure 0" ;;
        esac
        run --town $town --sec 60 --every 1 $a --out $D/g4_drive --name ${town}_${cfg}_$i > $D/g4_drive/${town}_${cfg}_$i.log
        echo "drive $town $cfg $i: $(grep '^digest' $D/g4_drive/${town}_${cfg}_$i.log) $(grep -c 'FAIL' $D/g4_drive/${town}_${cfg}_$i.log) FAIL"
      done
    done
  done
fi

# 4. speed: 12 s in real time, ReconView without and with the keyframes
if [[ " $STEPS " == *" speed "* ]]; then
  mkdir -p $D/g4_speed
  for i in ${SPEED_RUNS:-1 2 3 4 5}; do
    for town in 1000 osm_sapporo; do
      for cfg in model accum; do
        case $cfg in
          model) a="--recon model --accum 0 --measure 0" ;;
          accum) a="--recon model --accum 1 --measure 0" ;;
        esac
        run --town $town --realtime 12 --sec 2 --every 2 $a > $D/g4_speed/${town}_${cfg}_$i.log
        echo "speed $town $cfg $i: $(grep '^real time' $D/g4_speed/${town}_${cfg}_$i.log)"
      done
    done
  done
fi
# 5. what of the keyframes is drawn (docs/recon_view.md 8.6): every measurement has all variants (accum.variants)
if [[ " $STEPS " == *" keep "* ]]; then
  mkdir -p $D/g4_keep
  for town in 1000 1015-98782ab7 osm_sapporo; do
    run --town $town --sec 60 --every 2 --recon model --accum 1 --out $D/g4_keep --shotdir ../runs/recon/g4_keep/shots --shots all > $D/g4_keep/$town.log
    echo "keep $town: $(grep '^digest' $D/g4_keep/$town.log)"
  done
fi
echo G4_DONE
