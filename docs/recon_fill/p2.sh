#!/bin/bash
# ReconFill P2 (docs/recon_fill.md 6): the fill's colours by one-step generation. Needs P1's model
# (runs/reconfill_b/last.pt, web/models/reconfill.onnx: docs/recon_fill/p1.sh), its data, a GPU, headless Chrome.
#   bash docs/recon_fill/p2.sh base              the flicker of P1's fill and of the town's own fill (before training)
#   RUN=1 bash docs/recon_fill/p2.sh train       a colour net (RUN=1, 2, 3: the settings of 6.2), its offline numbers
#   RUN=1 bash docs/recon_fill/p2.sh export fidelity same drive speed     that model in the browser (gate G-P2b)
#   MAIN=/path/to/a/checkout/of/main ... same    (its web/ needs node_modules, models and .towns)
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1
D=runs/eval/recon_fill
PY=${PY:-.venv/bin/python}
PYONNX=${PYONNX:-uv run --offline --with onnx --with onnxruntime python}
STEPS=${*:-base}
RUN=${RUN:-1}
GEN=models/reconfill_gen.onnx
TOWNS="1000 1015-98782ab7 osm_sapporo"
run() { (cd web && node tools/recon_check.mjs "$@" 2>&1 | grep -v '^\[warn\]\|^\[error\] Failed to load resource\|onnxruntime'); }

# 0. what the flicker measure says of P1's fill and of the town's own fill: 60 s in three towns, every 2 s
if [[ " $STEPS " == *" base "* ]]; then
  mkdir -p $D/p2_base_p1 $D/p2_base_town
  for town in $TOWNS; do
    run --town $town --sec 60 --every 2 --recon model --fill model --out ../$D/p2_base_p1 > $D/p2_base_p1/$town.log
    run --town $town --sec 60 --every 2 --recon model --fill oracle --out ../$D/p2_base_town > $D/p2_base_town/$town.log
    echo "base $town: $(grep -h '^  flicker' $D/p2_base_p1/$town.log $D/p2_base_town/$town.log | sed 's/.*colours \([0-9.]*\) apart.*/\1/' | tr '\n' ' ')"
  done
fi

# 1. a colour net on top of P1's net, and its numbers on the eval towns' samples
if [[ " $STEPS " == *" train "* ]]; then
  case $RUN in
    1) SET="--mu -2.0 --sigma 2.0 --ratio 0.75 --p 0.75" ;;
    2) SET="--mu -0.4 --sigma 1.0 --ratio 0.25 --p 1.0" ;;
    3) SET="${SET3:?the better of runs 1 and 2: SET3='--mu .. --sigma .. --ratio .. --p ..'} --width 32 64 128 256" ;;
  esac
  O=runs/reconfill_gen$RUN
  [ -f $O/.done ] || { $PY scripts/train_fillgen.py --shape runs/reconfill_b/last.pt --data data/recon_fill/train --val data/recon_fill/eval --out $O --epochs ${EPOCHS:-100} $SET > $O.log 2>&1 && touch $O/.done; } || exit 1
  cp $O/log.json $D/p2_train_$RUN.json
  $PY tools/fill_eval.py runs/reconfill_b/last.pt $O/last.pt --names "P1（回帰）" "P2（色を生成、$RUN 本目）" --data data/recon_fill/eval \
    --out $D/p2_offline_$RUN.json --sheet docs/recon_fill/p2_offline_$RUN.jpg --frames ${SHEET_FRAMES:-60 300 520 700}
fi

if [[ " $STEPS " == *" export "* ]]; then
  $PYONNX scripts/export_fill_onnx.py runs/reconfill_gen$RUN/last.pt web/$GEN 2>&1 | tail -n 3
  $PYONNX scripts/export_fill_onnx.py --same web/models/reconfill.onnx web/$GEN
fi

# 2. the page with the generated colours: the measurement (the shape must be P1's), the flicker, the pictures
if [[ " $STEPS " == *" fidelity "* ]]; then
  mkdir -p $D/p2
  for town in $TOWNS; do
    run --town $town --sec 60 --every 2 --recon model --fill model --fillmodel $GEN --out ../$D/p2 --shotdir ../runs/recon_fill/p2/shots --shots all > $D/p2/$town.log
    echo "fidelity $town: $(grep '^digest' $D/p2/$town.log) $(grep -h '^  flicker' $D/p2/$town.log | sed 's/.*colours \([0-9.]*\) apart.*/flicker \1/')"
  done
fi

# 3. nothing changes without &fill (as docs/recon_fill/p1.sh)
if [[ " $STEPS " == *" same "* ]]; then
  mkdir -p $D/p2_same
  for i in ${SAME_RUNS:-1 2 3 4 5 6}; do
    run --town 1000 --sec 30 --every 2 --recon model --out ../$D/p2_same --name here_$i > $D/p2_same/here_$i.log
    echo "here $i: $(grep '^digest' $D/p2_same/here_$i.log)"
    if [ -n "$MAIN" ]; then
      (cd "$MAIN/web" && node tools/recon_check.mjs --town 1000 --sec 30 --every 2 --recon model --out "$OLDPWD/$D/p2_same" --name main_$i 2>&1) | grep -v '^\[warn\]\|^\[error\] Failed to load resource' > $D/p2_same/main_$i.log
      echo "main $i: $(grep '^digest' $D/p2_same/main_$i.log)"
    fi
  done
fi

# 4. the drive with and without the fill, as a visitor's page draws it
if [[ " $STEPS " == *" drive "* ]]; then
  mkdir -p $D/p2_drive
  for town in 1000 osm_sapporo; do
    for i in ${DRIVE_RUNS:-1 2 3 4}; do
      run --town $town --sec 60 --every 1 --recon off --model models/keipilot_mc_depth.onnx --out ../$D/p2_drive --name ${town}_off_$i > $D/p2_drive/${town}_off_$i.log
      run --town $town --sec 60 --every 1 --recon model --fill model --fillmodel $GEN --measure 0 --out ../$D/p2_drive --name ${town}_fillg_$i > $D/p2_drive/${town}_fillg_$i.log
      echo "drive $town $i: $(grep -h '^digest' $D/p2_drive/${town}_{off,fillg}_$i.log | awk '{print $4}' | tr '\n' ' ')"
    done
  done
fi

# 5. the page's speed in real time: no fill, P1's fill, the generated colours, five times each in turns
if [[ " $STEPS " == *" speed "* ]]; then
  mkdir -p $D/p2_speed
  for i in 1 2 3 4 5; do
    for town in 1000 osm_sapporo; do
      run --town $town --realtime 30 --sec 2 --every 2 --recon model --measure 0 > $D/p2_speed/${town}_accum_$i.log
      run --town $town --realtime 30 --sec 2 --every 2 --recon model --fill model --measure 0 > $D/p2_speed/${town}_fillm_$i.log
      run --town $town --realtime 30 --sec 2 --every 2 --recon model --fill model --fillmodel $GEN --measure 0 > $D/p2_speed/${town}_fillg_$i.log
    done
    grep -h "^real time" $D/p2_speed/*_$i.log | cut -c1-200
  done
fi
