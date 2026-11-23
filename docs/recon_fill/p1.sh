#!/bin/bash
# ReconFill P1 (docs/recon_fill.md 5): the fill model. Needs the data of docs/recon_fill/p1_dump.sh, a GPU, headless
# Chrome, web/models/keipilot_mc_depth.onnx.
#   bash docs/recon_fill/p1.sh time        random weights in the browser: which width, which worker (5.3)
#   bash docs/recon_fill/p1.sh train       the two models (with and without the loose rule's picture), offline numbers
#   MODEL=a bash docs/recon_fill/p1.sh export fidelity same drive speed     the chosen model in the browser (gate G-P1)
#   MAIN=/path/to/a/checkout/of/main ... same     (its web/ needs node_modules, models and .towns)
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1
D=runs/eval/recon_fill
PY=${PY:-.venv/bin/python}
PYONNX=${PYONNX:-uv run --offline --with onnx --with onnxruntime python}
STEPS=${*:-time train}
run() { (cd web && node tools/recon_check.mjs "$@" 2>&1 | grep -v '^\[warn\]\|^\[error\] Failed to load resource\|onnxruntime'); }

# 1. random weights: the page's speed in real time without a fill and with fill models of three widths, without the
#    loose input, and in a worker of its own
if [[ " $STEPS " == *" time "* ]]; then
  mkdir -p $D/p1_time
  $PYONNX scripts/export_fill_onnx.py web/models/reconfill_rand_a.onnx --random 16 32 64 128 2>&1 | tail -n 1
  $PYONNX scripts/export_fill_onnx.py web/models/reconfill_rand_b.onnx --random 24 48 96 192 2>&1 | tail -n 1
  $PYONNX scripts/export_fill_onnx.py web/models/reconfill_rand_c.onnx --random 12 24 48 96 2>&1 | tail -n 1
  $PYONNX scripts/export_fill_onnx.py web/models/reconfill_rand_nl.onnx --random 16 32 64 128 --no_loose 2>&1 | tail -n 1
  for i in ${TIME_RUNS:-1 2 3}; do
    run --town 1000 --realtime 30 --sec 2 --every 2 --recon model --measure 0 --name 1000_accum_$i > $D/p1_time/1000_accum_$i.log
    for m in a b c nl; do
      run --town 1000 --realtime 30 --sec 2 --every 2 --recon model --fill model --fillmodel models/reconfill_rand_$m.onnx --measure 0 > $D/p1_time/1000_rand${m}_$i.log
    done
    run --town 1000 --realtime 30 --sec 2 --every 2 --recon model --fill model --fillmodel models/reconfill_rand_a.onnx --fillworker own --measure 0 > $D/p1_time/1000_randaown_$i.log
    grep -h "^real time" $D/p1_time/1000_*_$i.log | cut -c1-260
  done
fi

# 2. the two models, and their numbers on the eval towns' samples
if [[ " $STEPS " == *" train "* ]]; then
  [ -f runs/reconfill_a/.done ] || { $PY scripts/train_fill.py --data data/recon_fill/train --val data/recon_fill/eval --out runs/reconfill_a --epochs ${EPOCHS:-30} > runs/reconfill_a.log 2>&1 && touch runs/reconfill_a/.done; } || exit 1
  [ -f runs/reconfill_b/.done ] || { $PY scripts/train_fill.py --data data/recon_fill/train --val data/recon_fill/eval --out runs/reconfill_b --epochs ${EPOCHS:-30} --no_loose > runs/reconfill_b.log 2>&1 && touch runs/reconfill_b/.done; } || exit 1
  cp runs/reconfill_a/log.json $D/p1_train_a.json; cp runs/reconfill_b/log.json $D/p1_train_b.json
  $PY tools/fill_eval.py runs/reconfill_a/last.pt runs/reconfill_b/last.pt --names "緩い規則の入力あり" "緩い規則の入力なし" --data data/recon_fill/eval \
    --out $D/p1_offline.json --sheet docs/recon_fill/p1_offline.jpg --frames ${SHEET_FRAMES:-60 300 520 700}
fi

M=${MODEL:-a}
if [[ " $STEPS " == *" export "* ]]; then
  $PYONNX scripts/export_fill_onnx.py runs/reconfill_$M/last.pt web/models/reconfill.onnx 2>&1 | tail -n 3
fi

# 3. the model's fill in the page: 60 s in three towns, every 2 s the frame without a fill, with the model's fill (this
#    frame's and the one before) and with the town's own fill of the same frame
if [[ " $STEPS " == *" fidelity "* ]]; then
  mkdir -p $D/p1
  for town in 1000 1015-98782ab7 osm_sapporo; do
    run --town $town --sec 60 --every 2 --recon model --fill model --out ../$D/p1 --shotdir ../runs/recon_fill/p1/shots --shots all > $D/p1/$town.log
    echo "fidelity $town: $(grep '^digest' $D/p1/$town.log)"
  done
fi

# 4. nothing changes without &fill (as docs/recon_fill/p0.sh, six runs each)
if [[ " $STEPS " == *" same "* ]]; then
  mkdir -p $D/p1_same
  for i in ${SAME_RUNS:-1 2 3 4 5 6}; do
    run --town 1000 --sec 30 --every 2 --recon model --out ../$D/p1_same --name here_$i > $D/p1_same/here_$i.log
    echo "here $i: $(grep '^digest' $D/p1_same/here_$i.log)"
    if [ -n "$MAIN" ]; then
      (cd "$MAIN/web" && node tools/recon_check.mjs --town 1000 --sec 30 --every 2 --recon model --out "$OLDPWD/$D/p1_same" --name main_$i 2>&1) | grep -v '^\[warn\]\|^\[error\] Failed to load resource' > $D/p1_same/main_$i.log
      echo "main $i: $(grep '^digest' $D/p1_same/main_$i.log)"
    fi
  done
fi

# 5. the drive with and without the model's fill, as a visitor's page draws it
if [[ " $STEPS " == *" drive "* ]]; then
  mkdir -p $D/p1_drive
  for town in 1000 osm_sapporo; do
    for i in ${DRIVE_RUNS:-1 2 3 4}; do
      run --town $town --sec 60 --every 1 --recon off --model models/keipilot_mc_depth.onnx --out ../$D/p1_drive --name ${town}_off_$i > $D/p1_drive/${town}_off_$i.log
      run --town $town --sec 60 --every 1 --recon model --measure 0 --out ../$D/p1_drive --name ${town}_accum_$i > $D/p1_drive/${town}_accum_$i.log
      run --town $town --sec 60 --every 1 --recon model --fill model --measure 0 --out ../$D/p1_drive --name ${town}_fillm_$i > $D/p1_drive/${town}_fillm_$i.log
      echo "drive $town $i: $(grep -h '^digest' $D/p1_drive/${town}_{off,accum,fillm}_$i.log | awk '{print $4}' | tr '\n' ' ')"
    done
  done
fi

# 6. the page's speed in real time, without and with the model's fill, five times each in turns
if [[ " $STEPS " == *" speed "* ]]; then
  mkdir -p $D/p1_speed
  for i in 1 2 3 4 5; do
    for town in 1000 osm_sapporo; do
      run --town $town --realtime 30 --sec 2 --every 2 --recon model --measure 0 > $D/p1_speed/${town}_accum_$i.log
      run --town $town --realtime 30 --sec 2 --every 2 --recon model --fill model ${FILLWORKER:+--fillworker $FILLWORKER} --measure 0 > $D/p1_speed/${town}_fillm_$i.log
    done
    grep -h "^real time" $D/p1_speed/*_$i.log | cut -c1-200
  done
fi
