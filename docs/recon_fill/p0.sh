#!/bin/bash
# ReconFill P0 (docs/recon_fill.md 3 and 4): the view's holes filled in from the town itself (&fill=oracle).
# Headless Chrome with a GPU; needs web/models/keipilot_mc_depth.onnx (docs/recon/s2.sh).
#   bash docs/recon_fill/p0.sh [same] [fidelity] [drive] [dump]      (from the repository root; all four without arguments)
#   MAIN=/path/to/a/checkout/of/main bash docs/recon_fill/p0.sh same  (its web/ needs node_modules, models and .towns)
set -o pipefail
cd "$(dirname "$0")/../../web" || exit 1
D=../runs/eval/recon_fill
STEPS=${*:-same fidelity drive dump}
run() { node tools/recon_check.mjs "$@" 2>&1 | grep -v '^\[warn\]\|^\[error\] Failed to load resource'; }

# 1. nothing changes without &fill: the measurements of this checkout and of main's (MAIN), without --fill, 30 s in
#    town 1000; runs of the same digests must have the same measurements (tools/recon_g3.py same)
if [[ " $STEPS " == *" same "* ]]; then
  mkdir -p $D/p0_same
  for i in ${SAME_RUNS:-1 2 3}; do
    run --town 1000 --sec 30 --every 2 --recon model --out $D/p0_same --name here_$i > $D/p0_same/here_$i.log
    echo "here $i: $(grep '^digest' $D/p0_same/here_$i.log)"
    if [ -n "$MAIN" ]; then
      (cd "$MAIN/web" && node tools/recon_check.mjs --town 1000 --sec 30 --every 2 --recon model --out "$OLDPWD/$D/p0_same" --name main_$i 2>&1) | grep -v '^\[warn\]\|^\[error\] Failed to load resource' > $D/p0_same/main_$i.log
      echo "main $i: $(grep '^digest' $D/p0_same/main_$i.log)"
    fi
  done
fi

# 2. the fill from the town itself: 60 s in three towns, every 2 s the frame without a fill (A, A+), with this frame's
#    fill and with the one before (the holes only), and the fill alone
if [[ " $STEPS " == *" fidelity "* ]]; then
  mkdir -p $D/p0
  for town in 1000 1015-98782ab7 osm_sapporo; do
    run --town $town --sec 60 --every 2 --recon model --fill oracle --out $D/p0 --shotdir ../runs/recon_fill/p0/shots --shots all > $D/p0/$town.log
    echo "fidelity $town: $(grep '^digest' $D/p0/$town.log)"
  done
fi

# 3. the drive: 60 s without ReconView, with it, and with the fill (holes, and all of it), as a visitor's page draws
#    them (no measurements); the fill does not touch the model's input, so the "drive" digests must be shared
if [[ " $STEPS " == *" drive "* ]]; then
  mkdir -p $D/p0_drive
  for town in 1000 osm_sapporo; do
    for i in ${DRIVE_RUNS:-1 2}; do
      run --town $town --sec 60 --every 1 --recon off --model models/keipilot_mc_depth.onnx --out $D/p0_drive --name ${town}_off_$i > $D/p0_drive/${town}_off_$i.log
      run --town $town --sec 60 --every 1 --recon model --measure 0 --out $D/p0_drive --name ${town}_accum_$i > $D/p0_drive/${town}_accum_$i.log
      run --town $town --sec 60 --every 1 --recon model --fill oracle --measure 0 --out $D/p0_drive --name ${town}_fill_$i > $D/p0_drive/${town}_fill_$i.log
      run --town $town --sec 60 --every 1 --recon model --fill oracle --fillshow all --measure 0 --out $D/p0_drive --name ${town}_fillall_$i > $D/p0_drive/${town}_fillall_$i.log
      echo "drive $town $i: $(grep -h '^digest' $D/p0_drive/${town}_{off,accum,fill,fillall}_$i.log | tr '\n' ' ')"
    done
  done
fi

# 4. the training data's writer: a few hundred samples, to measure its speed and size (data/ is not in git), and the
#    checks of what it wrote
if [[ " $STEPS " == *" dump "* ]]; then
  O=../data/recon_fill_trial
  mkdir -p $D/p0_dump
  node tools/fill_dump.mjs --out $O --town 1001 --sec 150 --every 1 --preview 4 2>&1 | grep -v '^\[warn\]' > $D/p0_dump/1001.log
  node tools/fill_dump.mjs --out $O --town 1018-98782ab7 --name 1018-98782ab7_bikes --bikes 8 --parked 8 --sec 150 --every 1 --preview 4 2>&1 | grep -v '^\[warn\]' > $D/p0_dump/1018-98782ab7_bikes.log
  node tools/fill_dump.mjs --out $O --town osm_sapporo_n_s2000 --tdir .towns --sec 150 --every 1 --preview 4 2>&1 | grep -v '^\[warn\]' > $D/p0_dump/osm_sapporo_n_s2000.log
  tail -qn 1 $D/p0_dump/*.log
  (cd .. && .venv/bin/python tools/fill_data.py check data/recon_fill_trial --json runs/eval/recon_fill/p0_dump/check.json > runs/eval/recon_fill/p0_dump/check.md; tail -n 3 runs/eval/recon_fill/p0_dump/check.md)
fi
