#!/bin/bash
# The future head (docs/future_aux.md), phase 3: round 13's training of mc14 again, with the head and without.
#   mc14f  mc14's train.py command (round13.sh) plus --future_weight FW (0.3, chosen in phase 2 from the validation
#          numbers of docs/multicam/rounds/future_small.sh); then the head against persistence per look-ahead
#          (tools/future_eval.py -> runs/keipilot_mc14f/future_eval.json)
#   mc14c  mc14's command with another random stream (--seed 1 --split_seed 0: the same training frames), for how
#          far two trainings of the same setting are apart
# MODELS picks the models ("mc14f", "mc14c" or both). The data, sampling weights and options are mc14's, read from
# its checkpoint (runs/keipilot_mc14/last.pt). About 55 minutes for mc14f and 47 for mc14c on the RTX 3060.
# STAGE (default 0: the training only) adds round 13's evaluation of the models, after all the training, with
# mc14's safety layers and file names (runs/eval/keipilot_<model>_hold_red_yh_..., compared with mc14's files there):
#   1  the suites the head aims at, KeiSim, 440 routes: parked, crosswalks, crosswalks_sub, bigjunction, their
#      bicycle versions (_bk8) and Sapporo's 300 routes. About 6.6 hours a model (6 workers)
#   2  the rest of round 13's town suites: KeiSim (the 740 routes with Sapporo's 300 of stage 1) beside KeiView
#      (150 routes, needs web/node_modules and web/.towns). 13 to 14 hours a model
# REPRO=1 first runs mc14 again on the parked suite's 600 m routes and on three routes of bigjunction_bk8 (20 minutes,
# into runs/future_aux/repro/) and stops if a route's result is not the one in runs/eval: the comparisons need mc14's
# files (of 20 November, before the bike_defer option) to be what this code and machine give. Every step skips what
# is already there.
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
grep -q -- "--future_weight" scripts/train.py || { echo "not the code with the future head"; exit 1; }
M13=runs/keipilot_mc13/last.pt
M14=runs/keipilot_mc14/last.pt
[ -f $M13 ] && [ -f $M14 ] || { echo "$M13 or $M14 is missing"; exit 1; }
arg() { uv run python -c "import torch; print(' '.join(torch.load('$M14', map_location='cpu')['args']['$1']))"; }
D=$(arg data) || exit 1
W=$(arg dir_weight) || exit 1
ls -d $D >/dev/null || exit 1
T="--bs 48 --speed_input --speed_drop 0.5 --cf_prob 0.5 --cf_weight 1.5 --mem_input --aux_weight 0.5 --workers 8"
T="$T --kv_weight 2 --dagger_weight 2 --mmap"
FW=${FW:-0.3}
train() {   # train NAME ARGS...: mc14's training with ARGS added, once
  local n=$1; shift
  [ -f runs/keipilot_$n/last.pt ] || uv run scripts/train.py --data $D --init $M13 --out runs/keipilot_$n --epochs 4 \
      --samples_per_epoch 200000 $T --dir_weight $W "$@" || exit 1
}
MODELS=${MODELS:-mc14f}
STAGE=${STAGE:-0}
L="--brake_hold --red_hold --yield_hold 4 1 3.5 4 0.5 0.5 10"
H=hold_red_yh_4_1_3.5_4_0.5_0.5_10
repro() {   # repro SUITE_NAME ARGS...: mc14 again on (a part of) a suite; fails if a route differs from runs/eval's
  local name=$1 O=runs/future_aux/repro/keipilot_mc14_${H}_$1.json; shift
  [ -f $O ] || uv run scripts/evaluate.py --agent model --ckpt $M14 $L "$@" --tag keipilot_mc14_repro --out $O || return 1
  uv run python - $O runs/eval/keipilot_mc14_${H}_$name.json <<'PY'
import json, sys
a, b = ({(r["town"], r["episode"]): r for r in json.load(open(f))["results"]} for f in sys.argv[1:3])
keys = ("status", "RC", "DS", "time", "infractions")
diff = [k for k in sorted(a) if k not in b or any(a[k].get(x) != b[k].get(x) for x in keys)]
for k in diff:
    print("differs:", k, [(a[k].get(x), b.get(k, {}).get(x)) for x in keys])
print(f"R13F_REPRO {sys.argv[2]}: {len(a) - len(diff)}/{len(a)} routes the same")
sys.exit(1 if diff or not a else 0)
PY
}
if [ "${REPRO:-0}" = 1 ]; then
  repro parked_600m --suite parked --route_length 600 --max_steps 4000 --workers 6 || exit 1
  repro bigjunction_bk8 --suite bigjunction_bk8 --only 3002:9800 3002:9801 3000:9801 --workers 3 || exit 1
  echo R13F_REPRO_DONE
fi
for m in $MODELS; do
  case $m in
    mc14f) train mc14f --future_weight $FW
           [ -f runs/keipilot_mc14f/future_eval.json ] || uv run python tools/future_eval.py \
               --ckpt runs/keipilot_mc14f/last.pt || exit 1
           echo R13F_TRAIN_MC14F_DONE;;
    mc14c) train mc14c --seed 1 --split_seed 0
           echo R13F_TRAIN_MC14C_DONE;;
    *) echo "unknown model $m"; exit 1;;
  esac
done

# ---- evaluation (round13.sh's commands and file names)
P=3                     # --ego_patience of Sapporo's evaluations: suffix _pat3
ev() {      # ev SUITE_NAME ARGS...: evaluate.py of model $N into its default file, once
  local name=$1; shift
  [ -f runs/eval/keipilot_${N}_${H}_$name.json ] || uv run scripts/evaluate.py --agent model \
      --ckpt runs/keipilot_$N/last.pt $L "$@" || exit 1
}
sap() {     # Sapporo with _pat$P: sap keisim|keiview
  local batches="100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500" rend="" opt="--workers 6"
  [ $1 = keiview ] && { rend=_keiview; opt="--renderer keiview --workers 3"; batches="100 200 300"; }
  for e in $batches; do
    local f=runs/eval/osm_sapporo/keipilot_${N}_${H}$([ $e = 100 ] || echo _ep$e)${rend}_pat$P
    [ -f $f.json ] || uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_$N/last.pt $L \
        --osm_town osm_sapporo --routes 20 --episode0 $e --ego_patience $P $opt --out $f.json || exit 1
  done
}
stage1() {
  for s in parked crosswalks crosswalks_sub bigjunction crosswalks_bk8 crosswalks_sub_bk8 bigjunction_bk8; do
    ev $s --suite $s --workers 6
  done
  sap keisim
}
ks() {      # the KeiSim town suites that stage 1 has not run
  for a in long3c long3d long3e long3f; do ev $a --suite $a --workers 6; ev ${a}_dense --suite $a --dense --workers 6; done
  ev ramp_routes --suite ramp_routes --workers 6
  ev ramps --suite ramps --workers 6
  ev suburban --suite suburban --workers 6
  ev ramp_routes_dense --suite ramp_routes --dense --workers 6
  ev suburban_dense --suite suburban --dense --workers 6
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
  ev ramp_routes_keiview --suite ramp_routes --renderer keiview --workers 3
  ev suburban_keiview --suite suburban --renderer keiview --workers 3
  sap keiview
}
if [ $STAGE -ge 1 ]; then
  for N in $MODELS; do (stage1) || exit 1; echo "R13F_STAGE1_${N^^}_DONE"; done
fi
if [ $STAGE -ge 2 ]; then
  [ -d web/node_modules ] && [ -d web/.towns ] || { echo "KeiView needs web/node_modules and web/.towns"; exit 1; }
  for N in $MODELS; do
    (ks) & k=$!
    (kv) & v=$!
    wait $k; a=$?
    wait $v; b=$?
    [ $a = 0 ] && [ $b = 0 ] || exit 1
    echo "R13F_STAGE2_${N^^}_DONE"
  done
fi
echo MC_ROUND13F_DONE
