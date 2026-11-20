#!/bin/bash
# The future head (docs/future_aux.md), phase 2: short runs from mc13 on mc14's data and weights, to choose
# --future_weight from the validation numbers alone (no closed-loop evaluation) and to time the training.
#   fw0  no head (the baseline)            fw01 / fw03 / fw10  --future_weight 0.1 / 0.3 / 1.0
#   fw0s1  no head, another random stream (--seed 1 --split_seed 0: the same frames): how far two runs of the
#          same setting are apart
# Each: round13.sh's train.py command for mc14 (its data, sampling weights and options, read from mc14's
# checkpoint), EPOCHS x SAMPLES samples instead of 4 x 200k. About 12 minutes a run on the RTX 3060.
# Every step skips what is already there.
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
O=runs/future_aux
run() {     # run NAME ARGS...: one short training run, once
  local n=$1; shift
  [ -f $O/$n/last.pt ] || uv run scripts/train.py --data $D --init $M13 --out $O/$n --epochs ${EPOCHS:-2} \
      --samples_per_epoch ${SAMPLES:-50000} $T --dir_weight $W "$@" || exit 1
  echo FA2_${n}_DONE
}
run fw0
run fw03 --future_weight 0.3
run fw01 --future_weight 0.1
run fw10 --future_weight 1.0
run fw0s1 --seed 1 --split_seed 0
echo FA_PHASE2_DONE
