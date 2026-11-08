#!/bin/bash
# After round 5's KeiView chain (runs/mc_work/round5.sh kv(): its last output is mc6's long3c dense in KeiView):
# 1. mc5b's KeiView Sapporo episode0 100 with _pat3 again (on 30 Oct, 05:59, a debugging replay took the GPU's memory
#    and the batch died with CUDA OOM; 200 and 300 are fine)
# 2. mc6's KeiView stands (ep211 j23 after a 止まれ, ep117 j35) replayed with frames (tools/replay_kv_freeze.py)
set -x
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=2
until [ -f runs/eval/keipilot_mc6_hold_red_long3c_dense_keiview.json ]; do
  systemctl --user is-active --quiet mc-round5 || break
  sleep 60
done
f=runs/eval/osm_sapporo/keipilot_mc5b_hold_red_keiview_pat3.json
[ -f $f ] || uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_mc5b/last.pt --brake_hold --red_hold \
    --osm_town osm_sapporo --routes 20 --episode0 100 --ego_patience 3 --renderer keiview --workers 3 --out $f
echo MC_R5FIX_KV_DONE
mkdir -p runs/mc_work/kv_freeze
uv run tools/replay_kv_freeze.py runs/keipilot_mc6/last.pt 211 3 17 26 runs/mc_work/kv_freeze/mc6_ep211 20 23 26
uv run tools/replay_kv_freeze.py runs/keipilot_mc6/last.pt 117 3 8 17 runs/mc_work/kv_freeze/mc6_ep117 11 13 17
echo MC_R5FIX_DONE
