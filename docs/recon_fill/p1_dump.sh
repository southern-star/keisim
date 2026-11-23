#!/bin/bash
# ReconFill P1 (docs/recon_fill.md 5): the training data. One run of web/tools/fill_dump.mjs per line of the list below
# (NAME TOWN [options]), WORKERS headless Chromes at a time, into data/recon_fill/train and data/recon_fill/eval
# (not in git). A run whose meta.json says "done" is skipped, so the script can be started again after a stop.
#   bash docs/recon_fill/p1_dump.sh [train] [eval]        (from the repository root; both without arguments)
#   WORKERS=3 SEC=600 bash docs/recon_fill/p1_dump.sh
# The towns of the measurements (1000, 1015-98782ab7, osm_sapporo) and osm_nishioi, 1000-dae67d73 are not in the
# training list; the eval list is the three towns of the measurements.
set -o pipefail
cd "$(dirname "$0")/../../web" || exit 1
O=../data/recon_fill
L=../runs/eval/recon_fill/p1_dump
SEC=${SEC:-600}
STEPS=${*:-train eval}
mkdir -p $L

TRAIN="
1001 1001
1002 1002 --bikes 8
1003 1003 --parked 8
1018-98782ab7 1018-98782ab7 --bikes 6 --parked 6
2000-8c5d282b 2000-8c5d282b
2002-8c5d282b 2002-8c5d282b --bikes 8
5 5 --tdir .towns
11 11 --tdir .towns --bikes 8
23 23 --tdir .towns --parked 8
25 25 --tdir .towns --bikes 6 --parked 6
37 37 --tdir .towns
3-98782ab7 3-98782ab7 --tdir .towns --bikes 8
4-98782ab7 4-98782ab7 --tdir .towns --parked 8
7-98782ab7 7-98782ab7 --tdir .towns --bikes 6 --parked 6
14-98782ab7 14-98782ab7 --tdir .towns
1016-98782ab7 1016-98782ab7 --tdir .towns --bikes 8
3-8c5d282b 3-8c5d282b --tdir .towns --parked 8
6-8c5d282b 6-8c5d282b --tdir .towns --bikes 6 --parked 6
13-8c5d282b 13-8c5d282b --tdir .towns
2005-8c5d282b 2005-8c5d282b --tdir .towns --bikes 8
2008-8c5d282b 2008-8c5d282b --tdir .towns --parked 8
12-7eae7d60 12-7eae7d60 --tdir .towns --bikes 6 --parked 6
15-7eae7d60 15-7eae7d60 --tdir .towns
21-7eae7d60 21-7eae7d60 --tdir .towns --bikes 8
3004-7eae7d60 3004-7eae7d60 --tdir .towns --parked 8
3005-7eae7d60 3005-7eae7d60 --tdir .towns --bikes 6 --parked 6
7-ed02ebe7 7-ed02ebe7 --tdir .towns
36-ed02ebe7 36-ed02ebe7 --tdir .towns --bikes 8
47-ed02ebe7 47-ed02ebe7 --tdir .towns --parked 8
2004-ed02ebe7 2004-ed02ebe7 --tdir .towns --bikes 6 --parked 6
2005-ed02ebe7 2005-ed02ebe7 --tdir .towns
11-dae67d73 11-dae67d73 --tdir .towns --bikes 8
30-dae67d73 30-dae67d73 --tdir .towns --parked 8
71-dae67d73 71-dae67d73 --tdir .towns --bikes 6 --parked 6
osm_sapporo_n_s2000 osm_sapporo_n_s2000 --tdir .towns
osm_sapporo_n_s3001 osm_sapporo_n_s3001 --tdir .towns
osm_sapporo_e_s2100 osm_sapporo_e_s2100 --tdir .towns
osm_sapporo_e_s3102 osm_sapporo_e_s3102 --tdir .towns
osm_sapporo_s_s2201 osm_sapporo_s_s2201 --tdir .towns
osm_sapporo_s_s3200 osm_sapporo_s_s3200 --tdir .towns
"
EVAL="
1000 1000
1015-98782ab7 1015-98782ab7
osm_sapporo osm_sapporo
1000_bikes 1000 --bikes 8 --parked 8
"

one() {   # one SPLIT SEC NAME TOWN [options]
  local split=$1 sec=$2 name=$3 town=$4; shift 4
  if grep -q '"done": true' $O/$split/$name/meta.json 2>/dev/null; then echo "$split $name: done before"; return; fi
  node tools/fill_dump.mjs --out $O/$split --name $name --town $town --sec $sec --every 1 --preview 2 "$@" 2>&1 | grep -v '^\[warn\]' > $L/${split}_$name.log
  echo "$split $name: $(tail -n 1 $L/${split}_$name.log | sed 's/^[^:]*: //' | cut -c1-110)"
}
export -f one
export O L
if [[ " $STEPS " == *" train "* ]]; then
  echo "$TRAIN" | grep . | xargs -P ${WORKERS:-3} -L 1 bash -c 'one train '$SEC' "$@"' _
fi
if [[ " $STEPS " == *" eval "* ]]; then
  echo "$EVAL" | grep . | xargs -P ${WORKERS:-3} -L 1 bash -c 'one eval 300 "$@"' _
fi
