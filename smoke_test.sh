#!/bin/bash
# End-to-end check on a tiny synthetic dataset (CPU or GPU, ~5-15 min on CPU).
# Uses its own scratch directories, so it never touches real results.
set -euo pipefail
cd "$(dirname "$0")"
SMOKE=${SMOKE_DIR:-/tmp/cda_smoke}
rm -rf "$SMOKE"
export DATA_ROOT_FAKE=$SMOKE/data CACHE_DIR=$SMOKE/cache CKPT_DIR=$SMOKE/ckpt RESULTS_DIR=$SMOKE/results

python src/make_fake_data.py "$DATA_ROOT_FAKE"
python src/data.py --dataset fake --workers 2
python src/make_jobs.py --smoke --outdir "$SMOKE/jobs"
while read -r line; do python $line; done < "$SMOKE/jobs/source.txt"
while read -r line; do python $line > /dev/null; done < "$SMOKE/jobs/adapt.txt"
python src/make_jobs.py --smoke --outdir "$SMOKE/jobs" --only-missing   # should report 0 adapt jobs
python src/analyze.py --manifest "$SMOKE/jobs/manifest.json" --out "$SMOKE/analysis"
echo "SMOKE TEST PASSED. Outputs in $SMOKE/analysis"
