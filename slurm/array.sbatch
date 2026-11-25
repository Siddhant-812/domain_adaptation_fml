#!/bin/bash
#SBATCH --job-name=cda
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Defaults above; submit_all.sh overrides partition/qos/gres/time/array on the command line.
# Submit from the repository root.
#
# Usage: sbatch --array=1-N slurm/array.sbatch <jobfile>
# Task i runs lines (i-1)*CHUNK+1 .. i*CHUNK of <jobfile>, PAR at a time on one GPU.
set -Eeuo pipefail

PROJ=${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}
JOBFILE=${1:?usage: sbatch --array=1-N slurm/array.sbatch <jobfile>}
CHUNK=${CHUNK:-1}
PAR=${PAR:-1}

cd "$PROJ"
source "$PROJ/env.sh"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / PAR > 0 ? ${SLURM_CPUS_PER_TASK:-8} / PAR : 1 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p logs "$RESULTS_DIR" "$CKPT_DIR"

start=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
end=$(( start + CHUNK - 1 ))

echo "host: $(hostname)  job: ${SLURM_ARRAY_JOB_ID:-interactive}_${SLURM_ARRAY_TASK_ID}"
echo "jobfile: $JOBFILE  lines $start-$end  PAR=$PAR"
python - <<'PY'
import torch
print("torch:", torch.__version__)
assert torch.cuda.is_available(), "no usable GPU allocated"
print("GPU:", torch.cuda.get_device_name(0))
PY
for f in "$JOBFILE" src/adapt.py src/train_source.py; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 2; }
done

# Each run prints "run <run_id>" first; xargs exits non-zero if any run failed.
sed -n "${start},${end}p" "$JOBFILE" | xargs -P "$PAR" -I{} bash -c 'python {}'
echo "finished: $(date --iso-8601=seconds)"
