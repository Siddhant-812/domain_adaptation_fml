#!/bin/bash
#SBATCH --job-name=cda_prep
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# CPU-only: decode all images once into uint8 tensors. Submit from the repository root.
set -Eeuo pipefail
PROJ=${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$PROJ"
source "$PROJ/env.sh"
export PYTHONUNBUFFERED=1
python src/data.py --dataset pacs --workers 16
python src/data.py --dataset officehome --workers 16
