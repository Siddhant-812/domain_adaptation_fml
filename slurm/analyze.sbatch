#!/bin/bash
#SBATCH --job-name=cda_analyze
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=l40
#SBATCH --qos=l40
# Submit from the repository root.
set -Eeuo pipefail
PROJ=${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$PROJ"
source "$PROJ/env.sh"
python src/analyze.py --manifest jobs/manifest.json --out analysis
