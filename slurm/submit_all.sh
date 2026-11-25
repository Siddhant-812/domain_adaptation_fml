#!/bin/bash
# Submit the whole pipeline: source models -> adaptation runs -> analysis.
# Adaptation runs are split round-robin over PARTITIONS (each partition's QoS has the same
# name), TASKS_PER_PART array tasks per partition, each running PAR runs at once on one GPU.
# Safe to re-run: only missing checkpoints / runs are submitted.
# Usage (from the repo root):  bash slurm/submit_all.sh
set -Eeuo pipefail
source env.sh
cd "$PROJECT_DIR"
mkdir -p logs

PARTITIONS=${PARTITIONS:-"l40 a40 dgx"}
TASKS_PER_PART=${TASKS_PER_PART:-3}

python src/make_jobs.py --only-missing

n_src=$(wc -l < jobs/source.txt)
n_adapt=$(wc -l < jobs/adapt.txt)
dep=()

if [ "$n_src" -gt 0 ]; then
  SRC=$(sbatch --parsable --partition="$SLURM_PARTITION" --qos="$SLURM_QOS" --gres="$SLURM_GRES" \
        --time="$SLURM_TIME_GPU" --job-name=cda_src --array=1-"$n_src" \
        --export=ALL,CHUNK=1,PAR=1 slurm/array.sbatch jobs/source.txt)
  echo "source training: job $SRC ($n_src tasks on $SLURM_PARTITION)"
  dep=(--dependency=afterok:"$SRC")
fi

adapt_ids=()
if [ "$n_adapt" -gt 0 ]; then
  read -ra PARTS <<< "$PARTITIONS"
  np=${#PARTS[@]}
  for i in "${!PARTS[@]}"; do
    p=${PARTS[$i]}
    f=jobs/adapt_$p.txt
    awk -v n="$np" -v k="$i" '(NR-1)%n==k' jobs/adapt.txt > "$f"
    n=$(wc -l < "$f")
    [ "$n" -gt 0 ] || continue
    chunk=$(( (n + TASKS_PER_PART - 1) / TASKS_PER_PART ))
    tasks=$(( (n + chunk - 1) / chunk ))
    J=$(sbatch --parsable --partition="$p" --qos="$p" --gres="$SLURM_GRES" --time="$SLURM_TIME_GPU" \
        "${dep[@]}" --job-name=cda_adapt_"$p" --array=1-"$tasks" \
        --export=ALL,CHUNK="$chunk",PAR="$PAR" slurm/array.sbatch "$f")
    echo "adaptation on $p: job $J ($n runs, $tasks tasks x $chunk runs, PAR=$PAR)"
    adapt_ids+=("$J")
  done
  dep=(--dependency=afterany:$(IFS=:; echo "${adapt_ids[*]}"))
fi

AN=$(sbatch --parsable --partition="$SLURM_PARTITION" --qos="$SLURM_QOS" "${dep[@]}" slurm/analyze.sbatch)
echo "analysis: job $AN (runs after adaptation finishes)"
echo "monitor with: squeue -u \$USER ; tail -f logs/*.out"
