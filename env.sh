# ---------------------------------------------------------------------------
# Edit for your machine, then `source env.sh` before anything else.
# ---------------------------------------------------------------------------

# Repository root (auto-detected)
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Raw datasets: folder that directly contains the domain folders
#   PACS:        $DATA_ROOT_PACS/{photo,art_painting,cartoon,sketch}/<class>/*.jpg
#   Office-Home: $DATA_ROOT_OFFICEHOME/{Art,Clipart,Product,"Real World"}/<class>/*.jpg
export DATA_ROOT_PACS=$PROJECT_DIR/data/pacs
export DATA_ROOT_OFFICEHOME=$PROJECT_DIR/data/officehome

# Outputs
export CACHE_DIR=$PROJECT_DIR/cache          # ~4 GB of uint8 tensors
export CKPT_DIR=$PROJECT_DIR/checkpoints     # source models, ~45 MB each
export RESULTS_DIR=$PROJECT_DIR/results      # one small JSON per run
export TORCH_HOME=$PROJECT_DIR/.torch        # ImageNet weights cache

# Python environment: set CONDA_ENV to a conda env name, or activate a venv yourself.
#   export CONDA_ENV=myenv
if [ -n "${CONDA_ENV:-}" ]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" && conda activate "$CONDA_ENV"
fi
# source $PROJECT_DIR/.venv/bin/activate

# SLURM settings used by slurm/submit_all.sh (check `sinfo -o "%P %G"`)
export SLURM_PARTITION=l40
export SLURM_QOS=l40                # some clusters require --qos equal to the partition name
export SLURM_GRES="gpu:1"
export SLURM_TIME_GPU="02:00:00"    # per array task (~1 min per run)
export CHUNK=${CHUNK:-6}            # runs per array task
export PAR=${PAR:-3}                # runs executed concurrently on the one GPU
