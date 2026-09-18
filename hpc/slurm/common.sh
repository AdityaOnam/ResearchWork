# Sourced by every job in this folder. Edit once for your cluster.
#   - activates the Python environment
#   - exports the QSSM path variables (see qssm/paths.py)
#   - resolves REPO so jobs can be submitted from anywhere

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Python environment (venv or conda). Edit one of these:
# source /path/to/venv/bin/activate
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate qssm

export QSSM_DATA_ROOT="${QSSM_DATA_ROOT:-/path/to/data}"
export QSSM_CKPT_DIR="${QSSM_CKPT_DIR:-$REPO/checkpoints}"
export QSSM_RESULTS_DIR="${QSSM_RESULTS_DIR:-$REPO/results}"
export QSSM_SPLITS_DIR="${QSSM_SPLITS_DIR:-$REPO/splits}"

# Subject N of a split file (1-based), for array jobs: subject_at test.txt 3
subject_at() { sed 's/#.*//; /^[[:space:]]*$/d' "$QSSM_SPLITS_DIR/$1" | sed -n "${2}p"; }

mkdir -p "$REPO/logs"
echo "[$(date '+%F %T')] job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:--} on $(hostname)"
