#!/bin/bash
# Shell extraction, one subject per array task (CPU only).
#   N=$(grep -cv '^\s*\(#\|$\)' splits/all.txt)
#   sbatch --array=1-$N hpc/slurm/preprocess_array.sbatch
#SBATCH --job-name=qssm_shells
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/hpc/slurm/common.sh"
SID="$(subject_at all.txt "$SLURM_ARRAY_TASK_ID")"
python "$REPO/hpc/preprocessing/shell_extractor.py" \
  --base-dir "${RAW_HCP_ROOT:-$QSSM_DATA_ROOT/raw}" --subject "$SID"
