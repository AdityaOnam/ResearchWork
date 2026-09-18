#!/bin/bash
# Free-water-corrected tractography + TractSeg, one subject per array task.
#   N=$(grep -cv '^\s*\(#\|$\)' splits/test.txt)
#   DWI_BASE=... FW_BASE=... sbatch --array=1-$N hpc/slurm/tractseg_array.sbatch
#SBATCH --job-name=qssm_tractseg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/hpc/slurm/common.sh"
SID="$(subject_at test.txt "$SLURM_ARRAY_TASK_ID")"
export NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
bash "$REPO/tractography/ss3t_tractseg/run_fw_tractseg.sh" "$SID"
