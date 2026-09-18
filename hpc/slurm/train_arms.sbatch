#!/bin/bash
# Train the four arms of the 2x2 encoder factorial as ONE array job (A0..A3).
#   sbatch hpc/slurm/train_arms.sbatch            # all four
#   sbatch --array=3 hpc/slurm/train_arms.sbatch  # A3 (QSSM) only
#SBATCH --job-name=qssm_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/hpc/slurm/common.sh"

ARGS=(
  "--encoder vit"                  # A0  attention / none
  "--encoder hybrid"               # A1  attention / q-scan
  "--encoder mamba --no-qscan"     # A2  scan      / none
  "--encoder mamba"                # A3  scan      / q-scan   (QSSM)
)
cd "$REPO/qssm/train"
python train_mamba.py ${ARGS[$SLURM_ARRAY_TASK_ID]} --real-frac 0.7 --ssm-kernel auto
