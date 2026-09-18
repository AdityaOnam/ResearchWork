#!/bin/bash
# Adaptive-compute controller: warm-up, then PPO fine-tuning of the K-predictor.
#   sbatch hpc/slurm/adaptive_compute.sbatch
#SBATCH --job-name=qssm_adaptive
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/hpc/slurm/common.sh"
cd "$REPO/hpc/adaptive_compute"
python train_warmup.py          # reads config/default_config.yaml
python train_rl_model.py --config config/default_config.yaml
