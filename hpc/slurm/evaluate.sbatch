#!/bin/bash
# Score every arm checkpoint on the held-out test split, then the noise and
# direction sweeps, then the statistics. Submit after train_arms finishes:
#   sbatch --dependency=afterok:<train_job_id> hpc/slurm/evaluate.sbatch
#SBATCH --job-name=qssm_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/hpc/slurm/common.sh"

cd "$REPO/qssm/evaluate"
python evaluate.py --all --save-maps
python evaluate_snr.py --all
python evaluate_directions.py --all
cd "$REPO/qssm/stats"
python stats.py --json "$QSSM_RESULTS_DIR/stats.json" --md "$QSSM_RESULTS_DIR/stats.md"
