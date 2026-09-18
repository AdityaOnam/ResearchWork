#!/bin/bash
# Cost benchmark (params, FLOPs, memory, latency). Requests the WHOLE GPU:
# bench.py refuses to time on a GPU shared with other processes.
#SBATCH --job-name=qssm_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/hpc/slurm/common.sh"
cd "$REPO/qssm/efficiency"
python bench.py --json "$QSSM_RESULTS_DIR/bench.json" --md "$QSSM_RESULTS_DIR/bench.md"
