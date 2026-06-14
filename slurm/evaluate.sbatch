#!/usr/bin/env bash
#SBATCH --job-name=gap-eval
#SBATCH --output=outputs/logs/evaluate-%j.out
#SBATCH --error=outputs/logs/evaluate-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
trap 'echo "SIGTERM received during evaluation"' TERM
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p outputs/logs
hostname
nvidia-smi || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python -m patent_gap.cli evaluate --config configs/experiment/heldout_test.yaml

