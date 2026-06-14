#!/usr/bin/env bash
#SBATCH --job-name=gap-preprocess
#SBATCH --output=outputs/logs/preprocess-%j.out
#SBATCH --error=outputs/logs/preprocess-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
trap 'echo "SIGTERM received; checkpoint directory: outputs/checkpoints"' TERM
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p outputs/logs outputs/checkpoints
hostname
nvidia-smi || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python -m patent_gap.cli preprocess --config configs/experiment/cras_smoke.yaml seed="${SLURM_ARRAY_TASK_ID:-0}"

