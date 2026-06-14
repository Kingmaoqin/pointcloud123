#!/usr/bin/env bash
#SBATCH --job-name=gap-closed-loop
#SBATCH --output=outputs/logs/closed-loop-%j.out
#SBATCH --error=outputs/logs/closed-loop-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
trap 'echo "SIGTERM received during closed loop; outputs/checkpoints retained"' TERM
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p outputs/logs outputs/checkpoints
hostname
nvidia-smi || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python -m patent_gap.cli closed-loop --config configs/experiment/synthetic_smoke.yaml simulation.steps=10

