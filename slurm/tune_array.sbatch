#!/usr/bin/env bash
#SBATCH --job-name=gap-tune
#SBATCH --output=outputs/logs/tune-%A-%a.out
#SBATCH --error=outputs/logs/tune-%A-%a.err
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
trap 'echo "SIGTERM received; worker checkpoint directory: outputs/checkpoints"' TERM
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p outputs/logs outputs/checkpoints outputs/optuna
hostname
nvidia-smi || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -m patent_gap.cli tune --config configs/tuning/optuna.yaml seed="${SLURM_ARRAY_TASK_ID:-0}" trials=30

