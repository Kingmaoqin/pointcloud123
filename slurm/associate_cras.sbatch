#!/usr/bin/env bash
#SBATCH --job-name=gap-assoc-cras
#SBATCH --output=outputs/logs/associate-cras-%j.out
#SBATCH --error=outputs/logs/associate-cras-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
trap 'echo "SIGTERM received; CRAS association summaries are resumable under data/processed/cras_full_assoc/parts"' TERM
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p outputs/logs data/processed
hostname
nvidia-smi || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
python -m patent_gap.cli associate-cras --config configs/experiment/cras_full.yaml

