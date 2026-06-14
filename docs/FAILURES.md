# Failures and Repairs

## Missing Slurm commands

- Symptom: `which sbatch` and `which srun` returned no command.
- Repair attempt 1: Generated Slurm scripts without hard-coded partition names so they can be used on a Slurm-enabled node.
- Repair attempt 2: Kept all CLI workflows runnable locally with plain Python.

## Default sandbox GPU-driver access failed

- Symptom: `nvidia-smi` failed because it could not communicate with the NVIDIA driver.
- Repair attempt 1: Re-ran `nvidia-smi` with elevated GPU-driver access after user authorization.
- Repair attempt 2: Verified four NVIDIA A100 80GB GPUs and PyTorch CUDA availability in the `MDPC` environment.
- Current status: GPU hardware is available. The first implementation remains CPU-first because deep models are optional for the patent baseline.

## Optional geometry packages missing in base Python

- Symptom: `open3d`, `ifcopenshell`, `optuna`, `torch`, `trimesh`, `laspy`, `shapely`, and `pynvml` were not all importable in base Python.
- Repair attempt 1: Implemented synthetic ray/visibility logic without these packages.
- Repair attempt 2: IFC inspection falls back to STEP text audit when `ifcopenshell` is unavailable and records that triangulation was not performed.
