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
- Repair attempt 2: Installed `ifcopenshell==0.8.5`, `open3d==0.19.0`, `optuna`, `laspy`, `shapely`, and `trimesh`.
- Current status: CRAS IFC triangulation succeeds. Base Python still lacks `torch`; CUDA PyTorch is available in the `MDPC` environment.

## CRAS 200k-point exact proximity smoke was too slow

- Symptom: exact `trimesh.proximity.closest_point` association for 200000 CRAS points against 1197750 IFC triangles did not finish within the interactive smoke window.
- Repair attempt 1: Added IFC triangulation cache reuse so geometry is not rebuilt on every preprocess run.
- Repair attempt 2: Reduced `configs/experiment/cras_smoke.yaml` to 20000 points for smoke validation while keeping the point count configurable for larger batch jobs.
- Current status: full-scale association should be implemented as chunked spatial tiling or run as a longer batch job; smoke mode remains bounded.
