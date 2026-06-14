# Failures and Repairs

## Missing Slurm commands

- Symptom: `which sbatch` and `which srun` returned no command.
- Repair attempt 1: Generated Slurm scripts without hard-coded partition names so they can be used on a Slurm-enabled node.
- Repair attempt 2: Kept all CLI workflows runnable locally with plain Python.

## GPU unavailable on current node

- Symptom: `nvidia-smi` failed because it could not communicate with the NVIDIA driver.
- Repair attempt 1: Implemented the first version using CPU geometry and NumPy.
- Repair attempt 2: Kept GPU-dependent packages optional and recorded CUDA availability in the reproducibility manifest.

## Optional geometry packages missing in base Python

- Symptom: `open3d`, `ifcopenshell`, `optuna`, `torch`, `trimesh`, `laspy`, `shapely`, and `pynvml` were not all importable in base Python.
- Repair attempt 1: Implemented synthetic ray/visibility logic without these packages.
- Repair attempt 2: IFC inspection falls back to STEP text audit when `ifcopenshell` is unavailable and records that triangulation was not performed.

