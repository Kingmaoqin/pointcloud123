# STATUS

## 2026-06-14

- Created project skeleton under `/home/xqin5/patent_gap_nbv`.
- Read patent PDF and extracted the algorithmic requirements into implementation modules.
- Downloaded CRAS IFC file to `data/raw/craslabbim.ifc`.
- Started CRAS annotated point cloud ZIP download with resume support.
- Implemented first-pass synthetic cube pipeline, tests, CLI, reports, and Slurm scripts.
- EDF Challenge requires user login; the project continues with CRAS and synthetic experiments.

## Known Environment Limits

- Slurm commands `sbatch` and `srun` were not found on the current node.
- `nvidia-smi` cannot communicate with the NVIDIA driver on the current node.
- Base Python lacks optional full-processing packages including `open3d`, `ifcopenshell`, and `optuna`.

