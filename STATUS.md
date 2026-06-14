# STATUS

## 2026-06-14

- Created project skeleton under `/home/xqin5/patent_gap_nbv`.
- Read patent PDF and extracted the algorithmic requirements into implementation modules.
- Downloaded CRAS IFC file to `data/raw/craslabbim.ifc`.
- Started CRAS annotated point cloud ZIP download with resume support.
- Implemented first-pass synthetic cube pipeline, tests, CLI, reports, and Slurm scripts.
- EDF Challenge requires user login; the project continues with CRAS and synthetic experiments.
- CRAS point cloud ZIP downloaded successfully and MD5 matched.
- CRAS ZIP contains one fused `CRASLAB_annotated.asc` file with `584701979` total lines from a full stream count.
- GPU access rechecked with elevated permission: 4 x NVIDIA A100 80GB PCIe are visible; GPU 3 was effectively free at inspection time.
- `/home/xqin5/.conda/envs/MDPC/bin/python` has `torch 2.6.0+cu124` with CUDA available and 4 visible devices.
- Installed `ifcopenshell==0.8.5`, `open3d==0.19.0`, `optuna`, `laspy`, `shapely`, and `trimesh` in the active base Python user site.
- CRAS IFC triangulation succeeded: 256 elements, 604187 vertices, 1197750 triangles, 24 cached materials, 0 geometry failures.
- CRAS 20000-point ASC smoke association succeeded after robust translation calibration: 10154 matched within 0.05 m, matched ratio 0.5077.
- CRAS full ASC association completed with Open3D closest-point backend: 584701977 valid points, 12835294 matched within 0.05 m, matched ratio 0.0219518567, 585 chunk summaries, runtime 2491.1 s.
- Synthetic smoke and held-out tests completed; outputs written under `outputs/`.
- `pytest -q` passed with 12 tests.
- Git initial implementation commit: `cf33eb5`.
- Latest source commit after full CRAS association pipeline: `7e661dc`.

## Known Environment Limits

- Slurm commands `sbatch` and `srun` were not found on the current node.
- Default sandboxed `nvidia-smi` could not access the driver, but elevated access confirms GPUs are available.
- Base Python still lacks `torch`; use `MDPC` for CUDA PyTorch tasks.
