# Cluster Environment

Recorded on 2026-06-14 in `/home/xqin5/patent_gap_nbv`.

## Commands

- `pwd`: `/home/xqin5`
- `hostname`: `dhai.bme.e.uh.edu`
- `uname -a`: `Linux dhai.bme.e.uh.edu 5.15.0-179-generic #189-Ubuntu SMP Tue May 5 18:20:56 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux`
- `python --version`: `Python 3.12.7`
- `which python`: `/opt/anaconda3/bin/python`
- `which conda`: `/opt/anaconda3/bin/conda`
- `which mamba`: not found
- `which sbatch`: not found
- `which srun`: not found
- `nvidia-smi` with elevated GPU-driver access: NVIDIA driver `580.159.03`, CUDA `13.0`
- `git --version`: `git version 2.34.1`
- `gcc --version`: `gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0`
- `quota -s`: `quota: command not found`

## Resources

- CPU cores: 48
- Memory: 503 GiB total, 452 GiB available at inspection time
- Filesystem for `/home/xqin5` and `/tmp`: 6.9 TiB total, 1.9 TiB available
- Slurm: not detected on this node
- GPU hardware: 4 x NVIDIA A100 80GB PCIe
- GPU memory at 2026-06-14 16:25:50:
  - GPU 0: 74239 MiB / 81920 MiB used, occupied mainly by `VLLM::EngineCore`
  - GPU 1: 74239 MiB / 81920 MiB used, occupied mainly by `VLLM::EngineCore`
  - GPU 2: 76335 MiB / 81920 MiB used, occupied mainly by `VLLM::EngineCore`
  - GPU 3: 14 MiB / 81920 MiB used, effectively free
- PyTorch CUDA check in `/home/xqin5/.conda/envs/MDPC/bin/python`: `torch 2.6.0+cu124`, CUDA available, 4 devices visible
- SCRATCH/WORK/PROJECT variables: none detected
- Data root selected: `/home/xqin5/patent_gap_nbv/data`

## Python Package Check

Base Python had `numpy`, `scipy`, `pandas`, `sklearn`, `matplotlib`, `yaml`, `tqdm`, `rich`, `pytest`, `plyfile`, `networkx`, `rtree`, `joblib`, `pyarrow`, `jinja2`, and `psutil`.

Missing in base Python at initial inspection: `hydra`, `omegaconf`, `open3d`, `ifcopenshell`, `trimesh`, `laspy`, `shapely`, `torch`, `torchvision`, `optuna`, `tensorboard`, and `pynvml`.

Installed after GPU/resource correction: `ifcopenshell==0.8.5`, `open3d==0.19.0`, `optuna`, `laspy`, `shapely`, and `trimesh`.

The `mdbimdt_baselines` conda environment had `open3d`, `trimesh`, and `torch`, but not `ifcopenshell` or `optuna`.

The `MDPC` conda environment has PyTorch CUDA support and can see all four A100 GPUs.

## Notes

The first implementation is CPU-only and does not require GPU access. The earlier `nvidia-smi` failure was due to default sandbox access, not missing hardware. CRAS IFC triangulation now succeeds in the active runtime after installing `ifcopenshell`.
