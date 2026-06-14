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
- `nvidia-smi`: failed, NVIDIA driver not reachable from this node
- `git --version`: `git version 2.34.1`
- `gcc --version`: `gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0`
- `quota -s`: `quota: command not found`

## Resources

- CPU cores: 48
- Memory: 503 GiB total, 452 GiB available at inspection time
- Filesystem for `/home/xqin5` and `/tmp`: 6.9 TiB total, 1.9 TiB available
- Slurm: not detected on this node
- GPU: not usable from this node because `nvidia-smi` cannot communicate with the driver
- SCRATCH/WORK/PROJECT variables: none detected
- Data root selected: `/home/xqin5/patent_gap_nbv/data`

## Python Package Check

Base Python had `numpy`, `scipy`, `pandas`, `sklearn`, `matplotlib`, `yaml`, `tqdm`, `rich`, `pytest`, `plyfile`, `networkx`, `rtree`, `joblib`, `pyarrow`, `jinja2`, and `psutil`.

Missing in base Python at initial inspection: `hydra`, `omegaconf`, `open3d`, `ifcopenshell`, `trimesh`, `laspy`, `shapely`, `torch`, `torchvision`, `optuna`, `tensorboard`, and `pynvml`.

The `mdbimdt_baselines` conda environment had `open3d`, `trimesh`, and `torch`, but not `ifcopenshell` or `optuna`.

## Notes

The first implementation is CPU-only and does not require GPU access. Full CRAS IFC triangulation remains blocked until `ifcopenshell` is installed in the active runtime.

