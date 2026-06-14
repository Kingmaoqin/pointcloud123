from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def git_commit() -> str:
    return _run(["git", "rev-parse", "HEAD"])


def cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_cuda_available": False,
        "torch_device_count": 0,
        "nvidia_smi_available": False,
        "nvidia_smi_summary": _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
    }
    if not str(info["nvidia_smi_summary"]).startswith("UNAVAILABLE"):
        info["nvidia_smi_available"] = True
    try:
        import torch  # type: ignore

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_device_count"] = int(torch.cuda.device_count())
        if info["torch_device_count"]:
            info["torch_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_properties(i).name,
                    "total_memory_mib": torch.cuda.get_device_properties(i).total_memory // 1024 // 1024,
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def build_manifest(config: dict[str, Any], seed: int) -> dict[str, Any]:
    gpu = cuda_info()
    return {
        "seed": seed,
        "config": config,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": git_commit(),
        "cwd": os.getcwd(),
        "cuda_available": bool(gpu.get("torch_cuda_available") or gpu.get("nvidia_smi_available")),
        "gpu": gpu,
    }


def save_manifest(path: str | Path, config: dict[str, Any], seed: int) -> dict[str, Any]:
    manifest = build_manifest(config, seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest
