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


def build_manifest(config: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "config": config,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": git_commit(),
        "cwd": os.getcwd(),
        "cuda_available": False,
    }


def save_manifest(path: str | Path, config: dict[str, Any], seed: int) -> dict[str, Any]:
    manifest = build_manifest(config, seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest

