from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." not in value:
            return int(value)
        return float(value)
    except ValueError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        target = merged
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _coerce_value(raw)
    return merged


def ensure_output_dirs(root: str | Path = "outputs") -> dict[str, Path]:
    root = Path(root)
    dirs = {
        "root": root,
        "logs": root / "logs",
        "figures": root / "figures",
        "tables": root / "tables",
        "reports": root / "reports",
        "checkpoints": root / "checkpoints",
        "optuna": root / "optuna",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs

