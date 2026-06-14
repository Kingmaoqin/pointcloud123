from __future__ import annotations

from typing import Any

import pandas as pd

from patent_gap.simulation.synthetic_cube import synthetic_scene


def apply_synthetic_corruption(scene: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Apply deterministic, spatially meaningful corruption for the smoke scene."""
    corrupted = {k: v.copy() for k, v in scene.items()}
    seed = int(config.get("seed", 0))
    if not config.get("corruptions"):
        return corrupted
    obs = corrupted["observations"]
    mats = corrupted["materials"]
    sem = corrupted["semantics"]
    for item in config.get("corruptions", []):
        ctype = item.get("type")
        if ctype == "contiguous_patch_removal":
            obs.loc[obs["patch_id"] == 0, ["directly_observed", "number_of_views", "number_of_valid_views", "coverage_ratio", "density_ratio"]] = [False, 0, 0, 0.0, 0.0]
        elif ctype == "quality_degradation":
            obs.loc[obs["patch_id"] == 4, ["coverage_ratio", "density_ratio"]] = [0.35, 0.20]
        elif ctype == "semantic_label_swap":
            sem.loc[sem["patch_id"] == 1, "obs_label"] = "slab"
        elif ctype == "material_missing":
            mats.loc[mats["patch_id"] == 2, ["material_present", "material_name", "material_category", "source", "confidence"]] = [False, None, "unknown", "UNKNOWN", 0.0]
        elif ctype == "material_conflict":
            mats.loc[mats["patch_id"] == 1, ["visual_material_category", "material_conflict"]] = ["concrete", True]
    corrupted["observations"] = obs
    corrupted["materials"] = mats
    corrupted["semantics"] = sem
    corrupted["seed"] = pd.DataFrame([{"seed": seed}])
    return corrupted


def default_corrupted_scene(seed: int = 0) -> dict[str, pd.DataFrame]:
    return synthetic_scene(seed)

