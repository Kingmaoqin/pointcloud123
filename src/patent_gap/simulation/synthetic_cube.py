from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SurfacePatch:
    patch_id: int
    element_guid: str
    ifc_class: str
    name: str
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]
    area: float
    material_name: str | None
    engineering_importance: float
    gt_missing: bool = False


def _one_hot(label: str, labels: list[str]) -> np.ndarray:
    arr = np.zeros(len(labels), dtype=float)
    arr[labels.index(label)] = 1.0
    return arr


def create_cube_patches(seed: int = 0) -> pd.DataFrame:
    del seed
    rows = [
        SurfacePatch(0, "GUID_OCCLUDED_WALL", "IfcWall", "+X wall occluded", (1, 0, 0), (1, 0, 0), 4.0, "paint", 1.0, True),
        SurfacePatch(1, "GUID_SEM_PIPE", "IfcFlowSegment", "-X semantic conflict", (-1, 0, 0), (-1, 0, 0), 4.0, "steel", 0.9, False),
        SurfacePatch(2, "GUID_MAT_PANEL", "IfcDoor", "+Y material missing", (0, 1, 0), (0, 1, 0), 4.0, None, 0.7, False),
        SurfacePatch(3, "GUID_GRAZING_SLAB", "IfcSlab", "-Y grazing angle", (0, -1, 0), (0, -1, 0), 4.0, "concrete", 0.6, False),
        SurfacePatch(4, "GUID_LOW_DENSITY", "IfcWall", "+Z low density", (0, 0, 1), (0, 0, 1), 4.0, "paint", 0.8, False),
        SurfacePatch(5, "GUID_OK_FLOOR", "IfcSlab", "-Z nominal", (0, 0, -1), (0, 0, -1), 4.0, "concrete", 0.5, False),
    ]
    return pd.DataFrame([p.__dict__ for p in rows])


def create_existing_views() -> pd.DataFrame:
    positions = [
        (3.0, 0.0, 0.0),
        (-3.0, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        (0.0, -3.0, 0.0),
        (0.0, 0.0, 3.0),
        (0.0, 0.0, -3.0),
        (2.5, 2.5, 1.5),
        (-2.5, -2.5, 1.5),
    ]
    rows = []
    for i, pos in enumerate(positions):
        p = np.asarray(pos, dtype=float)
        direction = -p / (np.linalg.norm(p) + 1e-12)
        rows.append({"view_id": f"existing_{i}", "position": tuple(p), "orientation": tuple(direction)})
    return pd.DataFrame(rows)


def _frontality(patch_centroid: Iterable[float], normal: Iterable[float], position: Iterable[float]) -> float:
    c = np.asarray(tuple(patch_centroid), dtype=float)
    n = np.asarray(tuple(normal), dtype=float)
    v = np.asarray(tuple(position), dtype=float) - c
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(max(0.0, np.dot(n, v)))


def build_observation_table(patches: pd.DataFrame, views: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for patch in patches.itertuples(index=False):
        visible_count = 0
        best_frontality = 0.0
        frontality_values = []
        for view in views.itertuples(index=False):
            f = _frontality(patch.centroid, patch.normal, view.position)
            if patch.patch_id == 0:
                visible = False
            elif patch.patch_id == 3:
                visible = f > 0.05
                f = min(f, 0.18)
            else:
                visible = f > 0.45
            if visible:
                visible_count += 1
                best_frontality = max(best_frontality, f)
                frontality_values.append(f)
        if patch.patch_id == 4:
            density_ratio = 0.20
            coverage_ratio = 0.35
            point_count = 80
        elif patch.patch_id == 0:
            density_ratio = 0.0
            coverage_ratio = 0.0
            point_count = 0
        else:
            density_ratio = float(np.clip(0.85 + rng.normal(0, 0.03), 0.65, 1.0))
            coverage_ratio = float(np.clip(0.90 + rng.normal(0, 0.03), 0.70, 1.0))
            point_count = int(500 * patch.area * density_ratio)
        rows.append(
            {
                "patch_id": patch.patch_id,
                "directly_observed": visible_count > 0 and point_count > 0,
                "number_of_views": visible_count,
                "number_of_valid_views": sum(v >= 0.45 for v in frontality_values),
                "best_frontality": best_frontality,
                "mean_frontality": float(np.mean(frontality_values)) if frontality_values else 0.0,
                "angular_diversity": min(1.0, visible_count / 3.0),
                "point_count": point_count,
                "point_density": point_count / patch.area,
                "coverage_ratio": coverage_ratio,
                "density_ratio": density_ratio,
                "projected_resolution_quality": 0.8 if visible_count else 0.0,
                "registration_confidence": 0.95 if visible_count else 0.2,
                "source_type": "MEASURED" if visible_count else "MODEL",
            }
        )
    return pd.DataFrame(rows)


def build_semantic_tables(patches: pd.DataFrame) -> pd.DataFrame:
    labels = ["wall", "pipe", "door", "slab", "unknown"]
    ifc_to_label = {
        "IfcWall": "wall",
        "IfcFlowSegment": "pipe",
        "IfcDoor": "door",
        "IfcSlab": "slab",
    }
    rows = []
    for patch in patches.itertuples(index=False):
        bim_label = ifc_to_label.get(patch.ifc_class, "unknown")
        obs_label = bim_label
        if patch.patch_id == 1:
            obs_label = "slab"
        if patch.patch_id == 0:
            obs_dist = np.full(len(labels), np.nan)
        else:
            obs_dist = 0.10 / (len(labels) - 1) * np.ones(len(labels))
            obs_dist[labels.index(obs_label)] = 0.90
        rows.append(
            {
                "patch_id": patch.patch_id,
                "semantic_labels": labels,
                "bim_label": bim_label,
                "obs_label": obs_label if patch.patch_id != 0 else None,
                "p_bim": _one_hot(bim_label, labels),
                "p_obs": obs_dist,
            }
        )
    return pd.DataFrame(rows)


def build_material_table(patches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patch in patches.itertuples(index=False):
        material_missing = pd.isna(patch.material_name)
        material_name = None if material_missing else patch.material_name
        visual_category = material_name or "unknown"
        conflict = False
        if patch.patch_id == 1:
            visual_category = "concrete"
            conflict = True
        rows.append(
            {
                "patch_id": patch.patch_id,
                "material_present": not material_missing,
                "material_name": material_name,
                "material_category": material_name or "unknown",
                "visual_material_category": visual_category,
                "material_conflict": conflict,
                "color_mean_r": 0.6,
                "color_mean_g": 0.6,
                "color_mean_b": 0.6,
                "source": "IFC" if not material_missing else "UNKNOWN",
                "confidence": 0.8 if not material_missing else 0.0,
            }
        )
    return pd.DataFrame(rows)


def synthetic_scene(seed: int = 0) -> dict[str, pd.DataFrame]:
    patches = create_cube_patches(seed)
    views = create_existing_views()
    return {
        "patches": patches,
        "views": views,
        "observations": build_observation_table(patches, views, seed),
        "semantics": build_semantic_tables(patches),
        "materials": build_material_table(patches),
    }
