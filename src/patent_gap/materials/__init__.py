"""
Material attribute layer.

Reads IFC material definitions and RGB statistics from matched point-cloud
observations to compute D_mat_missing and D_mat_conflict per patch.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


# Simple color→category mapping for visual material classification
def _rgb_to_category(r: float, g: float, b: float) -> str:
    """Classify a mean RGB color (0-255 range) into a rough material category."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    brightness = (r + g + b) / 3.0
    if brightness > 0.85:
        return "paint"
    if brightness < 0.20:
        return "generic"
    if abs(r - g) < 0.05 and abs(g - b) < 0.05:
        return "concrete" if 0.3 < brightness < 0.7 else "generic"
    if r > g + 0.15 and r > b + 0.15:
        return "wood"
    if b > r + 0.10 and b > g + 0.05:
        return "glass"
    if g > r + 0.05 and g > b + 0.05:
        return "vegetation"
    return "generic"


# IFC material name → rough physical category
def _mat_name_to_category(name: str) -> str:
    if not name:
        return "unknown"
    n = name.lower()
    for kw, cat in [
        ("glass", "glass"), ("steel", "metal"), ("metal", "metal"),
        ("alumin", "metal"), ("concrete", "concrete"), ("paint", "paint"),
        ("plaster", "plaster"), ("wood", "wood"), ("timber", "wood"),
        ("gypsum", "plaster"), ("tile", "tile"), ("carpet", "textile"),
        ("fabric", "textile"), ("rubber", "rubber"),
    ]:
        if kw in n:
            return cat
    return "generic"


def build_material_table(
    patches_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    materials_df: pd.DataFrame,
    assoc_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build the material evidence table for every patch.

    Outputs columns:
        patch_id, material_present, material_name, material_category,
        visual_material_category, material_conflict,
        color_mean_r, color_mean_g, color_mean_b,
        source, confidence, D_mat_missing, D_mat_conflict
    """
    cfg = config or {}
    conflict_threshold = float(cfg.get("material_conflict_threshold", 0.60))
    min_rgb_points = int(cfg.get("min_rgb_points", 3))
    if not 0 <= conflict_threshold <= 1:
        raise ValueError("material_conflict_threshold must be in [0, 1]")

    # Build element_index → material info from parquets
    mat_id_to_name = {int(r.material_id): str(r.material_name) for r in materials_df.itertuples()}
    mat_id_to_cat = {int(r.material_id): _mat_name_to_category(str(r.material_name))
                     for r in materials_df.itertuples()}

    elem_to_mat: dict[int, dict] = {}
    for row in elements_df.itertuples():
        try:
            mat_ids = json.loads(row.material_ids or "[]")
        except Exception:
            mat_ids = []
        if mat_ids:
            mid = int(mat_ids[0])
            elem_to_mat[int(row.element_index)] = {
                "material_name": mat_id_to_name.get(mid, None),
                "material_category": mat_id_to_cat.get(mid, "unknown"),
                "present": True,
            }
        else:
            elem_to_mat[int(row.element_index)] = {
                "material_name": None,
                "material_category": "unknown",
                "present": False,
            }

    # Compute RGB statistics from matched sample points per element_guid
    elem_rgb: dict[str, dict] = {}
    if assoc_df is not None and "r" in assoc_df.columns:
        matched = assoc_df[assoc_df["matched"]].copy() if "matched" in assoc_df.columns else assoc_df
        if len(matched) > 0 and "element_guid" in matched.columns:
            for guid, grp in matched.groupby("element_guid"):
                if len(grp) < min_rgb_points:
                    continue
                elem_rgb[str(guid)] = {
                    "mean_r": float(grp["r"].mean()),
                    "mean_g": float(grp["g"].mean()),
                    "mean_b": float(grp["b"].mean()),
                    "std_r": float(grp["r"].std()),
                    "sample_count": int(len(grp)),
                }

    rows: list[dict] = []
    for patch in patches_df.itertuples():
        elem_idx = int(patch.element_index)
        guid = str(patch.element_guid)
        mat_info = elem_to_mat.get(elem_idx, {"material_name": None, "material_category": "unknown", "present": False})

        ifc_present = mat_info["present"]
        ifc_name = mat_info["material_name"]
        ifc_cat = mat_info["material_category"]

        # Visual category from RGB
        rgb = elem_rgb.get(guid)
        visual_cat: str
        color_r, color_g, color_b = 128.0, 128.0, 128.0
        if rgb:
            color_r, color_g, color_b = rgb["mean_r"], rgb["mean_g"], rgb["mean_b"]
            visual_cat = _rgb_to_category(color_r, color_g, color_b)
            visual_confidence = float(
                np.clip(1.0 - float(rgb.get("std_r", 128.0)) / 128.0, 0.0, 1.0)
            )
        else:
            visual_cat = ifc_cat if ifc_present else "unknown"
            visual_confidence = 0.0

        # Material conflict: IFC category vs visual category when both are known
        conflict = False
        if (
            ifc_present
            and rgb
            and visual_confidence >= conflict_threshold
            and ifc_cat not in ("unknown", "generic")
            and visual_cat not in ("unknown", "generic", "vegetation")
        ):
            # Conservative: only flag clear mismatches
            clear_mismatches = {
                ("glass", "concrete"), ("glass", "wood"),
                ("metal", "concrete"), ("metal", "wood"),
                ("wood", "glass"), ("concrete", "glass"),
            }
            pair = (ifc_cat, visual_cat)
            conflict = pair in clear_mismatches

        d_mat_missing = 0.0 if ifc_present else 1.0
        d_mat_conflict = 1.0 if conflict else 0.0

        rows.append({
            "patch_id": int(patch.patch_id),
            "material_present": ifc_present,
            "material_name": ifc_name,
            "material_category": ifc_cat,
            "visual_material_category": visual_cat,
            "material_conflict": conflict,
            "color_mean_r": color_r,
            "color_mean_g": color_g,
            "color_mean_b": color_b,
            "source": "IFC" if ifc_present else "UNKNOWN",
            "confidence": visual_confidence if rgb else (0.9 if ifc_present else 0.0),
            "D_mat_missing": d_mat_missing,
            "D_mat_conflict": d_mat_conflict,
        })

    return pd.DataFrame(rows)
