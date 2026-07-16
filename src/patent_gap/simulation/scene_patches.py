"""将 SceneModel 分割为 Patch(复用母专利公式(1)–(8)的区域生长实现)。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..patches import (
    _build_adjacency, _region_grow, _triangle_areas, _triangle_centroids,
    _triangle_normals,
)
from .scene_gen import SceneModel


def build_scene_patches(scene: SceneModel, normal_threshold_deg: float = 20.0,
                        min_patch_area: float = 0.01) -> tuple[pd.DataFrame, np.ndarray]:
    """返回 (patches df, tri_to_patch (T,) int64, 地面/未达阈三角面=-1)。"""
    cos_thr = float(np.cos(np.deg2rad(normal_threshold_deg)))
    V, F = scene.vertices, scene.triangles
    normals = _triangle_normals(V, F)
    areas = _triangle_areas(V, F)
    centroids = _triangle_centroids(V, F)

    rows = []
    tri_to_patch = np.full(len(F), -1, dtype=np.int64)
    pid = 0
    for comp in scene.components:
        if comp.cls == "ground":
            continue
        idx = np.arange(comp.tri_start, comp.tri_end)
        if len(idx) == 0:
            continue
        local_faces = F[idx]
        adj = _build_adjacency(local_faces, len(idx))
        labels = _region_grow(normals[idx], adj, cos_thr)
        for lbl in np.unique(labels):
            m = labels == lbl
            a = float(areas[idx[m]].sum())
            if a < min_patch_area:
                continue
            w = areas[idx[m]] / (a + 1e-12)
            cen = (centroids[idx[m]] * w[:, None]).sum(axis=0)
            nrm = (normals[idx[m]] * w[:, None]).sum(axis=0)
            nl = float(np.linalg.norm(nrm))
            nrm = nrm / nl if nl > 1e-9 else np.array([0.0, 0.0, 1.0])
            rows.append({
                "patch_id": pid,
                "element_guid": f"comp_{comp.comp_id:04d}",
                "ifc_class": comp.cls,
                "name": f"{comp.cls}_{comp.comp_id}",
                "centroid_x": cen[0], "centroid_y": cen[1], "centroid_z": cen[2],
                "normal_x": nrm[0], "normal_y": nrm[1], "normal_z": nrm[2],
                "area": a,
                "engineering_importance": comp.importance,
                "gt_missing": False,
            })
            tri_to_patch[idx[m]] = pid
            pid += 1
    df = pd.DataFrame(rows)
    df["centroid"] = list(zip(df["centroid_x"], df["centroid_y"], df["centroid_z"]))
    df["normal"] = list(zip(df["normal_x"], df["normal_y"], df["normal_z"]))
    return df, tri_to_patch
