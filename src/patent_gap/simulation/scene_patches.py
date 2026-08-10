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
        # 缺口检测的目标集是 BIM 构件清单。竣工态临时占位物在 BIM 中查不到,
        # 不构成待扫资产, 只作为遮挡体参与光线求交(tri_to_patch 保持 -1)。
        if not comp.in_bim:
            continue
        # 非目标环境构件(厂房、围墙、杂物)同理: 它们遮挡视线、影响通行, 但不是
        # 本次验收要检查的资产。此前它们既被算作待扫目标, 又被 prior_tri_mask
        # 当作"非目标环境"从 M_plan⁰ 移除 —— 于是弱先验下规划器被要求扫一栋楼,
        # 却拿不到那栋楼的几何, 会以为自己能透视过去。凡 M_ref 声明为目标的,
        # M_plan⁰ 必须含其几何; 二者的口径必须一致。
        #
        # 注意: 这改变了目标集本身(合成变电站中占分块数的 5.7–8.0%), 因此 E7/E8
        # 与 E2/E5/E6 的**绝对**指标不可跨实验直接比较。各实验内部的对照不受影响,
        # 且 run.json 的 git_commit 守卫会拦住把两种口径拼在一起的续跑。
        if not comp.is_target:
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
