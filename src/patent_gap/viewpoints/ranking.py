from __future__ import annotations

import ast
from itertools import product
from typing import Any

import numpy as np
import pandas as pd


def _unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else vec


def _as_vec3(value: Any, default: tuple[float, float, float]) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = np.fromstring(value.strip("[]()"), sep=" ")
    try:
        vec = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        vec = np.asarray(default, dtype=float)
    if vec.size != 3 or not np.isfinite(vec).all():
        return np.asarray(default, dtype=float)
    return vec


def _frontality(normal: np.ndarray, centroid: np.ndarray, position: np.ndarray) -> float:
    return float(max(0.0, np.dot(_unit(normal), _unit(position - centroid))))


def _rotate_around_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula — rotate v around unit axis by angle_rad."""
    axis = _unit(axis)
    return (v * np.cos(angle_rad)
            + np.cross(axis, v) * np.sin(angle_rad)
            + axis * np.dot(axis, v) * (1 - np.cos(angle_rad)))


def _candidate_direction(base: np.ndarray, tangent: np.ndarray, bitangent: np.ndarray,
                         az_deg: float, el_deg: float) -> np.ndarray:
    """
    Generate a camera direction using proper spherical coordinate perturbation.

    Azimuth tilts the surface normal around the local bitangent; elevation then
    tilts around the local tangent. Both offsets therefore remain effective when
    the other offset is zero.
    """
    az_rad = np.deg2rad(float(az_deg))
    el_rad = np.deg2rad(float(el_deg))
    dir_azimuth = _rotate_around_axis(base, bitangent, az_rad)
    dir_final = _rotate_around_axis(dir_azimuth, tangent, el_rad)
    return _unit(dir_final)


def generate_candidates(patch_scores: pd.DataFrame, view_config: dict[str, Any] | None = None) -> pd.DataFrame:
    view_config = view_config or {}
    distances = view_config.get("distances", [1.5, 2.5, 4.0])
    az_offsets = view_config.get("azimuth_offsets_deg", [-45, 0, 45])
    el_offsets = view_config.get("elevation_offsets_deg", [-20, 0, 20])
    threshold = float(view_config.get("gap_threshold", 0.55))
    max_target_patches = int(view_config.get("max_target_patches", 30))
    if max_target_patches <= 0:
        return pd.DataFrame()
    distances = [float(distance) for distance in distances if float(distance) > 0]
    if not distances:
        raise ValueError("view distances must contain at least one positive value")
    high = patch_scores[patch_scores["G_gap"] >= threshold]
    if high.empty:
        high = patch_scores.head(3)
    # Limit to the highest-gap patches to keep candidate count manageable
    if len(high) > max_target_patches:
        high = high.nlargest(max_target_patches, "G_gap")
    rows = []
    seen: set[tuple[int, float, float, float]] = set()
    cid = 0
    for patch in high.itertuples(index=False):
        centroid = _as_vec3(
            getattr(patch, "centroid", None),
            (
                getattr(patch, "centroid_x", 0.0),
                getattr(patch, "centroid_y", 0.0),
                getattr(patch, "centroid_z", 0.0),
            ),
        )
        normal = _as_vec3(
            getattr(patch, "normal", None),
            (
                getattr(patch, "normal_x", 1.0),
                getattr(patch, "normal_y", 0.0),
                getattr(patch, "normal_z", 0.0),
            ),
        )
        base = _unit(normal)
        if np.linalg.norm(base) < 1e-8:
            base = np.array([1.0, 0.0, 0.0])
        # Build an orthonormal frame around the patch normal
        up = np.array([0.0, 0.0, 1.0])
        tangent = _unit(np.cross(base, up))
        if np.linalg.norm(tangent) < 1e-6:
            tangent = _unit(np.cross(base, np.array([1.0, 0.0, 0.0])))
        bitangent = _unit(np.cross(base, tangent))

        for distance, az, el in product(distances, az_offsets, el_offsets):
            direction = _candidate_direction(base, tangent, bitangent, az, el)
            position = centroid + float(distance) * direction
            orientation = _unit(centroid - position)
            dedupe_key = (
                int(patch.patch_id),
                round(float(position[0]), 7),
                round(float(position[1]), 7),
                round(float(position[2]), 7),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append({
                "view_id": f"candidate_{cid:04d}",
                "target_patch_id": int(patch.patch_id),
                "position": tuple(position.tolist()),
                "orientation": tuple(orientation.tolist()),
                "distance": float(distance),
                "azimuth_offset_deg": float(az),
                "elevation_offset_deg": float(el),
            })
            cid += 1
    return pd.DataFrame(rows)


def score_candidates(patch_scores: pd.DataFrame, candidates: pd.DataFrame, view_config: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Score all candidate viewpoints in a vectorised batch.

    For each candidate position, compute Q(patch, view) = frontality × distance_quality
    × resolution_quality against every patch, then aggregate view value V(v).
    Vectorised over patches; iterates over candidates (usually ≤1000).
    """
    eta = float((view_config or {}).get("eta", 0.10))
    gap_threshold = float((view_config or {}).get("gap_threshold", 0.55))
    frontality_min = float((view_config or {}).get("frontality_min", 0.35))
    fov_deg = float((view_config or {}).get("field_of_view_deg", 90.0))
    max_range = float((view_config or {}).get("max_range_m", 20.0))
    # 距离核须绑在传感器最优距离上, 不能写死室内量级的 2.5 m —— 缺省 2.5 配
    # (4, 8, 14) m 的户外候选距离档时, d=8 m 的权重只有 d=4 m 的 1/45、d=14 m
    # 的 1/2200 万, 等于生成了远档候选却给它们打零分, 有效半径只剩约 6 m。
    #
    # 但 σ **不能**跟着一起缩: 未传 r_opt_m 的调用方(webapp、cli、
    # run_real_pipeline、run_synthetic_pipeline、v1 闭环、greedy_view_set)
    # 会落到 2.5, 若同时把 σ 从原来的 2.0 改成 0.4·2.5=1.0, d=6 m 的权重要掉
    # 99%、d=8 m 掉 99.999% —— 有效半径从约 ±4 m 缩到 ±2 m, 比它要修的问题
    # 更糟, 且真实交付走的正是这几条通路。故 σ 缺省保持 2.0, 只有显式给出
    # r_opt_m 时才按 0.4·r_opt 缩放。
    r_opt = float((view_config or {}).get("r_opt_m", 2.5))
    sigma_d = 0.4 * r_opt if "r_opt_m" in (view_config or {}) else 2.0
    if not 0 < fov_deg <= 180:
        raise ValueError("field_of_view_deg must be in (0, 180]")
    if max_range <= 0:
        raise ValueError("max_range_m must be positive")
    fov_cos = float(np.cos(np.deg2rad(fov_deg / 2.0)))

    if candidates.empty or patch_scores.empty:
        return pd.DataFrame()

    patch_centroids = np.vstack(
        [
            _as_vec3(
                getattr(row, "centroid", None),
                (
                    getattr(row, "centroid_x", 0.0),
                    getattr(row, "centroid_y", 0.0),
                    getattr(row, "centroid_z", 0.0),
                ),
            )
            for row in patch_scores.itertuples(index=False)
        ]
    )
    patch_normals = np.vstack(
        [
            _as_vec3(
                getattr(row, "normal", None),
                (
                    getattr(row, "normal_x", 1.0),
                    getattr(row, "normal_y", 0.0),
                    getattr(row, "normal_z", 0.0),
                ),
            )
            for row in patch_scores.itertuples(index=False)
        ]
    )

    # Normalise patch normals
    n_norms = np.linalg.norm(patch_normals, axis=1, keepdims=True)
    n_norms = np.where(n_norms > 1e-8, n_norms, 1.0)
    patch_normals = patch_normals / n_norms

    patch_gap = pd.to_numeric(patch_scores["G_gap"], errors="coerce").fillna(0).clip(0, 1).to_numpy()
    patch_area = pd.to_numeric(patch_scores["area"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    patch_ids = patch_scores["patch_id"].values
    patch_is_high_gap = patch_gap >= gap_threshold

    rows = []
    for cand in candidates.itertuples(index=False):
        pos = _as_vec3(cand.position, (0.0, 0.0, 0.0))
        orientation = _unit(_as_vec3(cand.orientation, (0.0, 0.0, -1.0)))

        # Direction from each patch centroid to camera position (N×3)
        dirs = pos - patch_centroids          # N×3
        dists = np.linalg.norm(dirs, axis=1)  # N
        dists_safe = np.where(dists > 1e-6, dists, 1.0)
        dirs_norm = dirs / dists_safe[:, None]

        # Frontality: dot(patch_normal, direction_to_camera)
        frontality = np.sum(patch_normals * dirs_norm, axis=1).clip(0, 1)
        camera_to_patch = -dirs_norm
        camera_alignment = np.sum(camera_to_patch * orientation[None, :], axis=1)
        visible_mask = (
            (frontality > frontality_min)
            & (camera_alignment >= fov_cos)
            & (dists <= max_range)
            & (dists > 1e-6)
        )

        if visible_mask.sum() == 0:
            rows.append({
                "view_id": cand.view_id,
                "target_patch_id": cand.target_patch_id,
                "position": cand.position,
                "orientation": cand.orientation,
                "value": 0.0,
                "visible_gap_area": 0.0,
                "mean_quality": 0.0,
                "overlap": 0.0,
                "estimated_gain": 0.0,
                "visible_patch_ids": "",
            })
            continue

        vis_front = frontality[visible_mask]
        vis_dists = dists[visible_mask]
        vis_gap = patch_gap[visible_mask]
        vis_area = patch_area[visible_mask]
        vis_ids = patch_ids[visible_mask]
        vis_is_high = patch_is_high_gap[visible_mask]

        # Quality components
        # 距离核须绑在传感器最优距离上。原写死 2.5 m / σ=2.0 是室内量级, 而
        # OPEN_ISSUES #15 已把候选距离档改成户外的 (4, 8, 14) m 却没同步改这里:
        # 那组常数下 d=8 m 的权重只有 d=4 m 的 1/45、d=14 m 的 1/2200 万, 也就是
        # B1 生成了 8 m 与 14 m 档的候选却给它们打零分, 有效工作半径只剩约 6 m。
        # 这会把 B1 变成稻草人 —— 它在户外失效不该归因于"母专利方法弱"。
        dist_quality = np.exp(-((vis_dists - r_opt) ** 2) / (2 * sigma_d ** 2))
        res_quality = np.clip(1.0 / (1.0 + 0.15 * vis_dists), 0.0, 1.0)
        q = vis_front * dist_quality * res_quality

        value = float((vis_gap * q * vis_area).sum())
        visible_gap_area = float(vis_area[vis_is_high].sum())
        mean_quality = float(q.mean())

        # Redundancy penalty: fraction of visible patches not the primary target
        n_vis = len(vis_ids)
        target_pid = int(cand.target_patch_id)
        non_target = int((vis_ids != target_pid).sum())
        overlap = 0.1 * non_target / n_vis

        rows.append({
            "view_id": cand.view_id,
            "target_patch_id": cand.target_patch_id,
            "position": cand.position,
            "orientation": cand.orientation,
            "value": value - eta * overlap,
            "visible_gap_area": visible_gap_area,
            "mean_quality": mean_quality,
            "overlap": overlap,
            "estimated_gain": max(0.0, value - eta * overlap),
            "visible_patch_ids": ";".join(str(p) for p in vis_ids),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("value", ascending=False).reset_index(drop=True)


def greedy_sequential_ranking(patch_scores: pd.DataFrame, candidates: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    remaining = patch_scores.copy()
    selected = []
    candidate_pool = candidates.copy()
    for step in range(k):
        ranked = score_candidates(remaining, candidate_pool)
        if ranked.empty:
            break
        best = ranked.iloc[0].to_dict()
        best["greedy_step"] = step
        selected.append(best)
        visible = {int(x) for x in str(best["visible_patch_ids"]).split(";") if x != ""}
        if visible:
            remaining.loc[remaining["patch_id"].isin(visible), "G_gap"] *= 0.25
        candidate_pool = candidate_pool[candidate_pool["view_id"] != best["view_id"]]
    return pd.DataFrame(selected)
