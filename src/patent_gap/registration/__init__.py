"""
IFC ↔ point-cloud registration.

Strategy:
1. Apply a known coarse translation (found during CRAS association).
2. Optionally refine with point-to-plane ICP using Open3D.
3. Return the final 4×4 transformation matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# Known coarse translation from the CRAS association run
# (point_registered = point_raw + COARSE_TRANSLATION)
CRAS_COARSE_TRANSLATION = np.array([0.6848, 0.0, -0.6665])


def _points_array(points: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(points, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if len(out) == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return out


def _transform_array(transform: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(transform, dtype=np.float64)
    if out.shape != (4, 4) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(out[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} must be a homogeneous SE(3) matrix")
    return out


def apply_translation(points: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a 3-vector translation to Nx3 point array."""
    points = _points_array(points, "points")
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation must be a finite 3-vector")
    return points + translation.reshape(1, 3)


def icp_refinement(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_transform: np.ndarray | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_dist: float = 0.10,
) -> np.ndarray:
    """
    Point-to-point ICP implemented with scipy cKDTree (no Open3D required).

    Returns 4×4 SE(3) transformation matrix that aligns source → target.
    """
    from scipy.spatial import cKDTree

    if max_iterations < 0:
        raise ValueError("max_iterations cannot be negative")
    if max_correspondence_dist <= 0:
        raise ValueError("max_correspondence_dist must be positive")
    src = _points_array(source_points, "source_points")
    target_points = _points_array(target_points, "target_points")
    T = (
        np.eye(4, dtype=np.float64)
        if initial_transform is None
        else _transform_array(initial_transform, "initial_transform").copy()
    )

    # Subsample for speed if too many points
    n_max = 20_000
    if len(src) > n_max:
        idx = np.random.default_rng(0).choice(len(src), n_max, replace=False)
        src_sub = src[idx]
    else:
        src_sub = src

    if len(target_points) > 50_000:
        idx = np.random.default_rng(1).choice(len(target_points), 50_000, replace=False)
        tgt_sub = target_points[idx]
    else:
        tgt_sub = target_points

    tree = cKDTree(tgt_sub)
    prev_rmse = float("inf")

    for _ in range(max_iterations):
        # Apply current transform to source
        src_h = np.hstack([src_sub, np.ones((len(src_sub), 1))])
        src_t = (T @ src_h.T).T[:, :3]

        # Find correspondences
        dists, idx_nn = tree.query(src_t, distance_upper_bound=max_correspondence_dist)
        valid = dists < max_correspondence_dist
        if valid.sum() < 10:
            break

        src_v = src_t[valid]
        tgt_v = tgt_sub[idx_nn[valid]]

        rmse = float(np.sqrt((dists[valid] ** 2).mean()))
        if abs(prev_rmse - rmse) < tolerance:
            break
        prev_rmse = rmse

        # Compute optimal rigid transform (SVD)
        mu_s = src_v.mean(axis=0)
        mu_t = tgt_v.mean(axis=0)
        H = (src_v - mu_s).T @ (tgt_v - mu_t)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = mu_t - R @ mu_s

        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t
        T = dT @ T

    return T


def register_pointcloud(
    points_xyz: np.ndarray,
    ifc_vertices: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Register a raw CRAS point cloud to IFC coordinate frame.

    Returns
    -------
    registered_points : ndarray  (N, 3)
    transform_4x4     : ndarray  (4, 4)
    """
    points_xyz = _points_array(points_xyz, "points_xyz")
    ifc_vertices = _points_array(ifc_vertices, "ifc_vertices")
    cfg = config or {}
    refine = bool(cfg.get("refine_icp", True))
    max_iter = int(cfg.get("icp_max_iterations", 50))
    max_dist = float(cfg.get("icp_max_correspondence_dist", 0.10))

    # Step 1: coarse translation
    T_coarse = np.eye(4)
    T_coarse[:3, 3] = CRAS_COARSE_TRANSLATION

    if not refine:
        pts_registered = apply_translation(points_xyz, CRAS_COARSE_TRANSLATION)
        return pts_registered, T_coarse

    # Step 2: ICP refinement
    pts_coarse = apply_translation(points_xyz, CRAS_COARSE_TRANSLATION)
    T_icp = icp_refinement(
        pts_coarse,
        ifc_vertices,
        initial_transform=np.eye(4),
        max_iterations=max_iter,
        max_correspondence_dist=max_dist,
    )
    T_full = T_icp @ T_coarse

    pts_h = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])
    pts_registered = (T_full @ pts_h.T).T[:, :3]
    return pts_registered, T_full


def load_or_compute_transform(
    cache_path: str | Path,
    points_xyz: np.ndarray | None = None,
    ifc_vertices: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return cached 4×4 transform or compute and cache it."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return _transform_array(np.load(str(cache_path)), "cached transform")
    if points_xyz is None or ifc_vertices is None:
        # Fall back to coarse translation only
        T = np.eye(4)
        T[:3, 3] = CRAS_COARSE_TRANSLATION
        return T
    _, T = register_pointcloud(points_xyz, ifc_vertices, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), T)
    return T
