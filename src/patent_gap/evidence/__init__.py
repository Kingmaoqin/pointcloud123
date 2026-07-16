"""
Observation evidence layer — vectorised implementation.

Computes D_obs, D_ang, D_geo per patch in bulk rather than one patch at a time.
Key strategy:
  - Aggregate per *element* (not per patch) from chunk summaries and sample data
  - Distribute element-level statistics to patches proportionally by area
  - Angular coverage: for each sample point, compute incidence angle to its
    matched element's average normal, then aggregate per element
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compute_evidence_from_synthetic(
    patches_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    elem_stats_df: pd.DataFrame,
    patch_stats_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compute D_obs / D_ang / D_geo per patch from synthetic-scan element stats.

    elem_stats_df columns required:
        element_index, point_count, mean_frontality, best_frontality,
        valid_views, mean_distance_m
    """
    cfg = config or {}
    ref_density = float(cfg.get("reference_density_pts_m2", 500.0))
    gamma = float(cfg.get("frontality_gamma", 2.0))
    min_points_for_observed = int(cfg.get("min_points_for_observed", 1))
    if min_points_for_observed < 1:
        raise ValueError("min_points_for_observed must be at least 1")

    guid_to_idx = dict(zip(elements_df["GlobalId"], elements_df["element_index"]))

    stats: dict[int, dict] = {}
    for row in elem_stats_df.itertuples(index=False):
        stats[int(row.element_index)] = {
            "point_count":     int(row.point_count),
            "mean_frontality": float(row.mean_frontality),
            "best_frontality": float(row.best_frontality),
            "scanner_count": int(
                getattr(
                    row,
                    "scanner_count",
                    1 if int(row.point_count) > 0 else 0,
                )
            ),
            "valid_scanner_count": int(
                getattr(
                    row,
                    "valid_scanner_count",
                    getattr(row, "scanner_count", 1 if int(row.point_count) > 0 else 0),
                )
            ),
        }

    patch_stats: dict[int, dict] = {}
    if patch_stats_df is not None and not patch_stats_df.empty:
        required = {
            "patch_id",
            "point_count",
            "mean_frontality",
            "best_frontality",
            "scanner_count",
            "valid_scanner_count",
        }
        missing = required.difference(patch_stats_df.columns)
        if missing:
            raise ValueError(
                f"patch_stats_df is missing columns: {', '.join(sorted(missing))}"
            )
        for row in patch_stats_df.itertuples(index=False):
            patch_stats[int(row.patch_id)] = {
                "point_count": int(row.point_count),
                "mean_frontality": float(row.mean_frontality),
                "best_frontality": float(row.best_frontality),
                "scanner_count": int(row.scanner_count),
                "valid_scanner_count": int(row.valid_scanner_count),
            }
    has_patch_level_stats = patch_stats_df is not None

    p = patches_df.copy()
    p["elem_idx"] = p["element_guid"].map(guid_to_idx).fillna(-1).astype(int)
    elem_total_area = p.groupby("elem_idx")["area"].sum()

    rows: list[dict] = []
    for patch in p.itertuples(index=False):
        ei = int(patch.elem_idx)
        area = float(patch.area)
        t_area = float(elem_total_area.get(ei, max(area, 1e-6)))
        ratio = area / t_area if t_area > 0 else 1.0

        exact_patch_stats = patch_stats.get(int(patch.patch_id))
        st = (
            exact_patch_stats or {}
            if has_patch_level_stats
            else stats.get(ei, {})
        )
        full_cnt   = st.get("point_count", 0)
        best_front = st.get("best_frontality", 0.0)
        mean_front = st.get("mean_frontality", 0.0)
        scanner_count = st.get("scanner_count", 0)
        valid_scanners = st.get("valid_scanner_count", scanner_count)

        est_cnt = int(
            full_cnt
            if has_patch_level_stats
            else full_cnt * ratio
        )
        observed = est_cnt >= min_points_for_observed

        valid_patch = valid_scanners if observed else 0
        proj_res     = min(best_front ** gamma, 0.95) if observed else 0.0
        reg_conf     = 0.95 if observed else 0.10
        ang_div      = min(valid_patch / 3.0, 1.0)

        density = est_cnt / max(area, 1e-6)
        d_ratio = min(density / ref_density, 1.0)
        cov     = d_ratio

        rows.append({
            "patch_id":                    int(patch.patch_id),
            "directly_observed":           observed,
            "source_type":                 "MEASURED" if observed else "MODEL",
            "number_of_views":             scanner_count if observed else 0,
            "number_of_valid_views":       valid_patch,
            "best_frontality":             best_front,
            "mean_frontality":             mean_front,
            "angular_diversity":           ang_div,
            "point_count":                 est_cnt,
            "point_density":               density,
            "coverage_ratio":              cov,
            "density_ratio":               d_ratio,
            "projected_resolution_quality": proj_res,
            "registration_confidence":     reg_conf,
            # D_obs / D_ang / D_geo intentionally omitted so compute_patch_scores
            # derives them from coverage_ratio / density_ratio each iteration,
            # enabling proper closed-loop updates.
        })

    return pd.DataFrame(rows)


def compute_controlled_withheld_evidence(
    patches_df: pd.DataFrame,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Create a controlled observation table for a pre-declared missing region."""
    cfg = config or {}
    rng = np.random.default_rng(seed)
    ref_density = float(cfg.get("reference_density_pts_m2", 500.0))
    degradation_fraction = float(cfg.get("degradation_fraction", 0.15))
    missing_leakage_fraction = float(cfg.get("missing_leakage_fraction", 0.20))
    if not 0 <= degradation_fraction <= 1 or not 0 <= missing_leakage_fraction <= 1:
        raise ValueError("degradation and leakage fractions must be in [0, 1]")
    if "gt_missing" not in patches_df.columns:
        raise ValueError("patches_df must contain pre-declared gt_missing labels")

    rows: list[dict[str, Any]] = []
    for patch in patches_df.itertuples(index=False):
        missing = bool(patch.gt_missing)
        area = max(float(patch.area), 1e-6)
        if missing:
            leaked = rng.random() < missing_leakage_fraction
            views = int(leaked)
            best_frontality = float(rng.uniform(0.20, 0.55)) if leaked else 0.0
            mean_frontality = best_frontality * 0.8
            angular_diversity = float(rng.uniform(0.05, 0.25)) if leaked else 0.0
            coverage = float(rng.uniform(0.05, 0.35)) if leaked else 0.0
            density_ratio = float(rng.uniform(0.05, 0.35)) if leaked else 0.0
            projected_quality = float(rng.uniform(0.20, 0.50)) if leaked else 0.0
            registration_confidence = float(rng.uniform(0.70, 0.88)) if leaked else 0.15
        else:
            views = int(rng.integers(2, 6))
            best_frontality = float(rng.uniform(0.68, 0.98))
            mean_frontality = float(
                np.clip(best_frontality - rng.uniform(0.05, 0.20), 0, 1)
            )
            angular_diversity = float(np.clip(views / 5.0 + rng.normal(0, 0.05), 0, 1))
            coverage = float(rng.uniform(0.78, 0.99))
            density_ratio = float(rng.uniform(0.72, 1.0))
            projected_quality = float(rng.uniform(0.75, 0.98))
            registration_confidence = float(rng.uniform(0.88, 0.99))
            if rng.random() < degradation_fraction:
                views = 1
                best_frontality *= float(rng.uniform(0.35, 0.70))
                mean_frontality *= float(rng.uniform(0.35, 0.70))
                angular_diversity *= float(rng.uniform(0.20, 0.55))
                coverage *= float(rng.uniform(0.05, 0.45))
                density_ratio *= float(rng.uniform(0.05, 0.50))
                projected_quality *= float(rng.uniform(0.30, 0.65))

        point_density = ref_density * density_ratio
        rows.append(
            {
                "patch_id": int(patch.patch_id),
                "directly_observed": (not missing) or views > 0,
                "source_type": "MODEL" if missing else "MEASURED",
                "number_of_views": views,
                "number_of_valid_views": views,
                "best_frontality": best_frontality,
                "mean_frontality": mean_frontality,
                "angular_diversity": angular_diversity,
                "point_count": int(point_density * area),
                "point_density": point_density,
                "coverage_ratio": coverage,
                "density_ratio": density_ratio,
                "projected_resolution_quality": projected_quality,
                "registration_confidence": registration_confidence,
                "evidence_granularity": "controlled_patch",
            }
        )
    return pd.DataFrame(rows)


def aggregate_element_counts_from_chunks(chunk_dir: str | Path) -> dict[str, int]:
    """Sum element_counts across all chunk summary JSONs."""
    chunk_dir = Path(chunk_dir)
    parts_dir = chunk_dir / "parts"
    totals: dict[str, int] = {}
    for p in sorted(parts_dir.glob("*.summary.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for k, v in data.get("element_counts", {}).items():
            totals[str(k)] = totals.get(str(k), 0) + int(v)
    return totals


def compute_evidence(
    patches_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    assoc_df: pd.DataFrame,
    chunk_dir: str | Path | None = None,
    mesh_npz_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Vectorised evidence computation per patch.

    Strategy
    --------
    1. Aggregate point counts from chunk summaries at element level.
    2. From the sample association, compute per-element angular stats.
    3. Distribute element-level statistics to each patch proportional to area.
    4. Compute D_obs / D_ang / D_geo from the distributed stats.
    """
    cfg = config or {}
    ref_density = float(cfg.get("reference_density_pts_m2", 500.0))
    max_incidence_deg = float(cfg.get("max_incidence_angle_deg", 65.0))
    gamma = float(cfg.get("frontality_gamma", 2.0))
    cos_max = float(np.cos(np.deg2rad(max_incidence_deg)))

    # ── 1. Element-level point counts from full-dataset chunks ────────────────
    elem_full_counts: dict[int, int] = {}
    if chunk_dir is not None:
        raw = aggregate_element_counts_from_chunks(chunk_dir)
        for k, v in raw.items():
            try:
                elem_full_counts[int(k)] = v
            except ValueError:
                pass

    # ── 2. Element-level angular statistics from sample association ───────────
    # Build per-element normal lookup from patches
    # Use area-weighted average normal per element
    if "normal_x" not in patches_df.columns:
        # unpack tuple column
        patches_df = patches_df.copy()
        patches_df["normal_x"] = [n[0] for n in patches_df["normal"]]
        patches_df["normal_y"] = [n[1] for n in patches_df["normal"]]
        patches_df["normal_z"] = [n[2] for n in patches_df["normal"]]

    elem_normals: dict[int, np.ndarray] = {}
    for ei, grp in patches_df.groupby("element_index"):
        areas = grp["area"].values
        nx = (grp["normal_x"].values * areas).sum()
        ny = (grp["normal_y"].values * areas).sum()
        nz = (grp["normal_z"].values * areas).sum()
        n = np.array([nx, ny, nz])
        nm = np.linalg.norm(n)
        elem_normals[int(ei)] = n / nm if nm > 1e-8 else np.array([0.0, 0.0, 1.0])

    # Sample association points → per-element angular stats. A fused CRAS point
    # cloud has no scanner origins, so incidence angles are unavailable unless
    # explicit scanner_x/scanner_y/scanner_z columns are present.
    matched = assoc_df[assoc_df["matched"]].copy() if "matched" in assoc_df.columns else assoc_df.copy()
    has_scanner_origins = {
        "scanner_x",
        "scanner_y",
        "scanner_z",
    }.issubset(matched.columns)
    has_scanner_ids = "scanner_id" in matched.columns

    # Build element_index lookup
    elem_guid_to_idx = {str(r.GlobalId): int(r.element_index) for r in elements_df.itertuples()}

    # Per-element angular accumulator
    elem_frontality_sum: dict[int, float] = {}
    elem_valid_view_count: dict[int, int] = {}
    elem_best_frontality: dict[int, float] = {}
    elem_point_count_sample: dict[int, int] = {}

    elem_view_count: dict[int, int] = {}
    if len(matched) > 0 and "element_guid" in matched.columns:
        # Build element centroids for direction computation
        elem_centroids: dict[int, np.ndarray] = {}
        for ei, grp in patches_df.groupby("element_index"):
            areas = grp["area"].values
            if "centroid_x" in grp.columns:
                cx = (grp["centroid_x"].values * areas).sum() / areas.sum()
                cy = (grp["centroid_y"].values * areas).sum() / areas.sum()
                cz = (grp["centroid_z"].values * areas).sum() / areas.sum()
                elem_centroids[int(ei)] = np.array([cx, cy, cz])

        for guid, grp in matched.groupby("element_guid"):
            ei = elem_guid_to_idx.get(str(guid))
            if ei is None:
                continue
            n_pts = len(grp)
            elem_point_count_sample[ei] = elem_point_count_sample.get(ei, 0) + n_pts
            elem_view_count[ei] = (
                int(grp["scanner_id"].nunique()) if has_scanner_ids else 0
            )

            normal = elem_normals.get(ei, np.array([0.0, 0.0, 1.0]))
            centroid = elem_centroids.get(ei)
            if centroid is None or not has_scanner_origins:
                continue

            origins = grp[["scanner_x", "scanner_y", "scanner_z"]].to_numpy(dtype=float)
            directions = origins - centroid
            dist = np.linalg.norm(directions, axis=1, keepdims=True)
            dist_safe = np.where(dist > 1e-6, dist, 1.0)
            dirs_norm = directions / dist_safe
            cos_inc = np.clip(np.dot(dirs_norm, normal), 0, 1)
            front = np.power(cos_inc, gamma)

            valid = cos_inc >= cos_max
            elem_frontality_sum[ei] = elem_frontality_sum.get(ei, 0.0) + float(front.sum())
            if has_scanner_ids:
                elem_valid_view_count[ei] = int(grp.loc[valid, "scanner_id"].nunique())
            else:
                elem_valid_view_count[ei] = int(valid.any())
            bf_current = elem_best_frontality.get(ei, 0.0)
            elem_best_frontality[ei] = max(bf_current, float(front.max()))

    # ── 3. Per-element total area (for area-ratio distribution) ───────────────
    elem_total_area = patches_df.groupby("element_index")["area"].sum().to_dict()

    # ── 4. Build per-patch evidence rows (vectorised) ─────────────────────────
    # Precompute numpy arrays for speed
    patch_ids = patches_df["patch_id"].values
    elem_indices = patches_df["element_index"].values
    areas = patches_df["area"].values

    rows_dict: dict[str, list] = {
        "patch_id": [], "directly_observed": [], "source_type": [],
        "number_of_views": [], "number_of_valid_views": [],
        "best_frontality": [], "mean_frontality": [], "angular_diversity": [],
        "point_count": [], "point_density": [],
        "coverage_ratio": [], "density_ratio": [],
        "projected_resolution_quality": [], "registration_confidence": [],
        "D_obs": [], "D_ang": [], "D_geo": [],
    }

    for pid, ei, area in zip(patch_ids, elem_indices, areas):
        ei = int(ei)
        area = float(area)
        total_elem_area = float(elem_total_area.get(ei, max(area, 1e-6)))
        area_ratio = area / total_elem_area if total_elem_area > 0 else 1.0

        # Point count (estimated from full-dataset + area ratio)
        full_count = elem_full_counts.get(ei, 0)
        est_count = int(full_count * area_ratio)

        # Angular stats (from sample, distributed by area ratio)
        total_sample_count = elem_point_count_sample.get(ei, 0)
        view_count_patch = elem_view_count.get(ei, 0) if has_scanner_origins else 0
        valid_count_patch = elem_valid_view_count.get(ei, 0) if has_scanner_origins else 0
        best_front = elem_best_frontality.get(ei, float("nan"))
        front_sum = elem_frontality_sum.get(ei, 0.0)
        mean_front = (
            front_sum / max(total_sample_count, 1)
            if total_sample_count > 0 and has_scanner_origins
            else float("nan")
        )
        angular_div = (
            min(1.0, valid_count_patch / 3.0)
            if has_scanner_origins
            else float("nan")
        )

        # Density from full-dataset count
        est_density = est_count / max(area, 1e-6)
        density_ratio = float(np.clip(est_density / ref_density, 0.0, 1.5))

        # Coverage ratio: proxy from density ratio (no per-patch surface sampling needed)
        # A patch with ≥ reference density has ≈1.0 coverage; scale linearly below
        coverage_ratio = float(np.clip(density_ratio, 0.0, 1.0))

        directly_observed = est_count > 2 or valid_count_patch > 0
        n_valid_norm = float(np.clip(valid_count_patch / 3.0, 0, 1))
        proj_res = (
            min(best_front, 0.95)
            if directly_observed and has_scanner_origins
            else float("nan")
        )
        reg_conf = 0.90 if directly_observed else 0.10
        source_type = "MEASURED" if directly_observed else "MODEL"

        obs_numerator = 0.35 * float(not directly_observed) + 0.20 * (1.0 - reg_conf)
        obs_denominator = 0.55
        if has_scanner_origins:
            obs_numerator += 0.25 * (1.0 - n_valid_norm) + 0.20 * (1.0 - proj_res)
            obs_denominator += 0.45
        D_obs = float(np.clip(obs_numerator / obs_denominator, 0, 1))
        D_ang = (
            float(
                np.clip(
                    1.0
                    - (
                        0.50 * best_front
                        + 0.30 * min(1.0, valid_count_patch / 3.0)
                        + 0.20 * angular_div
                    ),
                    0,
                    1,
                )
            )
            if has_scanner_origins
            else float("nan")
        )
        D_geo = float(np.clip(
            0.60 * (1.0 - coverage_ratio) + 0.40 * (1.0 - min(1.0, density_ratio)),
            0, 1,
        ))

        rows_dict["patch_id"].append(int(pid))
        rows_dict["directly_observed"].append(directly_observed)
        rows_dict["source_type"].append(source_type)
        rows_dict["number_of_views"].append(view_count_patch)
        rows_dict["number_of_valid_views"].append(valid_count_patch)
        rows_dict["best_frontality"].append(best_front)
        rows_dict["mean_frontality"].append(mean_front)
        rows_dict["angular_diversity"].append(angular_div)
        rows_dict["point_count"].append(est_count)
        rows_dict["point_density"].append(est_density)
        rows_dict["coverage_ratio"].append(coverage_ratio)
        rows_dict["density_ratio"].append(min(1.0, density_ratio))
        rows_dict["projected_resolution_quality"].append(proj_res)
        rows_dict["registration_confidence"].append(reg_conf)
        rows_dict["D_obs"].append(D_obs)
        rows_dict["D_ang"].append(D_ang)
        rows_dict["D_geo"].append(D_geo)

    out = pd.DataFrame(rows_dict)
    out["angular_evidence_available"] = has_scanner_origins
    out["evidence_granularity"] = "element_area_proxy"
    return out
