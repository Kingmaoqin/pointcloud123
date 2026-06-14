from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ALPHAS = {
    "alpha_sem": 0.16,
    "alpha_mat_missing": 0.14,
    "alpha_mat_conflict": 0.10,
    "alpha_obs": 0.22,
    "alpha_ang": 0.16,
    "alpha_geo": 0.22,
}


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if np.isnan(q).any():
        return float("nan")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-12) / (b[mask] + 1e-12))))

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2)


def normalize_alphas(config: dict[str, Any] | None) -> dict[str, float]:
    values = dict(DEFAULT_ALPHAS)
    if config:
        values.update({k: float(v) for k, v in config.items() if k in values})
    total = sum(max(0.0, v) for v in values.values())
    if total <= 0:
        return DEFAULT_ALPHAS.copy()
    return {k: max(0.0, v) / total for k, v in values.items()}


def _available_weighted_sum(row: pd.Series, components: list[tuple[str, str]], weights: dict[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for col, weight_name in components:
        value = row[col]
        if pd.isna(value):
            continue
        weight = weights[weight_name]
        numerator += weight * float(value)
        denominator += weight
    if denominator <= 0:
        return float("nan")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def compute_patch_scores(scene: dict[str, pd.DataFrame], gap_config: dict[str, Any] | None = None) -> pd.DataFrame:
    patches = scene["patches"].copy()
    if "material_name" in patches.columns:
        patches = patches.drop(columns=["material_name"])
    observations = scene["observations"].copy()
    semantics = scene["semantics"].copy()
    materials = scene["materials"].copy()

    sem_rows = []
    for row in semantics.itertuples(index=False):
        sem_rows.append({"patch_id": row.patch_id, "D_sem": js_divergence(row.p_bim, row.p_obs)})
    sem_scores = pd.DataFrame(sem_rows)

    df = patches.merge(observations, on="patch_id").merge(sem_scores, on="patch_id").merge(materials, on="patch_id")
    df["D_mat_missing"] = (~df["material_present"]).astype(float)
    df["D_mat_conflict"] = df["material_conflict"].astype(float)
    df["D_obs"] = (
        0.35 * (~df["directly_observed"]).astype(float)
        + 0.25 * np.clip((2 - df["number_of_valid_views"]) / 2, 0, 1)
        + 0.20 * (1 - df["projected_resolution_quality"])
        + 0.20 * (1 - df["registration_confidence"])
    )
    df["D_ang"] = np.clip(
        1
        - (
            0.50 * df["best_frontality"]
            + 0.30 * np.clip(df["number_of_valid_views"] / 2, 0, 1)
            + 0.20 * df["angular_diversity"]
        ),
        0,
        1,
    )
    df["D_geo"] = np.clip(0.60 * (1 - df["coverage_ratio"]) + 0.40 * (1 - np.clip(df["density_ratio"], 0, 1)), 0, 1)
    weights = normalize_alphas(gap_config or {})
    components = [
        ("D_sem", "alpha_sem"),
        ("D_mat_missing", "alpha_mat_missing"),
        ("D_mat_conflict", "alpha_mat_conflict"),
        ("D_obs", "alpha_obs"),
        ("D_ang", "alpha_ang"),
        ("D_geo", "alpha_geo"),
    ]
    df["G_gap"] = df.apply(lambda row: _available_weighted_sum(row, components, weights), axis=1)
    lambda_importance = float((gap_config or {}).get("lambda_importance", 0.30))
    df["G_task"] = np.clip(df["G_gap"] * (1.0 + lambda_importance * df["engineering_importance"]), 0, 2)
    df["gt_missing"] = df["gt_missing"].astype(bool)
    ordered_cols = [
        "patch_id",
        "element_guid",
        "ifc_class",
        "name",
        "centroid",
        "normal",
        "area",
        "engineering_importance",
        "gt_missing",
        "D_sem",
        "D_mat_missing",
        "D_mat_conflict",
        "D_obs",
        "D_ang",
        "D_geo",
        "G_gap",
        "G_task",
        "number_of_valid_views",
        "point_density",
        "coverage_ratio",
        "density_ratio",
        "material_name",
        "visual_material_category",
        "source_type",
    ]
    return df[ordered_cols].sort_values("G_task", ascending=False).reset_index(drop=True)


def rank_components(patch_scores: pd.DataFrame, gap_config: dict[str, Any] | None = None) -> pd.DataFrame:
    threshold = float((gap_config or {}).get("high_gap_threshold", 0.55))
    rows = []
    for guid, group in patch_scores.groupby("element_guid"):
        area = group["area"].sum()
        high = group[group["G_gap"] >= threshold]
        rows.append(
            {
                "element_guid": guid,
                "ifc_class": group["ifc_class"].iloc[0],
                "mean_gap": group["G_gap"].mean(),
                "max_gap": group["G_gap"].max(),
                "p90_gap": group["G_gap"].quantile(0.90),
                "high_gap_area_ratio": high["area"].sum() / area if area else 0.0,
                "number_of_high_gap_patches": len(high),
                "critical_surface_gap": group["G_task"].max(),
                "engineering_importance": group["engineering_importance"].max(),
            }
        )
    comp = pd.DataFrame(rows)
    comp["G_component"] = (
        0.25 * comp["mean_gap"]
        + 0.25 * comp["max_gap"]
        + 0.20 * comp["p90_gap"]
        + 0.20 * comp["high_gap_area_ratio"]
        + 0.10 * comp["engineering_importance"]
    )
    return comp.sort_values("G_component", ascending=False).reset_index(drop=True)
