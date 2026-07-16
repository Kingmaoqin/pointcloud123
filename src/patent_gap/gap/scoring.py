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

# 公式(40)后 K_i 全集扩展权重表(升级路径; 仅当 D_occ/D_rng/D_regsup 列存在时启用,
# 母专利 B1 路径不受影响)。权重和 1.05 → 公式(19)自动归一。
EXTENDED_ALPHAS = {
    "alpha_sem": 0.11,
    "alpha_mat_missing": 0.05,
    "alpha_mat_conflict": 0.05,
    "alpha_obs": 0.15,
    "alpha_ang": 0.11,
    "alpha_geo": 0.15,
    "alpha_occ": 0.18,     # 公式(29)
    "alpha_rng": 0.15,     # 公式(34)
    "alpha_regsup": 0.10,  # 公式(40)
}


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.ndim != 1 or q.ndim != 1 or p.shape != q.shape:
        raise ValueError("p and q must be one-dimensional arrays with the same shape")
    if np.isnan(p).any() or np.isnan(q).any():
        return float("nan")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("probability distributions cannot contain negative values")
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-12) / (b[mask] + 1e-12))))

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2)


def normalize_alphas(config: dict[str, Any] | None, extended: bool = False) -> dict[str, float]:
    values = dict(EXTENDED_ALPHAS if extended else DEFAULT_ALPHAS)
    if config:
        values.update({k: float(v) for k, v in config.items() if k in values})
    total = sum(max(0.0, v) for v in values.values())
    if total <= 0:
        return (EXTENDED_ALPHAS if extended else DEFAULT_ALPHAS).copy()
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


def _validate_patch_table(df: pd.DataFrame, name: str, *, allow_empty: bool = False) -> None:
    if "patch_id" not in df.columns:
        raise ValueError(f"{name} must contain a patch_id column")
    if not allow_empty and df.empty:
        raise ValueError(f"{name} cannot be empty")
    duplicated = df["patch_id"].duplicated(keep=False)
    if duplicated.any():
        ids = df.loc[duplicated, "patch_id"].astype(str).unique()[:5]
        raise ValueError(f"{name} contains duplicate patch_id values: {', '.join(ids)}")


def _compute_semantic_scores(semantics: pd.DataFrame) -> pd.DataFrame:
    if "D_sem" in semantics.columns:
        return semantics[["patch_id", "D_sem"]].copy()
    required = {"p_bim", "p_obs"}
    missing = required.difference(semantics.columns)
    if missing:
        out = semantics[["patch_id"]].copy()
        out["D_sem"] = np.nan
        return out
    return pd.DataFrame(
        {
            "patch_id": semantics["patch_id"].to_numpy(),
            "D_sem": [
                js_divergence(p_bim, p_obs)
                for p_bim, p_obs in zip(semantics["p_bim"], semantics["p_obs"])
            ],
        }
    )


def _can_compute(df: pd.DataFrame, columns: set[str]) -> bool:
    return columns.issubset(df.columns)


def compute_patch_scores(scene: dict[str, pd.DataFrame], gap_config: dict[str, Any] | None = None) -> pd.DataFrame:
    required_tables = {"patches", "observations", "semantics", "materials"}
    missing_tables = required_tables.difference(scene)
    if missing_tables:
        raise ValueError(f"scene is missing tables: {', '.join(sorted(missing_tables))}")

    patches = scene["patches"].copy()
    observations = scene["observations"].copy()
    semantics = scene["semantics"].copy()
    materials = scene["materials"].copy()

    _validate_patch_table(patches, "patches")
    _validate_patch_table(observations, "observations", allow_empty=True)
    _validate_patch_table(semantics, "semantics", allow_empty=True)
    _validate_patch_table(materials, "materials", allow_empty=True)

    observations["_observation_row_present"] = True
    sem_scores = _compute_semantic_scores(semantics)
    df = patches.merge(
        observations,
        on="patch_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_obs"),
    )
    df = df.merge(sem_scores, on="patch_id", how="left", validate="one_to_one")
    df = df.merge(
        materials,
        on="patch_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_mat"),
    )
    if len(df) != len(patches):
        raise RuntimeError("patch scoring changed the patch row count")

    # D_mat: use pre-computed columns when available (real-data path),
    # otherwise compute from raw material_present / material_conflict flags.
    if "D_mat_missing" not in df.columns:
        if "material_present" in df.columns:
            present = df["material_present"].astype("boolean")
            df["D_mat_missing"] = present.map({True: 0.0, False: 1.0}).astype(float)
        else:
            df["D_mat_missing"] = np.nan
    if "D_mat_conflict" not in df.columns:
        if "material_conflict" in df.columns:
            conflict = df["material_conflict"].astype("boolean")
            df["D_mat_conflict"] = conflict.map({True: 1.0, False: 0.0}).astype(float)
        else:
            df["D_mat_conflict"] = np.nan

    # D_obs / D_ang / D_geo: use pre-computed columns when available (real-data path)
    if "D_obs" not in df.columns:
        obs_cols = {
            "directly_observed",
            "number_of_valid_views",
            "projected_resolution_quality",
            "registration_confidence",
        }
        if _can_compute(df, obs_cols):
            directly_observed = df["directly_observed"].astype("boolean")
            missing_observation = directly_observed.map({True: 0.0, False: 1.0}).astype(float)
            df["D_obs"] = np.clip(
                0.35 * missing_observation
                + 0.25 * np.clip((2 - df["number_of_valid_views"]) / 2, 0, 1)
                + 0.20 * (1 - df["projected_resolution_quality"])
                + 0.20 * (1 - df["registration_confidence"]),
                0,
                1,
            )
        else:
            df["D_obs"] = np.nan
    if "D_ang" not in df.columns:
        ang_cols = {"best_frontality", "number_of_valid_views", "angular_diversity"}
        if _can_compute(df, ang_cols):
            df["D_ang"] = np.clip(
                1.0
                - (
                    0.50 * df["best_frontality"]
                    + 0.30 * np.clip(df["number_of_valid_views"] / 2, 0, 1)
                    + 0.20 * df["angular_diversity"]
                ),
                0,
                1,
            )
        else:
            df["D_ang"] = np.nan
    if "D_geo" not in df.columns:
        if _can_compute(df, {"coverage_ratio", "density_ratio"}):
            df["D_geo"] = np.clip(
                0.60 * (1 - df["coverage_ratio"])
                + 0.40 * (1 - np.clip(df["density_ratio"], 0, 1)),
                0,
                1,
            )
        else:
            df["D_geo"] = np.nan

    observation_present = df["_observation_row_present"].eq(True)
    df.loc[~observation_present, "D_obs"] = 1.0

    # 升级路径: 若观测表带来 D_occ/D_rng/D_regsup(公式29/34/40), 启用扩展权重表;
    # 三列全部缺失时(母专利 B1 路径)行为与原实现逐位一致。
    extended_cols = [c for c in ("D_occ", "D_rng", "D_regsup") if c in df.columns]
    for component in [
        "D_sem",
        "D_mat_missing",
        "D_mat_conflict",
        "D_obs",
        "D_ang",
        "D_geo",
        *extended_cols,
    ]:
        df[component] = pd.to_numeric(df[component], errors="coerce").clip(0, 1)

    weights = normalize_alphas(gap_config or {}, extended=bool(extended_cols))
    components = [
        ("D_sem", "alpha_sem"),
        ("D_mat_missing", "alpha_mat_missing"),
        ("D_mat_conflict", "alpha_mat_conflict"),
        ("D_obs", "alpha_obs"),
        ("D_ang", "alpha_ang"),
        ("D_geo", "alpha_geo"),
    ] + [(c, "alpha_" + c[2:].lstrip("_")) for c in extended_cols]
    df["G_gap"] = df.apply(lambda row: _available_weighted_sum(row, components, weights), axis=1)
    lambda_importance = float((gap_config or {}).get("lambda_importance", 0.30))
    if "engineering_importance" not in df.columns:
        df["engineering_importance"] = 0.5
    df["engineering_importance"] = pd.to_numeric(
        df["engineering_importance"], errors="coerce"
    ).fillna(0.5).clip(0, 1)
    df["G_task"] = np.clip(df["G_gap"] * (1.0 + lambda_importance * df["engineering_importance"]), 0, 2)
    if "gt_missing" not in df.columns:
        df["gt_missing"] = False
    df["gt_missing"] = df["gt_missing"].fillna(False).astype(bool)

    # Normalise column names: resolve suffixed duplicates from merge
    for base in ("material_name", "centroid", "normal", "point_density",
                 "coverage_ratio", "density_ratio", "source_type", "visual_material_category"):
        if base not in df.columns:
            for suffix in ("_mat", "_obs", "_x", "_y"):
                candidate = base + suffix
                if candidate in df.columns:
                    df[base] = df[candidate]
                    break
            else:
                df[base] = None

    # centroid / normal may be stored as separate x/y/z columns in real-data path
    if "centroid" not in df.columns or df["centroid"].isna().all():
        if "centroid_x" in df.columns:
            df["centroid"] = list(zip(df["centroid_x"], df["centroid_y"], df["centroid_z"]))
    if "normal" not in df.columns or df["normal"].isna().all():
        if "normal_x" in df.columns:
            df["normal"] = list(zip(df["normal_x"], df["normal_y"], df["normal_z"]))

    ordered_cols = [
        "patch_id", "element_guid", "ifc_class", "name",
        "centroid", "normal", "area", "engineering_importance", "gt_missing",
        "D_sem", "D_mat_missing", "D_mat_conflict",
        "D_obs", "D_ang", "D_geo", "D_occ", "D_rng", "D_regsup",
        "G_gap", "G_task",
        "number_of_valid_views", "point_density",
        "coverage_ratio", "density_ratio",
        "material_name", "visual_material_category", "source_type",
    ]
    available = [c for c in ordered_cols if c in df.columns]
    return df[available].sort_values("G_task", ascending=False).reset_index(drop=True)


def rank_components(patch_scores: pd.DataFrame, gap_config: dict[str, Any] | None = None) -> pd.DataFrame:
    if patch_scores.empty:
        return pd.DataFrame(
            columns=[
                "element_guid",
                "ifc_class",
                "mean_gap",
                "max_gap",
                "p90_gap",
                "high_gap_area_ratio",
                "number_of_high_gap_patches",
                "critical_surface_gap",
                "engineering_importance",
                "G_component",
            ]
        )
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
