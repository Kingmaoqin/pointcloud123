from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from patent_gap.gap.scoring import compute_patch_scores
from patent_gap.simulation.synthetic_cube import synthetic_scene
from patent_gap.viewpoints.ranking import generate_candidates, score_candidates


def _simulate_new_observation(
    obs: pd.DataFrame,
    visible_patch_ids: set[int],
    patch_scores: pd.DataFrame,
    recovery_per_view: float,
) -> pd.DataFrame:
    """
    Update the observation table after adding a new scan viewpoint.

    For each newly visible patch:
    - Mark as directly observed
    - Increment view counts
    - Improve frontality (clamp upward)
    - Improve coverage_ratio and density_ratio using exponential recovery:
        new_value = old + recovery_rate * (1 - old)
      This is physically meaningful: each new scan recovers a fraction of the
      remaining gap, with diminishing returns.
    - Reset source_type to MEASURED
    """
    if not 0 <= recovery_per_view <= 1:
        raise ValueError("recovery_per_view must be in [0, 1]")
    obs = obs.copy()
    defaults: dict[str, Any] = {
        "directly_observed": False,
        "number_of_views": 0,
        "number_of_valid_views": 0,
        "best_frontality": 0.0,
        "mean_frontality": 0.0,
        "angular_diversity": 0.0,
        "coverage_ratio": 0.0,
        "density_ratio": 0.0,
        "projected_resolution_quality": 0.0,
        "registration_confidence": 0.0,
        "source_type": "MODEL",
    }
    for column, default in defaults.items():
        if column not in obs.columns:
            obs[column] = default

    mask = obs["patch_id"].isin(visible_patch_ids)
    if not mask.any():
        return obs

    gap_by_patch = patch_scores.set_index("patch_id")["G_gap"]
    visible_gap = (
        obs.loc[mask, "patch_id"].map(gap_by_patch).fillna(0.5).astype(float).clip(0, 1)
    )
    effective_recovery = (recovery_per_view * (0.5 + visible_gap)).clip(0, 1).to_numpy()

    obs.loc[mask, "directly_observed"] = True
    for column in ("number_of_views", "number_of_valid_views"):
        current = pd.to_numeric(obs.loc[mask, column], errors="coerce").fillna(0)
        obs.loc[mask, column] = current + 1

    best = pd.to_numeric(obs.loc[mask, "best_frontality"], errors="coerce").fillna(0)
    mean = pd.to_numeric(obs.loc[mask, "mean_frontality"], errors="coerce").fillna(0)
    diversity = pd.to_numeric(obs.loc[mask, "angular_diversity"], errors="coerce").fillna(0)
    obs.loc[mask, "best_frontality"] = np.maximum(best.to_numpy(), 0.80)
    obs.loc[mask, "mean_frontality"] = np.maximum(mean.to_numpy(), 0.65)
    obs.loc[mask, "angular_diversity"] = np.clip(diversity.to_numpy() + 0.20, 0, 1)

    for column in ("coverage_ratio", "density_ratio"):
        current = pd.to_numeric(obs.loc[mask, column], errors="coerce").fillna(0).clip(0, 1)
        obs.loc[mask, column] = np.clip(
            current.to_numpy() + effective_recovery * (1.0 - current.to_numpy()),
            0,
            1,
        )

    obs.loc[mask, "projected_resolution_quality"] = np.maximum(
        pd.to_numeric(obs.loc[mask, "projected_resolution_quality"], errors="coerce")
        .fillna(0)
        .to_numpy(),
        0.85,
    )
    obs.loc[mask, "registration_confidence"] = np.maximum(
        pd.to_numeric(obs.loc[mask, "registration_confidence"], errors="coerce")
        .fillna(0)
        .to_numpy(),
        0.95,
    )
    obs.loc[mask, "source_type"] = "MEASURED"

    valid_views = pd.to_numeric(obs["number_of_valid_views"], errors="coerce").fillna(0)
    best_frontality = pd.to_numeric(obs["best_frontality"], errors="coerce")
    angular_diversity = pd.to_numeric(obs["angular_diversity"], errors="coerce")
    projected_quality = pd.to_numeric(
        obs["projected_resolution_quality"], errors="coerce"
    )
    registration_confidence = pd.to_numeric(
        obs["registration_confidence"], errors="coerce"
    )
    coverage = pd.to_numeric(obs["coverage_ratio"], errors="coerce")
    density = pd.to_numeric(obs["density_ratio"], errors="coerce")
    directly_observed = obs["directly_observed"].astype("boolean")
    missing_observation = directly_observed.map({True: 0.0, False: 1.0}).astype(float)

    computed_obs = np.clip(
        0.35 * missing_observation.to_numpy()
        + 0.25 * np.clip((2.0 - valid_views.to_numpy()) / 2.0, 0, 1)
        + 0.20 * (1.0 - projected_quality.to_numpy())
        + 0.20 * (1.0 - registration_confidence.to_numpy()),
        0,
        1,
    )
    computed_ang = np.clip(
        1.0
        - (
            0.50 * best_frontality.to_numpy()
            + 0.30 * np.clip(valid_views.to_numpy() / 2.0, 0, 1)
            + 0.20 * angular_diversity.to_numpy()
        ),
        0,
        1,
    )
    computed_geo = np.clip(
        0.60 * (1.0 - coverage.to_numpy())
        + 0.40 * (1.0 - np.clip(density.to_numpy(), 0, 1)),
        0,
        1,
    )
    for column, values in (
        ("D_obs", computed_obs),
        ("D_ang", computed_ang),
        ("D_geo", computed_geo),
    ):
        if column not in obs.columns:
            obs[column] = values
        else:
            obs.loc[mask, column] = values[mask.to_numpy()]
    return obs


def apply_supplemental_observation(
    observations: pd.DataFrame,
    visible_patch_ids: set[int],
    patch_scores: pd.DataFrame,
    recovery_per_view: float,
) -> pd.DataFrame:
    return _simulate_new_observation(
        observations,
        visible_patch_ids,
        patch_scores,
        recovery_per_view,
    )


def _loop_config(config: dict[str, Any] | None) -> tuple[int, float, dict[str, Any], dict[str, Any]]:
    cfg = config or {}
    simulation_cfg = cfg.get("simulation", cfg)
    steps = int(simulation_cfg.get("steps", 10))
    recovery = float(simulation_cfg.get("recovery_per_view", 0.75))
    if steps < 0:
        raise ValueError("steps cannot be negative")
    return steps, recovery, dict(cfg.get("view", {})), dict(cfg.get("gap", {}))


def _gap_area(patch_scores: pd.DataFrame, target_patch_ids: set[int]) -> float:
    target = patch_scores["patch_id"].isin(target_patch_ids)
    return float(
        (
            patch_scores.loc[target, "G_gap"].fillna(0)
            * patch_scores.loc[target, "area"].fillna(0)
        ).sum()
    )


def _run_scene_loop(
    scene: dict[str, pd.DataFrame],
    config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    steps, recovery_per_view, view_cfg, gap_cfg = _loop_config(config)
    work_scene = {name: table.copy(deep=True) for name, table in scene.items()}
    initial_scores = compute_patch_scores(work_scene, gap_cfg)
    gt_ids = set(
        initial_scores.loc[initial_scores["gt_missing"].astype(bool), "patch_id"].astype(int)
    )
    target_patch_ids = gt_ids or set(initial_scores["patch_id"].astype(int))
    target_basis = "gt_missing" if gt_ids else "all_patches"
    initial_gap_area = _gap_area(initial_scores, target_patch_ids)
    total_target_area = float(
        initial_scores.loc[
            initial_scores["patch_id"].isin(target_patch_ids), "area"
        ].fillna(0).sum()
    )

    history: list[dict[str, Any]] = []
    all_rankings: list[pd.DataFrame] = []
    chosen: list[dict[str, Any]] = []
    selected_view_ids: set[str] = set()

    for step_index in range(steps):
        before_scores = compute_patch_scores(work_scene, gap_cfg)
        before_gap_area = _gap_area(before_scores, target_patch_ids)
        candidates = generate_candidates(before_scores, view_cfg)
        ranked = score_candidates(before_scores, candidates, view_cfg)
        if selected_view_ids and not ranked.empty:
            ranked = ranked[~ranked["view_id"].isin(selected_view_ids)].reset_index(drop=True)
        if ranked.empty:
            break

        ranked = ranked.copy()
        ranked["loop_step"] = step_index + 1
        all_rankings.append(ranked)
        best = ranked.iloc[0]
        selected_view_ids.add(str(best["view_id"]))
        chosen.append(best.to_dict())
        visible = {
            int(value)
            for value in str(best.get("visible_patch_ids", "")).split(";")
            if value.strip()
        }

        high_threshold = float(gap_cfg.get("high_gap_threshold", view_cfg.get("gap_threshold", 0.55)))
        visible_scores = before_scores[before_scores["patch_id"].isin(visible)]
        high_gap_visible = int((visible_scores["G_gap"] >= high_threshold).sum())
        gt_missing_visible = int(visible_scores["gt_missing"].astype(bool).sum())

        work_scene["observations"] = apply_supplemental_observation(
            work_scene["observations"],
            visible,
            before_scores,
            recovery_per_view,
        )
        after_scores = compute_patch_scores(work_scene, gap_cfg)
        after_gap_area = _gap_area(after_scores, target_patch_ids)
        recovered_gap_area = max(0.0, initial_gap_area - after_gap_area)
        recovery_rate = (
            float(np.clip(recovered_gap_area / initial_gap_area, 0, 1))
            if initial_gap_area > 0
            else 0.0
        )

        history.append(
            {
                "step": step_index + 1,
                "selected_view_id": str(best["view_id"]),
                "remaining_gap_area_before": before_gap_area,
                "remaining_gap_area": after_gap_area,
                "recovered_missing_area": recovered_gap_area,
                "recovery_rate": recovery_rate,
                "total_missing_area": total_target_area,
                "target_basis": target_basis,
                "semantic_conflict_resolved": 0,
                "material_evidence_added": 0,
                "gt_missing_patches_covered": gt_missing_visible,
                "high_gap_patches_visible": high_gap_visible,
                "angle_coverage_improvement": float(len(visible)),
                "number_of_views": step_index + 1,
                "cumulative_redundancy": float(
                    sum(float(item.get("overlap", 0.0)) for item in chosen)
                ),
                "view_value": float(best.get("value", 0.0)),
                "runtime_s": 0.0,
            }
        )

    history_df = pd.DataFrame(history)
    rankings_df = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    chosen_df = pd.DataFrame(chosen)
    return history_df, rankings_df, chosen_df


def run_closed_loop(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(config.get("seed", 0))
    return _run_scene_loop(synthetic_scene(seed), config)


def run_closed_loop_from_scene(
    scene: dict[str, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _run_scene_loop(scene, config)
