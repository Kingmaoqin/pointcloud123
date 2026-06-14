from __future__ import annotations

from typing import Any

import pandas as pd

from patent_gap.gap.scoring import compute_patch_scores
from patent_gap.simulation.synthetic_cube import synthetic_scene
from patent_gap.viewpoints.ranking import generate_candidates, score_candidates


def run_closed_loop(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(config.get("seed", 0))
    steps = int(config.get("simulation", {}).get("steps", 5))
    recovery_per_view = float(config.get("simulation", {}).get("recovery_per_view", 0.85))
    scene = synthetic_scene(seed)
    history = []
    all_rankings = []
    chosen = []
    selected_view_ids: set[str] = set()
    for step in range(steps):
        patch_scores = compute_patch_scores(scene, config.get("gap", {}))
        remaining_gap_area = float((patch_scores["G_gap"] * patch_scores["area"]).sum())
        recovered_missing_area = float(((1.0 - patch_scores["D_geo"]) * patch_scores["area"] * patch_scores["gt_missing"]).sum())
        candidates = generate_candidates(patch_scores, config.get("view", {}))
        ranked = score_candidates(patch_scores, candidates, config.get("view", {}))
        if selected_view_ids and not ranked.empty:
            ranked = ranked[~ranked["view_id"].isin(selected_view_ids)].reset_index(drop=True)
        ranked["loop_step"] = step
        all_rankings.append(ranked)
        if ranked.empty:
            break
        best = ranked.iloc[0]
        selected_view_ids.add(str(best["view_id"]))
        chosen.append(best.to_dict())
        visible = {int(x) for x in str(best["visible_patch_ids"]).split(";") if x != ""}
        obs = scene["observations"].copy()
        for patch_id in visible:
            mask = obs["patch_id"] == patch_id
            obs.loc[mask, "directly_observed"] = True
            obs.loc[mask, "number_of_views"] += 1
            obs.loc[mask, "number_of_valid_views"] += 1
            obs.loc[mask, "best_frontality"] = obs.loc[mask, "best_frontality"].clip(lower=0.75)
            obs.loc[mask, "mean_frontality"] = obs.loc[mask, "mean_frontality"].clip(lower=0.65)
            obs.loc[mask, "angular_diversity"] = (obs.loc[mask, "angular_diversity"] + 0.30).clip(upper=1.0)
            obs.loc[mask, "coverage_ratio"] = (obs.loc[mask, "coverage_ratio"] + recovery_per_view * (1 - obs.loc[mask, "coverage_ratio"])).clip(upper=1.0)
            obs.loc[mask, "density_ratio"] = (obs.loc[mask, "density_ratio"] + recovery_per_view * (1 - obs.loc[mask, "density_ratio"])).clip(upper=1.0)
            obs.loc[mask, "projected_resolution_quality"] = 0.85
            obs.loc[mask, "registration_confidence"] = 0.95
            obs.loc[mask, "source_type"] = "MEASURED"
        scene["observations"] = obs
        history.append(
            {
                "step": step,
                "selected_view_id": best["view_id"],
                "remaining_gap_area": remaining_gap_area,
                "recovered_missing_area": recovered_missing_area,
                "semantic_conflict_resolved": 0,
                "material_evidence_added": int(2 in visible),
                "angle_coverage_improvement": float(len(visible)),
                "number_of_views": step + 1,
                "cumulative_redundancy": float(sum(float(x.get("overlap", 0.0)) for x in chosen)),
                "runtime_s": 0.0,
            }
        )
    history_df = pd.DataFrame(history)
    rankings_df = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    chosen_df = pd.DataFrame(chosen)
    return history_df, rankings_df, chosen_df
