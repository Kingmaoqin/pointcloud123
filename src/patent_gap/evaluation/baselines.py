from __future__ import annotations

import numpy as np
import pandas as pd

from patent_gap.evaluation.metrics import patch_detection_metrics


def score_baselines(patch_scores: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    methods = {
        "Random view": rng.random(len(patch_scores)),
        "Geometry-only NBV": patch_scores["D_geo"].to_numpy(),
        "Evidence + geometry": (0.5 * patch_scores["D_obs"] + 0.5 * patch_scores["D_geo"]).to_numpy(),
        "Equal-weight patent score": patch_scores[["D_sem", "D_mat_missing", "D_mat_conflict", "D_obs", "D_ang", "D_geo"]].mean(axis=1, skipna=True).to_numpy(),
        "Tuned patent score": patch_scores["G_gap"].to_numpy(),
    }
    rows = []
    for method, scores in methods.items():
        tmp = patch_scores.copy()
        tmp["baseline_score"] = scores
        metrics = patch_detection_metrics(tmp, "baseline_score")
        metrics["method"] = method
        rows.append(metrics)
    return pd.DataFrame(rows)


def ablation_table(patch_scores: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["D_sem", "D_mat_missing", "D_mat_conflict", "D_obs", "D_ang", "D_geo"]
    rows = []
    variants = {
        "full": base_cols,
        "no_D_sem": [c for c in base_cols if c != "D_sem"],
        "no_material": [c for c in base_cols if c not in {"D_mat_missing", "D_mat_conflict"}],
        "no_D_obs": [c for c in base_cols if c != "D_obs"],
        "no_D_ang": [c for c in base_cols if c != "D_ang"],
        "no_D_geo": [c for c in base_cols if c != "D_geo"],
        "equal_weights": base_cols,
        "only_visible_area": ["D_geo"],
        "no_quality_Q": ["D_obs", "D_geo"],
    }
    for name, cols in variants.items():
        tmp = patch_scores.copy()
        tmp["ablation_score"] = tmp[cols].mean(axis=1, skipna=True)
        metrics = patch_detection_metrics(tmp, "ablation_score")
        metrics["ablation"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)

