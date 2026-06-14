from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


def _frontality(normal: np.ndarray, centroid: np.ndarray, position: np.ndarray) -> float:
    return float(max(0.0, np.dot(_unit(normal), _unit(position - centroid))))


def generate_candidates(patch_scores: pd.DataFrame, view_config: dict[str, Any] | None = None) -> pd.DataFrame:
    view_config = view_config or {}
    distances = view_config.get("distances", [1.5, 2.5, 4.0])
    az_offsets = view_config.get("azimuth_offsets_deg", [-45, 0, 45])
    el_offsets = view_config.get("elevation_offsets_deg", [-20, 0, 20])
    threshold = float(view_config.get("gap_threshold", 0.55))
    high = patch_scores[patch_scores["G_gap"] >= threshold]
    if high.empty:
        high = patch_scores.head(3)
    rows = []
    cid = 0
    for patch in high.itertuples(index=False):
        centroid = np.asarray(patch.centroid if hasattr(patch, "centroid") else (0, 0, 0), dtype=float)
        normal = np.asarray(patch.normal if hasattr(patch, "normal") else (1, 0, 0), dtype=float)
        base = _unit(normal)
        tangent = _unit(np.cross(base, np.array([0.0, 0.0, 1.0])))
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.array([1.0, 0.0, 0.0])
        bitangent = _unit(np.cross(base, tangent))
        for distance, az, el in product(distances, az_offsets, el_offsets):
            direction = _unit(
                base
                + np.tan(np.deg2rad(float(az))) * 0.25 * tangent
                + np.tan(np.deg2rad(float(el))) * 0.25 * bitangent
            )
            position = centroid + float(distance) * direction
            orientation = _unit(centroid - position)
            rows.append(
                {
                    "view_id": f"candidate_{cid:04d}",
                    "target_patch_id": int(patch.patch_id),
                    "position": tuple(position.tolist()),
                    "orientation": tuple(orientation.tolist()),
                    "distance": float(distance),
                    "azimuth_offset_deg": float(az),
                    "elevation_offset_deg": float(el),
                }
            )
            cid += 1
    return pd.DataFrame(rows)


def score_candidates(patch_scores: pd.DataFrame, candidates: pd.DataFrame, view_config: dict[str, Any] | None = None) -> pd.DataFrame:
    eta = float((view_config or {}).get("eta", 0.10))
    rows = []
    for cand in candidates.itertuples(index=False):
        pos = np.asarray(cand.position, dtype=float)
        visible_patch_ids: list[int] = []
        visible_gap_area = 0.0
        value = 0.0
        quality_values = []
        for patch in patch_scores.itertuples(index=False):
            centroid = np.asarray(patch.centroid, dtype=float)
            normal = np.asarray(patch.normal, dtype=float)
            f = _frontality(normal, centroid, pos)
            if f <= 0.35:
                continue
            distance = float(np.linalg.norm(pos - centroid))
            distance_quality = float(np.exp(-((distance - 2.5) ** 2) / (2 * 2.0**2)))
            projected_resolution_quality = float(np.clip(1.0 / (1.0 + 0.15 * distance), 0.0, 1.0))
            q = f * distance_quality * projected_resolution_quality
            visible_patch_ids.append(int(patch.patch_id))
            if patch.G_gap >= float((view_config or {}).get("gap_threshold", 0.55)):
                visible_gap_area += float(patch.area)
            value += float(patch.G_gap * q * patch.area)
            quality_values.append(q)
        overlap = 0.0
        if visible_patch_ids:
            overlap = 0.1 * len([p for p in visible_patch_ids if p != cand.target_patch_id]) / len(visible_patch_ids)
        rows.append(
            {
                "view_id": cand.view_id,
                "target_patch_id": cand.target_patch_id,
                "position": cand.position,
                "orientation": cand.orientation,
                "value": value - eta * overlap,
                "visible_gap_area": visible_gap_area,
                "mean_quality": float(np.mean(quality_values)) if quality_values else 0.0,
                "overlap": overlap,
                "estimated_gain": max(0.0, value - eta * overlap),
                "visible_patch_ids": ";".join(map(str, visible_patch_ids)),
            }
        )
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

