"""
Semantic label layer.

Maps CRAS point-cloud classification labels and IFC element classes to a
shared semantic vocabulary, then computes the JS-divergence-based semantic
conflict score D_sem for every patch.

CRAS classification legend (from dataset README):
  0=unknown/unclassified, 1=wall, 2=floor, 3=ceiling, 4=door,
  5=window, 6=chair, 7=table, 8=furniture/other, 14=column,
  30=cable/pipe, 31=equipment
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from patent_gap.gap.scoring import js_divergence


SEMANTIC_LABELS = [
    "wall", "floor", "ceiling", "door", "window",
    "column", "furniture", "pipe", "equipment", "unknown",
]

# CRAS point-cloud classification code → semantic label
CRAS_TO_SEMANTIC: dict[int, str] = {
    0: "unknown",
    1: "wall",
    2: "floor",
    3: "ceiling",
    4: "door",
    5: "window",
    6: "furniture",   # chair
    7: "furniture",   # table
    8: "furniture",   # other furniture
    10: "unknown",
    11: "unknown",
    12: "unknown",
    14: "column",
    15: "unknown",
    17: "unknown",
    30: "pipe",
    31: "equipment",
}

# IFC class → semantic label
IFC_TO_SEMANTIC: dict[str, str] = {
    "IfcWall": "wall",
    "IfcWallStandardCase": "wall",
    "IfcSlab": "floor",          # floor/ceiling depends on normal
    "IfcCovering": "ceiling",
    "IfcDoor": "door",
    "IfcWindow": "window",
    "IfcColumn": "column",
    "IfcBeam": "column",
    "IfcFurnishingElement": "furniture",
    "IfcFlowSegment": "pipe",
    "IfcFlowFitting": "pipe",
    "IfcDistributionElement": "equipment",
    "IfcBuildingElementProxy": "equipment",
    "IfcOpeningElement": "unknown",
    "IfcStairFlight": "unknown",
}


def _label_to_dist(label: str, labels: list[str] = SEMANTIC_LABELS) -> np.ndarray:
    """One-hot distribution over semantic labels."""
    dist = np.zeros(len(labels), dtype=float)
    idx = labels.index(label) if label in labels else labels.index("unknown")
    dist[idx] = 1.0
    return dist


def _ifc_class_to_sem(ifc_class: str, normal_z: float = 0.0) -> str:
    if ifc_class == "IfcSlab":
        return "ceiling" if normal_z < -0.5 else "floor"
    return IFC_TO_SEMANTIC.get(ifc_class, "unknown")


def _semantic_distribution_from_counts(
    counts: Mapping[str, float],
    smoothing: float,
    ignore_unknown: bool,
) -> np.ndarray:
    dist = np.zeros(len(SEMANTIC_LABELS), dtype=float)
    for label, count in counts.items():
        if label not in SEMANTIC_LABELS or (ignore_unknown and label == "unknown"):
            continue
        dist[SEMANTIC_LABELS.index(label)] += float(count)
    total = float(dist.sum())
    if total <= 0:
        return np.full(len(SEMANTIC_LABELS), np.nan)
    eligible = np.ones(len(SEMANTIC_LABELS), dtype=bool)
    if ignore_unknown:
        eligible[SEMANTIC_LABELS.index("unknown")] = False
    dist[eligible] += smoothing
    return dist / float(dist.sum())


def _semantic_distribution(
    labels: pd.Series,
    smoothing: float,
    ignore_unknown: bool,
) -> np.ndarray:
    return _semantic_distribution_from_counts(
        labels.value_counts().to_dict(),
        smoothing,
        ignore_unknown,
    )


def compute_semantic_scores(
    patches_df: pd.DataFrame,
    assoc_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compute D_sem for every patch.

    For each patch:
    - p_bim : one-hot distribution from IFC element class
    - p_obs : empirical distribution of CRAS classification labels for matched
              points whose element_guid matches this patch's element_guid.
              If no observed points → D_sem = NaN (not 0 or 1).

    Returns DataFrame with: patch_id, p_bim, p_obs, bim_label, obs_label, D_sem
    """
    cfg = config or {}
    smoothing = float(cfg.get("semantic_smoothing", 0.02))
    ignore_unknown = bool(cfg.get("ignore_unknown_labels", True))

    # Build observed semantic distributions per element_guid
    matched = assoc_df[assoc_df["matched"]].copy() if "matched" in assoc_df.columns else assoc_df.copy()
    matched["semantic"] = matched["classification"].map(CRAS_TO_SEMANTIC).fillna("unknown")

    elem_obs_dists: dict[str, np.ndarray] = {}
    if len(matched) > 0 and "element_guid" in matched.columns:
        for guid, grp in matched.groupby("element_guid"):
            elem_obs_dists[str(guid)] = _semantic_distribution(
                grp["semantic"],
                smoothing,
                ignore_unknown,
            )

    rows: list[dict] = []
    for patch in patches_df.itertuples():
        guid = str(patch.element_guid)
        ifc_class = str(patch.ifc_class)
        normal_z = float(patch.normal_z)

        bim_label = _ifc_class_to_sem(ifc_class, normal_z)
        p_bim = _label_to_dist(bim_label)

        p_obs = elem_obs_dists.get(guid, np.full(len(SEMANTIC_LABELS), np.nan))
        d_sem = js_divergence(p_bim, p_obs)  # NaN if p_obs is all NaN

        obs_label: str | None = None
        if not np.isnan(p_obs).any():
            obs_label = SEMANTIC_LABELS[int(np.argmax(p_obs))]

        rows.append({
            "patch_id": int(patch.patch_id),
            "bim_label": bim_label,
            "obs_label": obs_label,
            "p_bim": p_bim,
            "p_obs": p_obs,
            "D_sem": d_sem,
        })

    return pd.DataFrame(rows)


def compute_semantic_scores_synthetic(
    patches_df: pd.DataFrame,
    elem_stats_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compute D_sem from synthetic scan semantic_counts column.

    elem_stats_df must have:
        element_guid, semantic_counts (JSON string {classification_code: count})

    semantic_counts codes come from IFC_TO_CODE in gen_synthetic_scan.py,
    which mirror the CRAS classification scheme, so CRAS_TO_SEMANTIC applies.
    """
    cfg = config or {}
    smoothing = float(cfg.get("semantic_smoothing", 0.02))
    ignore_unknown = bool(cfg.get("ignore_unknown_labels", True))

    # Build per-element or per-patch observed semantic distributions.
    elem_obs: dict[str, np.ndarray] = {}
    patch_obs: dict[int, np.ndarray] = {}
    for row in elem_stats_df.itertuples(index=False):
        try:
            sem_counts: dict = json.loads(str(row.semantic_counts))
        except Exception:
            continue
        label_counts: dict[str, float] = {}
        for code_str, cnt in sem_counts.items():
            label = CRAS_TO_SEMANTIC.get(int(code_str), "unknown")
            label_counts[label] = label_counts.get(label, 0.0) + float(cnt)
        distribution = _semantic_distribution_from_counts(
            label_counts,
            smoothing,
            ignore_unknown,
        )
        if hasattr(row, "patch_id"):
            patch_obs[int(row.patch_id)] = distribution
        else:
            elem_obs[str(row.element_guid)] = distribution

    # Unpack normal_z for IfcSlab orientation
    def _normal_z(patch) -> float:
        if hasattr(patch, "normal_z"):
            return float(patch.normal_z)
        n = getattr(patch, "normal", None)
        if n is not None and hasattr(n, "__len__"):
            return float(n[2])
        return 0.0

    rows: list[dict] = []
    for patch in patches_df.itertuples(index=False):
        guid      = str(patch.element_guid)
        ifc_class = str(patch.ifc_class)
        nz        = _normal_z(patch)

        bim_label = _ifc_class_to_sem(ifc_class, nz)
        p_bim     = _label_to_dist(bim_label)
        p_obs = patch_obs.get(
            int(patch.patch_id),
            elem_obs.get(guid, np.full(len(SEMANTIC_LABELS), np.nan)),
        )

        d_sem = js_divergence(p_bim, p_obs)

        obs_label: str | None = None
        if not np.isnan(p_obs).any():
            obs_label = SEMANTIC_LABELS[int(np.argmax(p_obs))]

        rows.append({
            "patch_id":  int(patch.patch_id),
            "bim_label": bim_label,
            "obs_label": obs_label,
            "p_bim":     p_bim,
            "p_obs":     p_obs,
            "D_sem":     d_sem,
        })

    return pd.DataFrame(rows)


def compute_controlled_semantic_scores(
    patches_df: pd.DataFrame,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Create semantic evidence independent from the controlled missing labels."""
    cfg = config or {}
    noise = float(cfg.get("semantic_noise", 0.12))
    conflict_fraction = float(cfg.get("semantic_conflict_fraction", 0.05))
    if not 0 <= noise < 1 or not 0 <= conflict_fraction <= 1:
        raise ValueError("semantic noise and conflict fraction must be valid probabilities")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    known_labels = [label for label in SEMANTIC_LABELS if label != "unknown"]

    for patch in patches_df.itertuples(index=False):
        normal_z = float(getattr(patch, "normal_z", 0.0))
        bim_label = _ifc_class_to_sem(str(patch.ifc_class), normal_z)
        p_bim = _label_to_dist(bim_label)
        if bool(patch.gt_missing):
            p_obs = np.full(len(SEMANTIC_LABELS), np.nan)
            obs_label = None
        else:
            obs_label = bim_label
            if rng.random() < conflict_fraction:
                alternatives = [label for label in known_labels if label != bim_label]
                if alternatives:
                    obs_label = str(rng.choice(alternatives))
            p_obs = np.full(len(SEMANTIC_LABELS), noise / (len(SEMANTIC_LABELS) - 1))
            p_obs[SEMANTIC_LABELS.index(obs_label)] = 1.0 - noise
        rows.append(
            {
                "patch_id": int(patch.patch_id),
                "bim_label": bim_label,
                "obs_label": obs_label,
                "p_bim": p_bim,
                "p_obs": p_obs,
                "D_sem": js_divergence(p_bim, p_obs),
            }
        )
    return pd.DataFrame(rows)
