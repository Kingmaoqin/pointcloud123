from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_top_fraction(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    k = max(1, int(np.ceil(len(scores) * fraction)))
    order = np.argsort(scores)[::-1][:k]
    positives = labels.sum()
    if positives <= 0:
        return 0.0
    return float(labels[order].sum() / positives)


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    scores = np.clip(scores, 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(labels[mask].mean() - scores[mask].mean())
    return float(ece)


def patch_detection_metrics(patch_scores: pd.DataFrame, score_col: str = "G_gap") -> dict[str, float]:
    labels = patch_scores["gt_missing"].astype(int).to_numpy()
    scores = patch_scores[score_col].fillna(0).to_numpy()
    pred = (scores >= 0.5).astype(int)
    out: dict[str, float] = {}
    if len(np.unique(labels)) > 1:
        out["AUROC"] = float(roc_auc_score(labels, scores))
        out["AUPRC"] = float(average_precision_score(labels, scores))
    else:
        out["AUROC"] = 0.0
        out["AUPRC"] = 0.0
    out["F1"] = float(f1_score(labels, pred, zero_division=0))
    intersection = float(((labels == 1) & (pred == 1)).sum())
    union = float(((labels == 1) | (pred == 1)).sum())
    out["IoU"] = intersection / union if union else 0.0
    out["Recall@Top-5%"] = recall_at_top_fraction(labels, scores, 0.05)
    out["Recall@Top-10%"] = recall_at_top_fraction(labels, scores, 0.10)
    out["Brier"] = float(brier_score_loss(labels, np.clip(scores, 0, 1)))
    out["ECE"] = expected_calibration_error(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    out["PR_points"] = float(len(precision) + len(recall))
    return out


def component_metrics(component_ranking: pd.DataFrame, patch_scores: pd.DataFrame) -> dict[str, float]:
    missing_by_component = patch_scores.groupby("element_guid")["gt_missing"].max().astype(float)
    merged = component_ranking.join(missing_by_component, on="element_guid", rsuffix="_rel")
    relevance = merged["gt_missing"].fillna(0).to_numpy()
    scores = merged["G_component"].to_numpy()
    return {
        "NDCG@5": ndcg_at_k(relevance, scores, 5),
        "NDCG@10": ndcg_at_k(relevance, scores, 10),
        "Recall@5": float(relevance[np.argsort(scores)[::-1][:5]].sum() / max(1, relevance.sum())),
        "Recall@10": float(relevance[np.argsort(scores)[::-1][:10]].sum() / max(1, relevance.sum())),
    }

