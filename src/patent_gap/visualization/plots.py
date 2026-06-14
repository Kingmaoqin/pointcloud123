from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_patch_bars(patch_scores: pd.DataFrame, out_path: str | Path, column: str = "G_gap") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = patch_scores.sort_values(column, ascending=False)
    ax.bar(ordered["patch_id"].astype(str), ordered[column], color="#4477AA")
    ax.set_xlabel("patch_id")
    ax.set_ylabel(column)
    ax.set_ylim(0, max(1.0, float(ordered[column].max()) * 1.1))
    ax.set_title(f"Patch {column}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_component_ranking(component_ranking: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ordered = component_ranking.sort_values("G_component", ascending=True)
    ax.barh(ordered["element_guid"], ordered["G_component"], color="#228833")
    ax.set_xlabel("G_component")
    ax.set_title("Component Verification Priority")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_closed_loop(history: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["step"], history["remaining_gap_area"], marker="o", label="remaining gap area")
    ax.plot(history["step"], history["recovered_missing_area"], marker="s", label="recovered missing area")
    ax.set_xlabel("supplemental scan iteration")
    ax.set_ylabel("area-weighted value")
    ax.legend()
    ax.set_title("Closed-loop Supplemental Scan")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_baseline_results(results: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    metric = "AUPRC" if "AUPRC" in results.columns else results.columns[0]
    labels = results["method"] if "method" in results.columns else results.iloc[:, 0]
    ax.bar(labels, results[metric], color="#CC6677")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Baseline Comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_candidate_views(ranked_views: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    if ranked_views.empty:
        ax.text(0.5, 0.5, "no candidates", ha="center", va="center")
    else:
        xy = np.array([list(v[:2]) for v in ranked_views["position"]], dtype=float)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=ranked_views["value"], cmap="viridis", s=35)
        ax.scatter([0], [0], marker="s", c="black", s=80, label="cube center")
        ax.legend(loc="best")
        fig.colorbar(sc, ax=ax, label="view value")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Candidate View Positions")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ablation_results(results: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    metric = "AUPRC"
    grouped = results.groupby("ablation", as_index=False)[metric].mean() if "seed" in results.columns else results
    ax.bar(grouped["ablation"], grouped[metric], color="#AA3377")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Ablation Results")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_parameter_importance(trials: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = [c for c in trials.columns if c.startswith("alpha_") or c == "high_gap_threshold"]
    importances = []
    for param in params:
        if trials[param].nunique() <= 1:
            score = 0.0
        else:
            score = abs(float(np.corrcoef(trials[param], trials["objective"])[0, 1]))
            if np.isnan(score):
                score = 0.0
        importances.append({"parameter": param, "importance": score})
    df = pd.DataFrame(importances).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["parameter"], df["importance"], color="#66AAEE")
    ax.set_ylabel("|corr(parameter, objective)|")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Parameter Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
