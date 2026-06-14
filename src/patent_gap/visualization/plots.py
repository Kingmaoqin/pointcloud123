from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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

