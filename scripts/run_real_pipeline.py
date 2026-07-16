#!/usr/bin/env python3
"""
End-to-end CRAS real-data pipeline.

Steps:
  1. Segment IFC mesh into surface patches (region-growing)
  2. Aggregate element-level evidence from full-dataset chunk summaries
  3. Compute D_obs / D_ang / D_geo per patch
  4. Compute D_sem from CRAS classification labels
  5. Compute D_mat_missing / D_mat_conflict from IFC material data
  6. Merge into G_gap and G_task scores
  7. Rank components and generate candidate viewpoints
  8. Run baselines and ablation
  9. Run closed-loop simulation
  10. Generate interactive HTML visualization + all static figures
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from patent_gap.patches import build_patches
from patent_gap.evidence import compute_evidence
from patent_gap.semantics import compute_semantic_scores
from patent_gap.materials import build_material_table
from patent_gap.gap.scoring import compute_patch_scores, rank_components
from patent_gap.evaluation.baselines import ablation_table, score_baselines
from patent_gap.viewpoints.ranking import generate_candidates, score_candidates, greedy_sequential_ranking
from patent_gap.simulation.closed_loop import run_closed_loop_from_scene
from patent_gap.visualization.plots import (
    plot_patch_bars, plot_component_ranking, plot_closed_loop,
    plot_baseline_results, plot_candidate_views, plot_ablation_results,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
REPORTS = OUTPUTS / "reports"


def ensure_dirs() -> None:
    for d in [FIGURES, TABLES, REPORTS]:
        d.mkdir(parents=True, exist_ok=True)


# ─── Step 1: Patch segmentation ───────────────────────────────────────────────
def step1_patches(config: dict) -> pd.DataFrame:
    print("\n[1/10] Segmenting IFC mesh into surface patches …")
    t0 = time.time()
    patches_path = PROCESSED / "patches_real.parquet"
    patches_df = build_patches(
        mesh_npz_path=PROCESSED / "ifc_mesh.npz",
        triangle_element_map_path=PROCESSED / "triangle_element_map.npy",
        elements_parquet_path=PROCESSED / "ifc_elements.parquet",
        materials_parquet_path=PROCESSED / "ifc_materials.parquet",
        output_path=patches_path,
        config=config.get("patches", {}),
    )
    elapsed = time.time() - t0
    print(f"  → {len(patches_df):,} patches from {patches_df['element_guid'].nunique()} elements  [{elapsed:.1f}s]")
    return patches_df


# ─── Step 2-3: Evidence aggregation ───────────────────────────────────────────
def step2_evidence(patches_df: pd.DataFrame, elements_df: pd.DataFrame,
                   assoc_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    print("\n[2/10] Computing observation evidence (D_obs, D_ang, D_geo) …")
    t0 = time.time()
    evidence_df = compute_evidence(
        patches_df=patches_df,
        elements_df=elements_df,
        assoc_df=assoc_df,
        chunk_dir=PROCESSED / "cras_full_assoc",
        config=config.get("evidence", {}),
    )
    elapsed = time.time() - t0
    print(f"  → {len(evidence_df):,} patches with evidence  [{elapsed:.1f}s]")
    print(f"     Directly observed: {evidence_df['directly_observed'].sum()} / {len(evidence_df)}")
    print(f"     Mean D_obs={evidence_df['D_obs'].mean():.3f}  D_ang={evidence_df['D_ang'].mean():.3f}  D_geo={evidence_df['D_geo'].mean():.3f}")
    return evidence_df


# ─── Step 4: Semantic scores ───────────────────────────────────────────────────
def step4_semantics(patches_df: pd.DataFrame, assoc_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    print("\n[3/10] Computing semantic conflict scores (D_sem) …")
    t0 = time.time()
    sem_df = compute_semantic_scores(patches_df, assoc_df, config.get("semantics", {}))
    elapsed = time.time() - t0
    nan_count = sem_df["D_sem"].isna().sum()
    valid = sem_df["D_sem"].dropna()
    print(f"  → D_sem computed: {len(valid)} valid, {nan_count} NA (no obs)  [{elapsed:.1f}s]")
    if len(valid) > 0:
        print(f"     Mean D_sem={valid.mean():.3f}  max={valid.max():.3f}")
    return sem_df


# ─── Step 5: Material scores ───────────────────────────────────────────────────
def step5_materials(patches_df: pd.DataFrame, elements_df: pd.DataFrame,
                    materials_df: pd.DataFrame, assoc_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    print("\n[4/10] Computing material attribute scores (D_mat) …")
    t0 = time.time()
    mat_df = build_material_table(patches_df, elements_df, materials_df, assoc_df,
                                   config.get("materials", {}))
    elapsed = time.time() - t0
    missing_frac = mat_df["D_mat_missing"].mean()
    conflict_frac = mat_df["D_mat_conflict"].mean()
    print(f"  → Material missing rate: {missing_frac:.1%}  conflict rate: {conflict_frac:.1%}  [{elapsed:.1f}s]")
    return mat_df


# ─── Step 6: Gap scoring ───────────────────────────────────────────────────────
def step6_gap_scores(patches_df: pd.DataFrame, evidence_df: pd.DataFrame,
                     sem_df: pd.DataFrame, mat_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    print("\n[5/10] Computing G_gap / G_task scores …")
    t0 = time.time()

    # Preserve the evidence layer's precomputed D_* values. In particular,
    # fused CRAS data has no scanner origins, so D_obs deliberately excludes
    # unavailable angular/resolution terms instead of propagating NaN.
    obs_cols = [
        "patch_id", "directly_observed", "number_of_views",
        "number_of_valid_views", "best_frontality", "mean_frontality",
        "angular_diversity", "coverage_ratio", "density_ratio",
        "projected_resolution_quality", "registration_confidence",
        "point_density", "source_type", "D_obs", "D_ang", "D_geo",
    ]
    obs_sub = evidence_df[[c for c in obs_cols if c in evidence_df.columns]].copy()

    # Real CRAS has no patch-level ground-truth missing labels.
    pat_sub = patches_df.copy()

    # Build semantics table as expected
    sem_sub = sem_df[["patch_id", "p_bim", "p_obs"]].copy()

    # Build materials table as expected
    mat_sub = mat_df[["patch_id", "material_present", "material_name",
                       "material_conflict", "visual_material_category",
                       "D_mat_missing", "D_mat_conflict"]].copy()

    scene = {
        "patches": pat_sub,
        "observations": obs_sub,
        "semantics": sem_sub,
        "materials": mat_sub,
    }

    gap_config = config.get("gap", {})
    patch_scores = compute_patch_scores(scene, gap_config)
    elapsed = time.time() - t0
    print(f"  → {len(patch_scores):,} patches scored  [{elapsed:.1f}s]")
    print(f"     Mean G_gap={patch_scores['G_gap'].mean():.3f}  max={patch_scores['G_gap'].max():.3f}")
    print(f"     High-gap patches (>0.55): {(patch_scores['G_gap'] >= 0.55).sum()}")

    # Save
    patch_scores.to_csv(TABLES / "patch_scores_real.csv", index=False)
    print(f"     Saved → {TABLES / 'patch_scores_real.csv'}")
    return patch_scores


# ─── Step 7: Component ranking ────────────────────────────────────────────────
def step7_components(patch_scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    print("\n[6/10] Ranking components …")
    comp = rank_components(patch_scores, config.get("gap", {}))
    comp.to_csv(TABLES / "component_ranking_real.csv", index=False)
    print(f"  → {len(comp)} components ranked")
    print("     Top-3 by G_component:")
    for row in comp.head(3).itertuples():
        print(f"       {row.element_guid[:20]} ({row.ifc_class})  G_comp={row.G_component:.3f}")
    return comp


# ─── Step 8: Baselines and ablation ───────────────────────────────────────────
def step8_baselines(patch_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[7/10] Running baselines and ablation …")
    labels = patch_scores.get("gt_missing")
    has_binary_ground_truth = (
        labels is not None
        and labels.notna().any()
        and labels.astype(bool).nunique() == 2
    )
    if not has_binary_ground_truth:
        print("  → skipped: CRAS has no patch-level binary missing-ground-truth labels")
        print("     Use the controlled withheld-zone experiment for AUROC/AUPRC/F1.")
        for stale_path in (
            TABLES / "baseline_results_real.csv",
            TABLES / "ablation_results_real.csv",
        ):
            stale_path.unlink(missing_ok=True)
        return pd.DataFrame(), pd.DataFrame()

    # For baselines / ablation, D_obs and D_geo must exist in patch_scores
    missing_cols = [c for c in ["D_obs", "D_geo", "D_ang", "D_sem", "D_mat_missing", "D_mat_conflict"]
                    if c not in patch_scores.columns]
    if missing_cols:
        print(f"  [WARN] missing columns for baselines: {missing_cols}; skipping real-data baselines")
        return pd.DataFrame(), pd.DataFrame()

    baseline_df = score_baselines(patch_scores, seed=0)
    baseline_df["seed"] = 0
    abl_df = ablation_table(patch_scores)
    abl_df["seed"] = 0

    baseline_df.to_csv(TABLES / "baseline_results_real.csv", index=False)
    abl_df.to_csv(TABLES / "ablation_results_real.csv", index=False)
    print("  → Saved baseline_results_real.csv and ablation_results_real.csv")
    return baseline_df, abl_df


# ─── Step 9: Candidate views ──────────────────────────────────────────────────
def step9_views(patch_scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    print("\n[8/10] Generating candidate viewpoints …")
    view_config = config.get("view", {})
    candidates = generate_candidates(patch_scores, view_config)
    ranked = score_candidates(patch_scores, candidates, view_config)
    greedy = greedy_sequential_ranking(patch_scores, candidates, k=10)

    ranked.to_csv(TABLES / "candidate_view_ranking_real.csv", index=False)
    greedy.to_csv(TABLES / "candidate_view_greedy_real.csv", index=False)
    print(f"  → {len(candidates)} candidates generated, {len(ranked)} scored")
    if not ranked.empty:
        best = ranked.iloc[0]
        print(f"     Top-1 view: id={best['view_id']}  value={float(best['value']):.4f}  visible_gap_area={float(best['visible_gap_area']):.2f}m²")
    return ranked


# ─── Step 10: Closed loop ─────────────────────────────────────────────────────
def step10_closed_loop(
    patches_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    sem_df: pd.DataFrame,
    mat_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    print("\n[9/10] Running closed-loop simulation …")
    loop_config = {**config, "simulation": {"steps": 10, "recovery_per_view": 0.75}}
    scene = {
        "patches": patches_df,
        "observations": evidence_df,
        "semantics": sem_df,
        "materials": mat_df,
    }
    history_df, rankings_df, chosen_df = run_closed_loop_from_scene(scene, loop_config)
    history_df.to_csv(TABLES / "closed_loop_results_real.csv", index=False)
    chosen_df.to_csv(TABLES / "closed_loop_selected_views_real.csv", index=False)
    if len(history_df) > 0:
        final = history_df.iloc[-1]
        print(f"  → {len(history_df)} iterations completed")
        print(f"     Final remaining gap area: {final['remaining_gap_area']:.3f}")
        print(f"     Final recovery rate: {final.get('recovery_rate', 0):.1%}")
    return history_df


# ─── Visualization ────────────────────────────────────────────────────────────
def generate_all_figures(
    patch_scores: pd.DataFrame,
    comp_ranking: pd.DataFrame,
    ranked_views: pd.DataFrame,
    history_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    assoc_df: pd.DataFrame,
    elements_df: pd.DataFrame,
) -> None:
    print("\n[10/10] Generating figures …")

    # Static matplotlib figures
    plot_patch_bars(patch_scores, FIGURES / "patch_gap_map_real.png", column="G_gap")
    print("  ✓ patch_gap_map_real.png")

    for col in ["D_sem", "D_obs", "D_ang", "D_geo", "D_mat_missing", "D_mat_conflict"]:
        if col in patch_scores.columns:
            plot_patch_bars(patch_scores.head(60), FIGURES / f"{col}_map_real.png", column=col)
    print("  ✓ D_*_map_real.png")

    plot_component_ranking(comp_ranking.head(25), FIGURES / "component_gap_map_real.png")
    print("  ✓ component_gap_map_real.png")

    if len(history_df) > 0:
        plot_closed_loop(history_df, FIGURES / "closed_loop_curve_real.png")
        print("  ✓ closed_loop_curve_real.png")

    if not baseline_df.empty:
        plot_baseline_results(baseline_df, FIGURES / "baseline_comparison_real.png")
        print("  ✓ baseline_comparison_real.png")

    if not ablation_df.empty:
        plot_ablation_results(ablation_df, FIGURES / "ablation_results_real.png")
        print("  ✓ ablation_results_real.png")

    if not ranked_views.empty:
        plot_candidate_views(ranked_views.head(30), FIGURES / "candidate_views_real.png")
        print("  ✓ candidate_views_real.png")

    # Interactive Plotly HTML
    try:
        from patent_gap.visualization.interactive import build_interactive_report
        tri_map = np.load(str(PROCESSED / "triangle_element_map.npy"))
        build_interactive_report(
            patches_df=patch_scores,
            component_ranking=comp_ranking,
            ranked_views=ranked_views if not ranked_views.empty else pd.DataFrame(),
            history_df=history_df,
            baseline_df=baseline_df if not baseline_df.empty else pd.DataFrame(columns=["method", "AUROC", "AUPRC", "F1"]),
            ablation_df=ablation_df if not ablation_df.empty else pd.DataFrame(columns=["ablation", "AUROC", "AUPRC"]),
            mesh_npz_path=PROCESSED / "ifc_mesh.npz",
            tri_map=tri_map,
            elements_df=elements_df,
            assoc_df=assoc_df,
            output_path=REPORTS / "interactive_gap_report.html",
        )
        print("  ✓ interactive_gap_report.html")
    except Exception as exc:
        print(f"  [WARN] Interactive viz failed: {exc}")


def write_real_summary(
    patch_scores: pd.DataFrame,
    component_ranking: pd.DataFrame,
    ranked_views: pd.DataFrame,
    history_df: pd.DataFrame,
) -> None:
    semantic_valid = int(patch_scores["D_sem"].notna().sum())
    angular_valid = int(patch_scores["D_ang"].notna().sum())
    summary = {
        "source": "cras_real_association",
        "evaluation_status": "diagnostic_only_no_patch_level_ground_truth",
        "n_patches": int(len(patch_scores)),
        "n_elements": int(patch_scores["element_guid"].nunique()),
        "G_gap_mean": float(patch_scores["G_gap"].mean()),
        "G_gap_max": float(patch_scores["G_gap"].max()),
        "high_gap_patches_count": int((patch_scores["G_gap"] >= 0.55).sum()),
        "high_gap_fraction": float((patch_scores["G_gap"] >= 0.55).mean()),
        "D_obs_mean": float(patch_scores["D_obs"].mean()),
        "D_geo_mean": float(patch_scores["D_geo"].mean()),
        "D_sem_valid_count": semantic_valid,
        "D_sem_valid_fraction": float(semantic_valid / len(patch_scores)),
        "D_ang_valid_count": angular_valid,
        "D_ang_valid_fraction": float(angular_valid / len(patch_scores)),
        "material_missing_rate": float(patch_scores["D_mat_missing"].mean()),
        "candidate_views": int(len(ranked_views)),
        "top1_view_value": (
            float(ranked_views["value"].iloc[0]) if not ranked_views.empty else None
        ),
        "closed_loop_steps": int(len(history_df)),
        "closed_loop_final_net_recovery_rate": (
            float(history_df["recovery_rate"].iloc[-1]) if not history_df.empty else None
        ),
        "closed_loop_monotonic": (
            bool(history_df["remaining_gap_area"].is_monotonic_decreasing)
            if not history_df.empty
            else None
        ),
        "top3_components": [
            {
                "guid": str(row.element_guid),
                "class": str(row.ifc_class),
                "G_component": float(row.G_component),
            }
            for row in component_ranking.head(3).itertuples(index=False)
        ],
        "limitations": [
            "No patch-level binary missing-ground-truth labels are available for CRAS.",
            "The available 20k association sample has classification=0 for every point, so D_sem is unavailable.",
            "Scanner origins are unavailable in the cached association summaries, so D_ang is unavailable before supplemental scans.",
            "A supplemental scan can reveal angular deficiency; the diagnostic gap trajectory is therefore not guaranteed to be monotonic.",
        ],
    }
    path = REPORTS / "real_data_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  ✓ {path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="CRAS real-data gap analysis pipeline")
    parser.add_argument("--config", default=str(ROOT / "configs/experiment/cras_full.yaml"))
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()

    ensure_dirs()

    # Load config
    try:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        config = {}

    # If force, remove caches
    if args.force:
        for p in [PROCESSED / "patches_real.parquet"]:
            if p.exists():
                p.unlink()
                print(f"[force] removed {p}")

    t_start = time.time()
    print("=" * 65)
    print(" BIM-3D Vision Gap Analysis — Real CRAS Data Pipeline")
    print("=" * 65)

    # Load shared data
    print("\nLoading shared data …")
    elements_df = pd.read_parquet(PROCESSED / "ifc_elements.parquet")
    materials_df = pd.read_parquet(PROCESSED / "ifc_materials.parquet")
    assoc_df = pd.read_parquet(PROCESSED / "cras_point_sample_associations.parquet")
    print(f"  elements: {len(elements_df)}  materials: {len(materials_df)}  sample assoc pts: {len(assoc_df):,}")

    # Pipeline
    patches_df = step1_patches(config)
    evidence_df = step2_evidence(patches_df, elements_df, assoc_df, config)
    sem_df = step4_semantics(patches_df, assoc_df, config)
    mat_df = step5_materials(patches_df, elements_df, materials_df, assoc_df, config)
    patch_scores = step6_gap_scores(patches_df, evidence_df, sem_df, mat_df, config)
    comp_ranking = step7_components(patch_scores, config)
    baseline_df, ablation_df = step8_baselines(patch_scores)
    ranked_views = step9_views(patch_scores, config)
    history_df = step10_closed_loop(
        patches_df,
        evidence_df,
        sem_df,
        mat_df,
        config,
    )

    generate_all_figures(
        patch_scores, comp_ranking, ranked_views,
        history_df, baseline_df, ablation_df,
        assoc_df, elements_df,
    )
    write_real_summary(patch_scores, comp_ranking, ranked_views, history_df)

    total = time.time() - t_start
    print(f"\n{'='*65}")
    print(f" Pipeline complete in {total:.1f}s")
    print(f"{'='*65}")
    print(f" Patch scores  → {TABLES / 'patch_scores_real.csv'}")
    print(f" Components    → {TABLES / 'component_ranking_real.csv'}")
    print(f" Views         → {TABLES / 'candidate_view_ranking_real.csv'}")
    print(f" Closed loop   → {TABLES / 'closed_loop_results_real.csv'}")
    print(f" Interactive   → {REPORTS / 'interactive_gap_report.html'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
