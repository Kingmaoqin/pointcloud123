from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from patent_gap.corruption.synthetic import apply_synthetic_corruption
from patent_gap.data.audit import write_data_audit
from patent_gap.data.pointcloud import associate_cras_stream, preprocess_cras_point_sample
from patent_gap.evaluation.baselines import ablation_table, score_baselines
from patent_gap.evaluation.metrics import component_metrics, patch_detection_metrics
from patent_gap.gap.scoring import compute_patch_scores, rank_components
from patent_gap.ifc.reader import triangulate_ifc
from patent_gap.simulation.closed_loop import run_closed_loop
from patent_gap.simulation.synthetic_cube import synthetic_scene
from patent_gap.utils.config import apply_overrides, ensure_output_dirs, load_yaml
from patent_gap.utils.repro import save_manifest, set_seed
from patent_gap.viewpoints.ranking import generate_candidates, greedy_sequential_ranking, score_candidates
from patent_gap.visualization.plots import (
    plot_ablation_results,
    plot_baseline_results,
    plot_candidate_views,
    plot_closed_loop,
    plot_component_ranking,
    plot_parameter_importance,
    plot_patch_bars,
)


def _save_core_outputs(config: dict[str, Any], scene: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dirs = ensure_output_dirs(config.get("outputs_dir", "outputs"))
    patch_scores = compute_patch_scores(scene, config.get("gap", {}))
    components = rank_components(patch_scores, config.get("gap", {}))
    candidates = generate_candidates(patch_scores, config.get("view", {}))
    ranked_views = score_candidates(patch_scores, candidates, config.get("view", {}))
    patch_scores.to_csv(dirs["tables"] / "patch_scores.csv", index=False)
    components.to_csv(dirs["tables"] / "component_ranking.csv", index=False)
    ranked_views.to_csv(dirs["tables"] / "candidate_view_ranking.csv", index=False)
    greedy_sequential_ranking(patch_scores, candidates, k=5).to_csv(dirs["tables"] / "candidate_view_greedy.csv", index=False)
    plot_patch_bars(patch_scores, dirs["figures"] / "patch_gap_map.png", "G_gap")
    plot_patch_bars(patch_scores, dirs["figures"] / "D_sem_map.png", "D_sem")
    plot_patch_bars(patch_scores, dirs["figures"] / "D_mat_missing_map.png", "D_mat_missing")
    plot_patch_bars(patch_scores, dirs["figures"] / "D_mat_conflict_map.png", "D_mat_conflict")
    plot_patch_bars(patch_scores, dirs["figures"] / "D_obs_map.png", "D_obs")
    plot_patch_bars(patch_scores, dirs["figures"] / "D_ang_map.png", "D_ang")
    plot_patch_bars(patch_scores, dirs["figures"] / "D_geo_map.png", "D_geo")
    plot_component_ranking(components, dirs["figures"] / "component_gap_map.png")
    plot_candidate_views(ranked_views, dirs["figures"] / "candidate_views.png")
    return patch_scores, components, ranked_views


def command_inspect_data(args: argparse.Namespace, overrides: list[str]) -> None:
    config = apply_overrides(load_yaml(args.config), overrides)
    audit = write_data_audit(config, "docs/data_audit.md")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def command_preprocess(args: argparse.Namespace, overrides: list[str]) -> None:
    config = apply_overrides(load_yaml(args.config), overrides)
    dirs = ensure_output_dirs(config.get("outputs_dir", "outputs"))
    result = {}
    if config.get("scene") == "cras":
        data_config = load_yaml(config.get("data_config", "configs/data/cras.yaml"))
        result = triangulate_ifc(data_config.get("ifc_path", "data/raw/craslabbim.ifc"), "data/processed")
        point_config = dict(data_config)
        point_config.update(config.get("preprocess", {}))
        if result.get("triangulated") and Path(point_config.get("pointcloud_zip", "")).exists():
            result["point_association"] = preprocess_cras_point_sample(point_config, "data/processed")
        message = f"CRAS preprocess complete: triangulated={result.get('triangulated')} cached={result.get('cached', False)}"
    else:
        scene = synthetic_scene(int(config.get("seed", 0)))
        patch_scores, components, ranked_views = _save_core_outputs(config, scene)
        message = f"preprocess complete: {len(patch_scores)} patches, {len(components)} components, {len(ranked_views)} candidate views"
    (dirs["reports"] / "preprocess_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(message)


def command_associate_cras(args: argparse.Namespace, overrides: list[str]) -> None:
    config = apply_overrides(load_yaml(args.config), overrides)
    dirs = ensure_output_dirs(config.get("outputs_dir", "outputs"))
    data_config = load_yaml(config.get("data_config", "configs/data/cras.yaml"))
    result = triangulate_ifc(data_config.get("ifc_path", "data/raw/craslabbim.ifc"), "data/processed")
    if not result.get("triangulated"):
        raise RuntimeError(f"IFC triangulation unavailable: {result}")
    assoc_config = dict(data_config)
    assoc_config.update(config.get("association", config.get("preprocess", {})))
    run_name = str(assoc_config.get("run_name", config.get("experiment", "cras_full_assoc")))
    summary = associate_cras_stream(assoc_config, "data/processed", run_name=run_name)
    (dirs["reports"] / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def command_corrupt(args: argparse.Namespace, overrides: list[str]) -> None:
    config = apply_overrides(load_yaml(args.config), overrides)
    scene = apply_synthetic_corruption(synthetic_scene(int(config.get("seed", 0))), config)
    patch_scores, _, _ = _save_core_outputs({"outputs_dir": "outputs", "gap": {}, "view": {}}, scene)
    print(patch_scores[["patch_id", "D_sem", "D_mat_missing", "D_mat_conflict", "D_obs", "D_ang", "D_geo", "G_gap"]].to_string(index=False))


def command_compute_gap(args: argparse.Namespace, overrides: list[str]) -> None:
    config = apply_overrides(load_yaml(args.config), overrides)
    set_seed(int(config.get("seed", 0)))
    scene = synthetic_scene(int(config.get("seed", 0)))
    patch_scores, components, _ = _save_core_outputs(config, scene)
    metrics = patch_detection_metrics(patch_scores)
    component_out = component_metrics(components, patch_scores)
    dirs = ensure_output_dirs(config.get("outputs_dir", "outputs"))
    (dirs["reports"] / "gap_metrics.json").write_text(json.dumps({"patch": metrics, "component": component_out}, indent=2), encoding="utf-8")
    print(json.dumps({"patch": metrics, "component": component_out}, indent=2))


def command_rank_views(args: argparse.Namespace, overrides: list[str]) -> None:
    config = apply_overrides(load_yaml(args.config), overrides)
    scene = synthetic_scene(int(config.get("seed", 0)))
    _, _, ranked_views = _save_core_outputs(config, scene)
    print(ranked_views.head(10).to_string(index=False))


def command_closed_loop(args: argparse.Namespace, overrides: list[str]) -> None:
    config = apply_overrides(load_yaml(args.config), overrides)
    dirs = ensure_output_dirs(config.get("outputs_dir", "outputs"))
    history, rankings, chosen = run_closed_loop(config)
    history.to_csv(dirs["tables"] / "closed_loop_results.csv", index=False)
    rankings.to_csv(dirs["tables"] / "closed_loop_candidate_rankings.csv", index=False)
    chosen.to_csv(dirs["tables"] / "closed_loop_selected_views.csv", index=False)
    plot_closed_loop(history, dirs["figures"] / "closed_loop_curve.png")
    save_manifest(dirs["reports"] / "reproducibility_manifest.json", config, int(config.get("seed", 0)))
    print(history.to_string(index=False))


def command_tune(args: argparse.Namespace, overrides: list[str]) -> None:
    config = apply_overrides(load_yaml(args.config), overrides)
    dirs = ensure_output_dirs("outputs")
    trials = int(config.get("trials", 50))
    rng = set_seed(int(config.get("seed", 0)))
    search_space = config.get("search_space", {})
    param_names = ["alpha_sem", "alpha_mat_missing", "alpha_mat_conflict", "alpha_obs", "alpha_ang", "alpha_geo"]

    def evaluate_gap(gap: dict[str, float], trial_number: int) -> dict[str, float]:
        scene = synthetic_scene(int(config.get("seed", 0)) + trial_number)
        patch_scores = compute_patch_scores(scene, gap)
        metrics = patch_detection_metrics(patch_scores)
        objective = 0.4 * metrics["AUPRC"] + 0.3 * metrics["Recall@Top-10%"] - 0.1 * metrics["ECE"]
        return {"objective": objective, **metrics}

    rows = []
    try:
        import optuna

        storage = f"sqlite:///{dirs['optuna'] / 'synthetic_optuna.db'}"
        sampler = optuna.samplers.TPESampler(seed=int(config.get("seed", 0)))
        study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage, study_name="synthetic_gap", load_if_exists=True)

        def objective(trial: Any) -> float:
            raw_values = []
            for name in param_names:
                lo, hi = search_space.get(name, [0.01, 1.0])
                raw_values.append(trial.suggest_float(name, float(lo), float(hi), log=True))
            raw = pd.Series(raw_values, index=param_names, dtype=float)
            raw = raw / raw.sum()
            lo, hi = search_space.get("high_gap_threshold", [0.50, 0.85])
            gap = {name: float(raw[name]) for name in param_names}
            gap["high_gap_threshold"] = trial.suggest_float("high_gap_threshold", float(lo), float(hi))
            metrics = evaluate_gap(gap, trial.number)
            for key, value in {**gap, **metrics}.items():
                trial.set_user_attr(key, float(value))
            return float(metrics["objective"])

        existing = len(study.trials)
        if existing < trials:
            study.optimize(objective, n_trials=trials - existing, timeout=config.get("timeout_s"))
        for trial in study.trials:
            row = {"trial": trial.number, "objective": trial.value if trial.value is not None else float("nan")}
            row.update(trial.params)
            row.update(trial.user_attrs)
            rows.append(row)
        best = {"trial": study.best_trial.number, "objective": study.best_value, **study.best_trial.params, **study.best_trial.user_attrs}
    except Exception as exc:
        best = None
        for trial in range(trials):
            raw = rng.uniform(0.01, 1.0, size=6)
            alphas = raw / raw.sum()
            gap = {
                "alpha_sem": float(alphas[0]),
                "alpha_mat_missing": float(alphas[1]),
                "alpha_mat_conflict": float(alphas[2]),
                "alpha_obs": float(alphas[3]),
                "alpha_ang": float(alphas[4]),
                "alpha_geo": float(alphas[5]),
                "high_gap_threshold": float(rng.uniform(0.50, 0.85)),
            }
            metrics = evaluate_gap(gap, trial)
            row = {"trial": trial, "fallback_reason": str(exc), **gap, **metrics}
            rows.append(row)
            if best is None or metrics["objective"] > best["objective"]:
                best = row
    df = pd.DataFrame(rows)
    df.to_csv(dirs["optuna"] / "synthetic_trials.csv", index=False)
    plot_parameter_importance(df, dirs["figures"] / "parameter_importance.png")
    best = best or {}
    (dirs["reports"] / "best_parameters.yaml").write_text("\n".join(f"{k}: {v}" for k, v in best.items()), encoding="utf-8")
    print(json.dumps(best, indent=2))


def command_evaluate(args: argparse.Namespace, overrides: list[str]) -> None:
    config = apply_overrides(load_yaml(args.config), overrides)
    dirs = ensure_output_dirs(config.get("outputs_dir", "outputs"))
    seeds = config.get("split", {}).get("test_seeds", [int(config.get("seed", 0))])
    metric_rows = []
    baseline_rows = []
    ablation_rows = []
    for seed in seeds:
        scene = synthetic_scene(int(seed))
        patch_scores, components, ranked_views = _save_core_outputs(config | {"seed": seed}, scene)
        metrics = patch_detection_metrics(patch_scores)
        metrics["seed"] = seed
        metrics.update(component_metrics(components, patch_scores))
        metrics["Top-1 actual gain"] = float(ranked_views["visible_gap_area"].iloc[0]) if not ranked_views.empty else 0.0
        metrics["Top-3 cumulative gain"] = float(ranked_views["visible_gap_area"].head(3).sum()) if not ranked_views.empty else 0.0
        metric_rows.append(metrics)
        b = score_baselines(patch_scores, int(seed))
        b["seed"] = seed
        baseline_rows.append(b)
        a = ablation_table(patch_scores)
        a["seed"] = seed
        ablation_rows.append(a)
    summary = pd.DataFrame(metric_rows)
    baselines = pd.concat(baseline_rows, ignore_index=True)
    ablations = pd.concat(ablation_rows, ignore_index=True)
    summary.to_csv(dirs["reports"] / "summary_metrics.csv", index=False)
    baselines.to_csv(dirs["tables"] / "baseline_results.csv", index=False)
    ablations.to_csv(dirs["tables"] / "ablation_results.csv", index=False)
    plot_baseline_results(baselines.groupby("method", as_index=False)["AUPRC"].mean(), dirs["figures"] / "baseline_comparison.png")
    plot_ablation_results(ablations, dirs["figures"] / "ablation_results.png")
    print(summary.mean(numeric_only=True).to_string())


def command_report(args: argparse.Namespace, overrides: list[str]) -> None:
    del overrides
    root = Path(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = root / "reports" / "summary_metrics.csv"
    summary_html = ""
    if summary_path.exists():
        summary_html = pd.read_csv(summary_path).describe().to_html()
    tables = sorted(str(p.relative_to(root)) for p in (root / "tables").glob("*.csv")) if (root / "tables").exists() else []
    figures = sorted(str(p.relative_to(root)) for p in (root / "figures").glob("*.png")) if (root / "figures").exists() else []
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Patent Gap NBV Report</title></head>
<body>
<h1>Patent Gap NBV Final Report</h1>
<p>This report is generated from real files produced by the local pipeline.</p>
<h2>Summary Metrics</h2>{summary_html}
<h2>Tables</h2><ul>{''.join(f'<li>{t}</li>' for t in tables)}</ul>
<h2>Figures</h2><ul>{''.join(f'<li>{f}</li>' for f in figures)}</ul>
</body></html>"""
    output.write_text(html, encoding="utf-8")
    md = output.with_suffix(".md")
    md.write_text("# Final Report\n\nGenerated outputs:\n\n" + "\n".join(f"- {t}" for t in tables + figures) + "\n", encoding="utf-8")
    print(f"report written to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patent_gap")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["inspect-data", "preprocess", "associate-cras", "corrupt", "compute-gap", "rank-views", "closed-loop", "tune", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
    p = sub.add_parser("report")
    p.add_argument("--input", default="outputs")
    p.add_argument("--output", default="outputs/reports/final_report.html")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)
    commands = {
        "inspect-data": command_inspect_data,
        "preprocess": command_preprocess,
        "associate-cras": command_associate_cras,
        "corrupt": command_corrupt,
        "compute-gap": command_compute_gap,
        "rank-views": command_rank_views,
        "closed-loop": command_closed_loop,
        "tune": command_tune,
        "evaluate": command_evaluate,
        "report": command_report,
    }
    commands[args.command](args, overrides)


if __name__ == "__main__":
    main()
