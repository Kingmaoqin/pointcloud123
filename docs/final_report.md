# Final Report

## Completed

- Created `/home/xqin5/patent_gap_nbv`.
- Downloaded CRAS IFC and point cloud ZIP.
- Verified MD5 checksums for both CRAS files.
- Installed missing geometry/tuning dependencies and triangulated CRAS IFC.
- Exported `data/processed/ifc_mesh.npz`, `ifc_elements.parquet`, `ifc_materials.parquet`, and triangle-to-element/GUID maps.
- Associated a 20000-point CRAS ASC smoke sample to IFC geometry after robust translation calibration; 10154 points matched within 5 cm.
- Implemented CPU-only synthetic cube pipeline.
- Implemented six gap indicators, component ranking, candidate view ranking, closed-loop update, baselines, ablations, and deterministic parameter search.
- Generated Slurm scripts without hard-coded partition names.
- Rechecked GPU access with elevated permission: 4 x NVIDIA A100 80GB PCIe are visible.
- Ran `pytest`: 10 tests passed.

## Key Held-out Synthetic Metrics

From `outputs/reports/summary_metrics.csv` over test seeds 100-104:

- AUROC: 1.0
- AUPRC: 1.0
- F1: 1.0
- IoU: 1.0
- NDCG@5: 1.0
- NDCG@10: 1.0
- Top-1 actual visible gap area: 4.0
- Top-3 cumulative visible gap area: 12.0

These are synthetic regression metrics, not CRAS full-scene metrics.

## Best Parameter Search Result

The best fallback search trial is saved in `outputs/reports/best_parameters.yaml`.

## CRAS Status

CRAS files are present and checksum-valid. The ZIP contains one fused ASC point cloud. IFC triangulation succeeded with 256 elements, 604187 vertices, 1197750 triangles, 24 cached materials, and 0 failed geometry elements. A 20000-point point-cloud smoke association matched 50.77% of sampled points within 5 cm after estimating a translation of approximately `[-0.682, 0, 0.667] m`.

## Important Outputs

- `outputs/reports/final_report.html`
- `outputs/reports/summary_metrics.csv`
- `outputs/reports/best_parameters.yaml`
- `outputs/reports/reproducibility_manifest.json`
- `outputs/tables/patch_scores.csv`
- `outputs/tables/component_ranking.csv`
- `outputs/tables/candidate_view_ranking.csv`
- `outputs/tables/baseline_results.csv`
- `outputs/tables/ablation_results.csv`
- `outputs/tables/closed_loop_results.csv`
- `outputs/figures/patch_gap_map.png`
- `outputs/figures/component_gap_map.png`
- `outputs/figures/closed_loop_curve.png`
- `outputs/figures/baseline_comparison.png`

## Limitations

- No Slurm runtime detected on the current node.
- GPU hardware is available, but the active base Python lacks `torch`; use the `MDPC` environment for CUDA-enabled PyTorch.
- Full CRAS point-cloud-to-patch association over the 584M-line ASC file is still a scaling step; smoke mode now validates the association path on 20000 points.
- The current full metrics are synthetic smoke/held-out regression results, not full CRAS point-cloud benchmark results.
