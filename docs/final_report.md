# Final Report

## 2026-06-20 Audit Update

The current implementation uses 9,438 IFC surface patches and separates three
different claims:

1. Controlled withheld-zone evaluation supplies independent binary ground truth.
2. Raycast output diagnoses visibility and scanner coverage.
3. CRAS real association supplies a full-scene engineering diagnostic but no
   patch-level binary missing labels.

The controlled benchmark has 2,298 positive patches out of 9,438 (24.35%
prevalence). Results are AUROC 0.9958, AUPRC 0.9866, AUPRC lift 4.052,
fixed-threshold F1 0.8939, balanced accuracy 0.9059, and MCC 0.8710. The
test-set oracle F1 of 0.9616 at threshold 0.3708 is retained only as a
diagnostic and is not used as the primary F1.

The ten-step controlled supplemental-scan experiment reduces weighted target gap
area from 450.51 to 175.92 (60.95% net recovery), with a non-increasing result at
every step.

The real CRAS run scores all 9,438 patches and ranks 810 candidate views. It is
marked diagnostic-only: the association sample contains only classification code
0, scanner origins are absent from cached summaries, and patch-level binary
ground truth is unavailable. Supervised real-data metrics are therefore skipped.

The local workstation is available with:

```bash
python scripts/run_web_app.py --host 127.0.0.1 --port 7862
```

It supports interactive 3D inspection, model and score import, candidate scan
views, before/after comparison, and closed-loop rescan simulation.

## Completed

- Created `/home/xqin5/patent_gap_nbv`.
- Downloaded CRAS IFC and point cloud ZIP.
- Verified MD5 checksums for both CRAS files.
- Installed missing geometry/tuning dependencies and triangulated CRAS IFC.
- Exported `data/processed/ifc_mesh.npz`, `ifc_elements.parquet`, `ifc_materials.parquet`, and triangle-to-element/GUID maps.
- Associated a 20000-point CRAS ASC smoke sample to IFC geometry after robust translation calibration; 10154 points matched within 5 cm.
- Ran full CRAS ASC closest-surface association over 584701977 valid points in 585 chunks.
- Implemented CPU-only synthetic cube pipeline.
- Implemented six gap indicators, component ranking, candidate view ranking, closed-loop update, baselines, ablations, and Optuna TPE parameter search.
- Generated Slurm scripts without hard-coded partition names.
- Rechecked GPU access with elevated permission: 4 x NVIDIA A100 80GB PCIe are visible.
- Ran `pytest`: 22 tests passed.

## Legacy Synthetic Regression Metrics

From `outputs/reports/summary_metrics.csv` over test seeds 100-104:

- AUROC: 1.0
- AUPRC: 1.0
- F1: 1.0
- IoU: 1.0
- NDCG@5: 1.0
- NDCG@10: 1.0
- Top-1 actual visible gap area: 4.0
- Top-3 cumulative visible gap area: 12.0

These are legacy six-patch regression metrics, not the current controlled
withheld-zone evaluation and not CRAS full-scene metrics.

## Best Parameter Search Result

The best Optuna TPE search trial is saved in `outputs/reports/best_parameters.yaml`.

## CRAS Status

CRAS files are present and checksum-valid. The ZIP contains one fused ASC point cloud. IFC triangulation succeeded with 256 elements, 604187 vertices, 1197750 triangles, 24 cached materials, and 0 failed geometry elements. A 20000-point point-cloud smoke association matched 50.77% of sampled points within 5 cm after estimating a translation of approximately `[-0.682, 0, 0.667] m`.

Full CRAS association completed with `open3d.compute_closest_points`:

- Valid points: 584701977
- Matched within 5 cm: 12835294
- Unmatched: 571866683
- Matched ratio: 0.02195185668065562
- Runtime: 2491.1 s
- Throughput: 234720.6 points/s
- Summary: `data/processed/cras_full_assoc/summary.json`

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
- Full CRAS point-cloud-to-element association is complete. Patch-level CRAS scoring remains a next refinement: the current full run aggregates to IFC element IDs/classes, while synthetic tests cover patch-level gap scoring and view ranking.
- The current full metrics are synthetic smoke/held-out regression results, not full CRAS point-cloud benchmark results.
