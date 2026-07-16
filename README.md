# patent_gap_nbv

Reproducible reference implementation for the patent method:

> BIM-3D vision joint modeling for spatial information gap analysis and supplemental scan viewpoint evaluation.

The implementation includes the full CRAS IFC scene, controlled withheld-zone
evaluation, raycast visibility diagnostics, candidate supplemental-scan ranking,
closed-loop updates, and a local Gradio/Plotly workstation.

## Local Web Workstation

```bash
cd /home/xqin5/patent_gap_nbv
python scripts/run_web_app.py --host 127.0.0.1 --port 7862
```

Open `http://127.0.0.1:7862`. The workstation provides:

- CRAS real-association, raycast-diagnostic, and controlled-evaluation sources.
- Interactive 3D gap maps and candidate supplemental-scan viewpoints.
- Single-step and multi-step rescan simulation with before/after comparison.
- Import of `.ifc`, `.npz`, `.obj`, `.ply`, `.stl`, and `.glb` models.
- Optional patch or element score import from CSV.

Uploads are restricted by extension, size, geometry count, finite coordinates,
and face-index bounds. NumPy inputs are loaded with `allow_pickle=False`.

## Quick Start

```bash
cd /home/xqin5/patent_gap_nbv
python -m pytest -q
ruff check src tests scripts/run_web_app.py scripts/run_synthetic_pipeline.py scripts/run_real_pipeline.py --ignore E402
python scripts/gen_synthetic_scan.py
python scripts/run_synthetic_pipeline.py
python scripts/run_real_pipeline.py
```

## Evaluation Scope

- `controlled_withheld_zone`: the formal quantitative benchmark. The withheld
  region is declared independently of the scored evidence.
- `raycast`: a visibility and sensor-coverage diagnostic, not an independent
  ground-truth benchmark.
- `real`: the full CRAS association diagnostic. CRAS currently has no
  patch-level binary missing labels, so AUROC/AUPRC/F1 are intentionally not
  reported for this source.

## CRAS Commands

```bash
bash scripts/download_cras.sh
python -m patent_gap.cli inspect-data --config configs/data/cras.yaml
python scripts/preprocess_cras.py --config configs/experiment/cras_smoke.yaml
python scripts/run_cras_full_association.py max_points=1000000
```

## Supported CLI

```bash
python -m patent_gap.cli inspect-data --config configs/data/cras.yaml
python -m patent_gap.cli preprocess --config configs/experiment/cras_smoke.yaml
python -m patent_gap.cli associate-cras --config configs/experiment/cras_full.yaml
python -m patent_gap.cli corrupt --config configs/corruption/mixed.yaml seed=0
python -m patent_gap.cli compute-gap --config configs/experiment/synthetic_smoke.yaml
python -m patent_gap.cli rank-views --config configs/experiment/synthetic_smoke.yaml
python -m patent_gap.cli closed-loop --config configs/experiment/synthetic_smoke.yaml simulation.steps=10
python -m patent_gap.cli tune --config configs/tuning/optuna.yaml
python -m patent_gap.cli evaluate --config configs/experiment/heldout_test.yaml
python -m patent_gap.cli report --input outputs --output outputs/reports/final_report.html
```

## Outputs

Primary outputs are written to:

- `outputs/tables/patch_scores_synth.csv`
- `outputs/tables/patch_scores_real.csv`
- `outputs/tables/candidate_view_ranking_synth.csv`
- `outputs/tables/candidate_view_ranking_real.csv`
- `outputs/tables/closed_loop_results_synth.csv`
- `outputs/tables/closed_loop_results_real.csv`
- `outputs/figures/`
- `outputs/reports/synthetic_summary.json`
- `outputs/reports/real_data_summary.json`
- `outputs/reports/interactive_gap_report.html`
