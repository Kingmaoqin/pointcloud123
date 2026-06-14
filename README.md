# patent_gap_nbv

Reproducible reference implementation for the patent method:

> BIM-3D vision joint modeling for spatial information gap analysis and supplemental scan viewpoint evaluation.

The project intentionally starts with interpretable geometry and a synthetic cube regression scene. CRAS data support is included through download, audit, and optional IFC triangulation hooks. If `ifcopenshell` is unavailable, IFC audit falls back to STEP text inspection and records the limitation instead of inventing geometry.

## Quick Start

```bash
cd /home/xqin5/patent_gap_nbv
python -m pytest -q
python scripts/run_smoke_test.py
python -m patent_gap.cli closed-loop --config configs/experiment/synthetic_smoke.yaml simulation.steps=5
python -m patent_gap.cli evaluate --config configs/experiment/heldout_test.yaml
python -m patent_gap.cli report --input outputs --output outputs/reports/final_report.html
```

## CRAS Commands

```bash
bash scripts/download_cras.sh
python -m patent_gap.cli inspect-data --config configs/data/cras.yaml
python scripts/preprocess_cras.py --config configs/experiment/cras_smoke.yaml
```

## Supported CLI

```bash
python -m patent_gap.cli inspect-data --config configs/data/cras.yaml
python -m patent_gap.cli preprocess --config configs/experiment/cras_smoke.yaml
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

- `outputs/tables/patch_scores.csv`
- `outputs/tables/component_ranking.csv`
- `outputs/tables/candidate_view_ranking.csv`
- `outputs/tables/baseline_results.csv`
- `outputs/tables/ablation_results.csv`
- `outputs/tables/closed_loop_results.csv`
- `outputs/figures/`
- `outputs/reports/final_report.html`
- `outputs/reports/reproducibility_manifest.json`

