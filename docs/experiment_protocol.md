# Experiment Protocol

## Data

Primary real dataset: CRAS Labs BIM Dataset.

- IFC: `data/raw/craslabbim.ifc`
- Point cloud ZIP: `data/raw/craslabannotated.zip`

The CRAS ZIP contains one fused ASC point cloud. It does not expose 21 independent scan stations in the ZIP manifest, so virtual scanning must be synthesized from BIM geometry or fused point cloud geometry.

EDF Challenge data requires user login and is not present under `data/raw/edf/`.

TS40K is optional and should not be fully downloaded until the CRAS smoke path is complete.

## Splits

Synthetic held-out tests split by seed:

- Train/tuning seeds: 0, 1, 2
- Validation seed: 10
- Test seeds: 100, 101, 102, 103, 104

The implementation does not tune on held-out seeds.

## Baselines

Implemented in the smoke path:

- Random view
- Geometry-only NBV
- Evidence + geometry
- Equal-weight patent score
- Tuned patent score

The framework has output slots for additional baselines and larger-scene oracle comparison once full IFC triangulation is available.

## Required Experiments Covered

- Synthetic cube regression test
- Six gap indicators
- Component ranking
- Candidate viewpoint generation and ranking
- Greedy closed-loop supplemental scanning for 5 steps
- Parameter search with a deterministic fallback when Optuna is unavailable
- Baseline comparison
- Ablation table
- CSV, PNG, HTML, Markdown, and JSON outputs

## Current CRAS Limit

CRAS files were downloaded and checksummed successfully. Full IFC triangulation and point-to-patch association require `ifcopenshell`; the current base Python does not provide it. The CRAS preprocess command records this limitation instead of fabricating mesh outputs.

