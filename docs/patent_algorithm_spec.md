# Patent Algorithm Specification

This implementation follows the local patent file `pczhuanli/专利 (1).pdf`.

## Analysis Object

The primary unit is a BIM surface patch. Each patch stores:

- BIM element identifier and IFC class
- patch centroid, normal, area, and component importance
- semantic distributions from BIM and observation
- material presence and material conflict state
- observation evidence statistics
- angle and geometric coverage statistics

## Gap Indicators

The code computes six interpretable terms:

- `D_sem`: Jensen-Shannon divergence between BIM and observed semantic distributions, normalized by `log(2)`.
- `D_mat_missing`: material attribute missing indicator.
- `D_mat_conflict`: IFC material versus visual material conflict indicator.
- `D_obs`: direct measurement, valid view count, resolution, and traceability deficit.
- `D_ang`: best frontality, valid view count, and angular diversity deficit.
- `D_geo`: surface sample coverage and density ratio deficit.

When a term is not available, the weighted sum renormalizes over available terms rather than treating missing values as zero or one.

## Patch Score

Two forms are represented:

- Patent literal weighted patch gap: `G_gap = sum(alpha_i * D_i)`.
- Decoupled task priority: `G_task = G_gap * (1 + lambda_importance * W_importance)`.

Weights are non-negative and normalized before scoring.

## Component Ranking

Each component aggregates:

- mean gap
- max gap
- p90 gap
- high-gap area ratio
- number of high-gap patches
- engineering importance

The output is `outputs/tables/component_ranking.csv`.

## View Ranking

Candidate viewpoints are generated around high-gap patches. For each candidate:

`V(v) = sum(G_gap(x) * Q(x, v) * A(x)) - eta * R(v)`

where `Q` includes visibility, distance quality, frontality, and projected resolution quality. The project outputs both independent ranking and greedy sequential ranking.

## Synthetic Regression Scene

The synthetic cube contains six surface patches:

- occluded surface
- semantic conflict surface
- material missing surface
- grazing-angle surface
- low-density surface
- nominal surface

This is the required regression test before scaling to CRAS.

