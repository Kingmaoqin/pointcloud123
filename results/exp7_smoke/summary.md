# E7 规划环境先验强度消融

config_hash `0384413e6b` · commit `960ca9a-dirty` · 1 场景 × 1 种子 × 3 先验档 × 2 方法


## B10_full

| 指标 | LEGACY | P100 | P0 |
|---|---|---|---|
| awc_gap_recovery ↑ | 0.635 | 0.635 | 0.599 |
| asset_recovery ↑ | 0.509 | 0.509 | 0.543 |
| crit_recall ↑ | 0.440 | 0.440 | 0.486 |
| dens_ok ↑ | 0.338 | 0.338 | 0.345 |
| dens_ok_vs_full ↑ | 0.659 | 0.659 | 0.673 |
| ig_per_m ↑ | 0.006 | 0.008 | 0.008 |
| path_len_m ↓ | 106.738 | 83.295 | 78.767 |
| n_stations ↓ | 2.000 | 2.000 | 2.000 |
| vis_mae ↓ | 0.031 | 0.031 | 0.056 |
| vis_over ↓ | 0.031 | 0.031 | 0.048 |
| invalid_view_frac ↓ | 0.022 | 0.022 | 0.031 |
| mplan_known_ratio ↑ | 1.000 | 0.972 | 0.971 |
| n | 1 | 1 | 1 |

## B11_disc

| 指标 | LEGACY | P100 | P0 |
|---|---|---|---|
| awc_gap_recovery ↑ | 0.635 | 0.635 | 0.598 |
| asset_recovery ↑ | 0.509 | 0.509 | 0.543 |
| crit_recall ↑ | 0.440 | 0.440 | 0.486 |
| dens_ok ↑ | 0.338 | 0.338 | 0.343 |
| dens_ok_vs_full ↑ | 0.659 | 0.659 | 0.668 |
| ig_per_m ↑ | 0.006 | 0.008 | 0.008 |
| path_len_m ↓ | 106.738 | 83.295 | 78.517 |
| n_stations ↓ | 2.000 | 2.000 | 2.000 |
| vis_mae ↓ | 0.005 | 0.005 | 0.015 |
| vis_over ↓ | 0.001 | 0.001 | 0.002 |
| invalid_view_frac ↓ | 0.000 | 0.000 | 0.000 |
| mplan_known_ratio ↑ | 1.000 | 0.972 | 0.971 |
| n | 1 | 1 | 1 |

## 先验减弱的代价（各档 vs LEGACY，配对置换检验 + Holm）

| 方法 | 档 | 指标 | Δ(档−LEGACY) | 95%CI | p_holm |
|---|---|---|---|---|---|

## 在线遮挡发现的价值 vs 先验强度（B11_disc − B10_full）

| 档 | 指标 | Δ(B11−B10) | 95%CI | p_holm |
|---|---|---|---|---|

## 终止原因

| 方法 | 档 | rounds | stations | length | no_candidate |
|---|---|---|---|---|---|
| B10_full | LEGACY | 1 | 0 | 0 | 0 |
| B10_full | P100 | 1 | 0 | 0 | 0 |
| B10_full | P0 | 1 | 0 | 0 | 0 |
| B11_disc | LEGACY | 1 | 0 | 0 | 0 |
| B11_disc | P100 | 1 | 0 | 0 | 0 |
| B11_disc | P0 | 1 | 0 | 0 | 0 |
