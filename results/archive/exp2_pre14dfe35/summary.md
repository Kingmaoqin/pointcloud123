# E2 pilot 结果汇总

- config_hash `b0263d9f95`, commit `65b9bbb-dirty`, runs=120

| 方法 | awc恢复率 | dens_ok | crit_recall | 路径(m) | 站数 | 单位路径增益 | 总耗时(s) | awc/1000s |
|---|---|---|---|---|---|---|---|---|
| B0_random | 0.536 | 0.474 | 0.627 | 137 | 6.0 | 0.0042 | 1355 | 0.3960 |
| Bdisp_maxmin | 0.915 | 0.532 | 0.865 | 239 | 6.0 | 0.0039 | 1559 | 0.5893 |
| B1_patent | 0.758 | 0.354 | 0.249 | 132 | 6.0 | 0.0062 | 1345 | 0.5671 |
| B5_occ_rng | 0.925 | 0.526 | 0.877 | 240 | 6.0 | 0.0041 | 1559 | 0.5993 |
| B10_full | 0.956 | 0.539 | 0.907 | 227 | 6.0 | 0.0045 | 1534 | 0.6273 |

## 配对置换检验(Holm 校正)

| 对比 | 指标 | Δ均值 | p_raw | p_holm |
|---|---|---|---|---|
| B10_full vs B0_random | awc_gap_recovery | +0.420 | 0.0001 | 0.0016 |
| B10_full vs B0_random | ig_per_m | +0.000 | 0.6108 | 0.6108 |
| B10_full vs B0_random | crit_recall | +0.281 | 0.0001 | 0.0016 |
| B10_full vs B0_random | awc_per_1000s | +0.231 | 0.0002 | 0.0020 |
| B10_full vs Bdisp_maxmin | awc_gap_recovery | +0.041 | 0.0153 | 0.0564 |
| B10_full vs Bdisp_maxmin | ig_per_m | +0.001 | 0.0023 | 0.0147 |
| B10_full vs Bdisp_maxmin | crit_recall | +0.042 | 0.0264 | 0.0564 |
| B10_full vs Bdisp_maxmin | awc_per_1000s | +0.038 | 0.0009 | 0.0081 |
| B10_full vs B1_patent | awc_gap_recovery | +0.198 | 0.0001 | 0.0016 |
| B10_full vs B1_patent | ig_per_m | -0.002 | 0.0001 | 0.0016 |
| B10_full vs B1_patent | crit_recall | +0.658 | 0.0001 | 0.0016 |
| B10_full vs B1_patent | awc_per_1000s | +0.060 | 0.0021 | 0.0147 |
| B10_full vs B5_occ_rng | awc_gap_recovery | +0.030 | 0.0001 | 0.0016 |
| B10_full vs B5_occ_rng | ig_per_m | +0.000 | 0.0141 | 0.0564 |
| B10_full vs B5_occ_rng | crit_recall | +0.031 | 0.0060 | 0.0300 |
| B10_full vs B5_occ_rng | awc_per_1000s | +0.028 | 0.0018 | 0.0144 |