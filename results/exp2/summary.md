# E2 pilot 结果汇总

- config_hash `1ebf14ca78`, commit `7a91f13`, runs=36

| 方法 | awc恢复率 | dens_ok | crit_recall | 路径(m) | 站数 | 单位路径增益 |
|---|---|---|---|---|---|---|
| B0_random | 0.674 | 0.464 | 0.736 | 131 | 6.0 | 0.0058 |
| B1_patent | 0.741 | 0.344 | 0.228 | 142 | 6.0 | 0.0056 |
| B5_occ_rng | 0.919 | 0.529 | 0.833 | 215 | 6.0 | 0.0045 |
| B10_full | 0.938 | 0.539 | 0.882 | 156 | 6.0 | 0.0061 |

## 配对置换检验(Holm 校正)

| 对比 | 指标 | Δ均值 | p_raw | p_holm |
|---|---|---|---|---|
| B10_full vs B0_random | awc_gap_recovery | +0.264 | 0.0047 | 0.0423 |
| B10_full vs B0_random | ig_per_m | +0.000 | 0.8128 | 0.9516 |
| B10_full vs B0_random | crit_recall | +0.145 | 0.0973 | 0.4805 |
| B10_full vs B1_patent | awc_gap_recovery | +0.197 | 0.0047 | 0.0423 |
| B10_full vs B1_patent | ig_per_m | +0.000 | 0.4110 | 0.9516 |
| B10_full vs B1_patent | crit_recall | +0.654 | 0.0047 | 0.0423 |
| B10_full vs B5_occ_rng | awc_gap_recovery | +0.020 | 0.3172 | 0.9516 |
| B10_full vs B5_occ_rng | ig_per_m | +0.002 | 0.0047 | 0.0423 |
| B10_full vs B5_occ_rng | crit_recall | +0.049 | 0.0961 | 0.4805 |