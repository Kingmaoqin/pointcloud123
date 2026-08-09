# E2 pilot 结果汇总

- config_hash `54f6a8f7d8`, commit `30ed463-dirty`, runs=144, 失败 0
- 终止原因分布: {'rounds': 144}

| 方法 | awc恢复率 | 构件等权 | dens_ok | vs完整普查 | crit_recall | 路径(m) | 站数 | 单位路径增益 | awc/1000s |
|---|---|---|---|---|---|---|---|---|---|
| B0_random | 0.470 | 0.620 | 0.509 | 1.015 | 0.621 | 138 | 6.0 | 0.0037 | 0.3447 |
| Bdisp_maxmin | 0.924 | 0.905 | 0.554 | 1.107 | 0.885 | 244 | 6.0 | 0.0039 | 0.5916 |
| Bbim_offline | 0.896 | 0.873 | 0.557 | 1.106 | 0.871 | 230 | 6.0 | 0.0040 | 0.5824 |
| B1_patent | 0.779 | 0.322 | 0.409 | 0.814 | 0.311 | 132 | 6.0 | 0.0068 | 0.5845 |
| B5_occ_rng | 0.927 | 0.906 | 0.563 | 1.117 | 0.920 | 249 | 6.0 | 0.0039 | 0.5931 |
| B10_full | 0.970 | 0.929 | 0.574 | 1.147 | 0.931 | 156 | 6.0 | 0.0064 | 0.6979 |

## 配对置换检验(Holm 校正)

| 对比 | 指标 | Δ均值 | p_raw | p_holm |
|---|---|---|---|---|
| B10_full vs B0_random | awc_gap_recovery | +0.499 | 0.0001 | 0.0010 |
| B10_full vs B0_random | asset_recovery | +0.309 | 0.0001 | 0.0010 |
| B10_full vs Bdisp_maxmin | awc_gap_recovery | +0.046 | 0.0012 | 0.0060 |
| B10_full vs Bdisp_maxmin | asset_recovery | +0.024 | 0.1283 | 0.1299 |
| B10_full vs Bbim_offline | awc_gap_recovery | +0.074 | 0.0022 | 0.0088 |
| B10_full vs Bbim_offline | asset_recovery | +0.056 | 0.0522 | 0.1299 |
| B10_full vs B1_patent | awc_gap_recovery | +0.191 | 0.0001 | 0.0010 |
| B10_full vs B1_patent | asset_recovery | +0.607 | 0.0001 | 0.0010 |
| B10_full vs B5_occ_rng | awc_gap_recovery | +0.043 | 0.0001 | 0.0010 |
| B10_full vs B5_occ_rng | asset_recovery | +0.023 | 0.0433 | 0.1299 |