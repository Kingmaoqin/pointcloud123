# E5: 跨规模/密度稳定性

- config_hash `23e6699a18`, commit `30ed463-dirty`, runs=108
- 预算按每 9.3 个 BIM 构件给 1 站。固定 6 站时 L/high 只拿到完整扫描的 17%、S/low 拿到 46%, 大场景被系统性少给三到四倍; 而按 n_full 分配又与场景复杂度反着走

## 逐场景 awc 恢复率

| 场景 | 构件数 | 完整站数 | 预算站数 | Bdisp_maxmin | Bbim_offline | B5_occ_rng | B10_full |
|---|---|---|---|---|---|---|---|
| S/low | 53 | 13 | 6 | 0.981 | 0.892 | 0.982 | 0.985 |
| S/mid | 65 | 11 | 7 | 0.993 | 0.978 | 0.994 | 0.998 |
| S/high | 87 | 10 | 9 | 0.969 | 0.984 | 0.995 | 0.991 |
| M/low | 65 | 25 | 7 | 0.807 | 0.953 | 0.958 | 0.953 |
| M/mid | 81 | 21 | 9 | 0.858 | 0.957 | 0.941 | 0.966 |
| M/high | 123 | 17 | 13 | 0.888 | 0.989 | 0.949 | 0.982 |
| L/low | 81 | 44 | 9 | 0.830 | 0.960 | 0.933 | 0.935 |
| L/mid | 115 | 37 | 13 | 0.877 | 0.976 | 0.977 | 0.966 |
| L/high | 150 | 34 | 16 | 0.934 | 0.989 | 0.963 | 0.987 |

## 稳定性(按 9 类场景的**场景均值**统计)

| 方法 | 指标 | 均值 | 标准差 | 变异系数 | 最差场景 |
|---|---|---|---|---|---|
| Bdisp_maxmin | awc_gap_recovery | 0.904 | 0.064 | 0.071 | M/low 0.807 |
| Bdisp_maxmin | asset_recovery | 0.942 | 0.043 | 0.046 | S/high 0.832 |
| Bdisp_maxmin | crit_recall | 0.952 | 0.030 | 0.032 | S/high 0.879 |
| Bdisp_maxmin | dens_ok | 0.594 | 0.068 | 0.114 | L/low 0.450 |
| Bbim_offline | awc_gap_recovery | 0.964 | 0.029 | 0.030 | S/low 0.892 |
| Bbim_offline | asset_recovery | 0.958 | 0.037 | 0.038 | S/mid 0.893 |
| Bbim_offline | crit_recall | 0.958 | 0.039 | 0.040 | S/mid 0.892 |
| Bbim_offline | dens_ok | 0.611 | 0.046 | 0.075 | L/low 0.526 |
| B5_occ_rng | awc_gap_recovery | 0.966 | 0.021 | 0.022 | L/low 0.933 |
| B5_occ_rng | asset_recovery | 0.970 | 0.028 | 0.029 | S/low 0.903 |
| B5_occ_rng | crit_recall | 0.984 | 0.015 | 0.015 | S/low 0.951 |
| B5_occ_rng | dens_ok | 0.615 | 0.041 | 0.067 | M/low 0.564 |
| B10_full | awc_gap_recovery | 0.974 | 0.019 | 0.020 | L/low 0.935 |
| B10_full | asset_recovery | 0.973 | 0.025 | 0.026 | S/low 0.906 |
| B10_full | crit_recall | 0.984 | 0.014 | 0.014 | S/low 0.951 |
| B10_full | dens_ok | 0.628 | 0.038 | 0.060 | S/high 0.589 |

## B10 相对无信息基线 Bdisp 的优势, 逐规模

| 规模 | 指标 | Δ均值 | 95%CI | p |
|---|---|---|---|---|
| S | awc_gap_recovery | +0.010 | [+0.003, +0.019] | 0.0781 |
| S | asset_recovery | +0.045 | [-0.009, +0.103] | 0.1914 |
| M | awc_gap_recovery | +0.116 | [+0.095, +0.135] | 0.0234 |
| M | asset_recovery | +0.027 | [+0.011, +0.044] | 0.0703 |
| L | awc_gap_recovery | +0.082 | [+0.062, +0.105] | 0.0234 |
| L | asset_recovery | +0.022 | [+0.016, +0.028] | 0.0234 |