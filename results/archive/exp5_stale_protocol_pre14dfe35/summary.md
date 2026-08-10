# E5: 跨规模/密度稳定性

- config_hash `7016dc479a`, commit `a7a24fa`, runs=43
- 预算按完整扫描站数的 35% 给, 不再对所有场景一律 6 站 —— 固定预算下 S 场景拿到完整扫描的 46–60%, L 场景只有 13–17%

## 逐场景 awc 恢复率

| 场景 | 完整站数 | 预算站数 | Bdisp_maxmin | B5_occ_rng | B10_full |
|---|---|---|---|---|---|
| S/low | 13 | 5 | 0.935 | 0.978 | 0.980 |
| S/mid | 11 | 4 | 0.927 | 0.909 | 0.964 |
| S/high | 10 | 4 | 0.787 | 0.688 | 0.714 |
| M/low | 25 | 9 | 0.834 | 0.968 | 0.970 |
| M/mid | 21 | 7 | 0.834 | 0.844 | 0.940 |

## 稳定性(按 9 类场景的**场景均值**统计)

| 方法 | 指标 | 均值 | 标准差 | 变异系数 | 最差场景 |
|---|---|---|---|---|---|
| Bdisp_maxmin | awc_gap_recovery | 0.863 | 0.058 | 0.067 | S/high 0.787 |
| Bdisp_maxmin | asset_recovery | 0.828 | 0.147 | 0.178 | S/high 0.553 |
| Bdisp_maxmin | crit_recall | 0.831 | 0.127 | 0.152 | S/high 0.599 |
| Bdisp_maxmin | dens_ok | 0.494 | 0.079 | 0.160 | S/high 0.375 |
| B5_occ_rng | awc_gap_recovery | 0.877 | 0.106 | 0.121 | S/high 0.688 |
| B5_occ_rng | asset_recovery | 0.712 | 0.258 | 0.362 | S/high 0.334 |
| B5_occ_rng | crit_recall | 0.719 | 0.268 | 0.373 | S/high 0.334 |
| B5_occ_rng | dens_ok | 0.482 | 0.099 | 0.205 | S/high 0.334 |
| B10_full | awc_gap_recovery | 0.914 | 0.101 | 0.110 | S/high 0.714 |
| B10_full | asset_recovery | 0.800 | 0.222 | 0.278 | S/high 0.388 |
| B10_full | crit_recall | 0.816 | 0.226 | 0.276 | S/high 0.382 |
| B10_full | dens_ok | 0.517 | 0.100 | 0.193 | S/high 0.354 |

## B10 相对无信息基线 Bdisp 的优势, 逐规模

| 规模 | 指标 | Δ均值 | 95%CI | p |
|---|---|---|---|---|
| S | awc_gap_recovery | +0.003 | [-0.060, +0.060] | 0.8945 |
| S | asset_recovery | -0.072 | [-0.193, +0.014] | 0.5391 |
| M | awc_gap_recovery | +0.130 | [+0.113, +0.153] | 0.2500 |
| M | asset_recovery | +0.038 | [+0.007, +0.080] | 0.2500 |