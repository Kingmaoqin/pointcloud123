# E6: 真实 IFC + 真实点云（CRAS labs@FEUP）

- config_hash `103897b322`, commit `30ed463-dirty`, runs=12
- 场景: 604187 顶点 / 1197750 三角面 / 256 构件 / 7019 分块, 18.2×24.49 m, 表面 3836.5 m²

> **限制**：数据集只有一份融合 ASC、无逐站位姿，补扫站的重新采集由射线
> 仿真器在真实 IFC 网格上完成——几何与缺口是真的，重扫是仿真的。CRAS 是
> 实验楼而非变电站，本实验检验流水线能否吃真实数据，不替代变电站效果验证。
> `measured` 模式下构件级匹配点数按面积分摊到分块，是一处近似。

| 缺口真值 | 方法 | awc恢复率 | 构件等权 | 关键设备召回 | 路径(m) | 站数 | 终止原因 |
|---|---|---|---|---|---|---|---|
| measured | Bdisp_maxmin | 0.654 | 0.652 | 0.763 | 31 | 6 | rounds |
| measured | Bbim_offline | 0.654 | 0.649 | 0.763 | 18 | 6 | rounds |
| measured | B1_patent | 0.653 | 0.647 | 0.763 | 10 | 2 | no_candidate |
| measured | B5_occ_rng | 0.654 | 0.650 | 0.763 | 13 | 2 | no_candidate |
| measured | B10_full | 0.653 | 0.647 | 0.763 | 12 | 2 | no_candidate |
| measured | B11_disc | 0.653 | 0.647 | 0.763 | 12 | 2 | no_candidate |
| synthetic | Bdisp_maxmin | 0.025 | 0.108 | 0.047 | 31 | 6 | rounds |
| synthetic | Bbim_offline | 0.022 | 0.064 | 0.047 | 18 | 6 | rounds |
| synthetic | B1_patent | 0.019 | 0.039 | 0.047 | 10 | 2 | no_candidate |
| synthetic | B5_occ_rng | 0.022 | 0.075 | 0.047 | 13 | 2 | no_candidate |
| synthetic | B10_full | 0.019 | 0.045 | 0.047 | 12 | 2 | no_candidate |
| synthetic | B11_disc | 0.019 | 0.045 | 0.047 | 12 | 2 | no_candidate |