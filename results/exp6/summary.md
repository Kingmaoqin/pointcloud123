# E6: 真实 IFC + 真实点云（CRAS labs@FEUP）

- config_hash `e25e566192`, commit `795df72-dirty`, runs=12
- 场景: 604187 顶点 / 1197750 三角面 / 256 构件 / 7019 分块, 18.2×24.49 m, 表面 3836.5 m²

> **限制**：数据集只有一份融合 ASC、无逐站位姿，补扫站的重新采集由射线
> 仿真器在真实 IFC 网格上完成——几何与缺口是真的，重扫是仿真的。CRAS 是
> 实验楼而非变电站，本实验检验流水线能否吃真实数据，不替代变电站效果验证。
> `measured` 模式下构件级匹配点数按面积分摊到分块，是一处近似。

| 缺口真值 | 方法 | awc恢复率 | 构件等权 | 关键设备召回 | 路径(m) | 站数 | 终止原因 |
|---|---|---|---|---|---|---|---|
| measured | Bdisp_maxmin | 0.908 | 0.964 | 0.956 | 321 | 28 | rounds |
| measured | Bbim_offline | 0.938 | 0.947 | 0.921 | 366 | 28 | rounds |
| measured | B1_patent | 0.851 | 0.889 | 0.877 | 321 | 28 | rounds |
| measured | B5_occ_rng | 0.973 | 0.981 | 0.974 | 371 | 28 | rounds |
| measured | B10_full | 0.959 | 0.981 | 0.956 | 178 | 28 | rounds |
| measured | B11_disc | 0.959 | 0.981 | 0.956 | 178 | 28 | rounds |
| synthetic | Bdisp_maxmin | 0.454 | 0.839 | 0.583 | 321 | 28 | rounds |
| synthetic | Bbim_offline | 0.721 | 0.733 | 0.417 | 366 | 28 | rounds |
| synthetic | B1_patent | 0.138 | 0.468 | 0.083 | 321 | 28 | rounds |
| synthetic | B5_occ_rng | 0.874 | 0.871 | 0.750 | 371 | 28 | rounds |
| synthetic | B10_full | 0.797 | 0.879 | 0.625 | 178 | 28 | rounds |
| synthetic | B11_disc | 0.797 | 0.879 | 0.625 | 178 | 28 | rounds |