# 数据血缘

## 原始数据
- IFC：`data/raw/craslabbim.ifc`，配置见 `configs/data/cras.yaml`，MD5字段为 `e20658f0d2d9e13c62363169b7fa3193`。
- 点云压缩包：`data/raw/craslabannotated.zip`，配置MD5字段为 `e5ecedab8f2a1d1f91861a3aec028a72`。

## 预处理
- `outputs/reports/preprocess_summary.json` 记录 IFC 已三角化，构件数 `256`，材料数 `24`，顶点数 `604187`，三角面数 `1197750`。
- 输出包括 `ifc_mesh.npz`、`ifc_elements.parquet`、`ifc_materials.parquet`、`triangle_element_map.npy`、`triangle_element_guid.npy`。

## 点云关联
- 样本关联：`cras_point_sample_associations.parquet`，输入 `20000` 点，匹配 `10154` 点，未匹配 `9846` 点，阈值 `0.05` m。
- 全量关联摘要：`cras_full_assoc_summary.json`，输入 `584701977` 点，匹配 `12835294` 点，未匹配 `571866683` 点，匹配率 `0.0220`。

## 受控synthetic数据
- `scripts/gen_synthetic_scan.py` 生成 9 个虚拟扫描站，射线方位角 0-359 度、仰角 -30 至 80 度、步长 1 度。
- `scripts/run_synthetic_pipeline.py` 将 `centroid_y > 14.0m` 定义为预声明留出标签；该规则是实验构造，不是自然建筑规律。
- `patch_stats.csv` 当前 `803` 个Patch有射线命中统计，`scanner_positions.csv` 当前 `9` 个站位。

## 输出
- synthetic定量输出：`outputs/reports/synthetic_summary.json`；标签有Patch级真值。
- CRAS真实诊断输出：`outputs/reports/real_data_summary.json`；无Patch级二元缺失真值，只作工程诊断。
