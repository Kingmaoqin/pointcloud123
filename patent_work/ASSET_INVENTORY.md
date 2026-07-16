# 资产审计

| 类别 | 资产 | 路径 | 审计结论 |
| --- | --- | --- | --- |
| 核心源码 | patches | src/patent_gap/patches/__init__.py | 三角面法向、面积、中心、共享边邻接、区域生长、Patch属性 |
| 核心源码 | evidence | src/patent_gap/evidence/__init__.py | synthetic/raycast/real三类观测、角度、几何证据 |
| 核心源码 | semantics | src/patent_gap/semantics/__init__.py | IFC/CRAS语义映射与JS散度 |
| 核心源码 | materials | src/patent_gap/materials/__init__.py | IFC材料缺失与RGB材料冲突 |
| 核心源码 | gap | src/patent_gap/gap/scoring.py | 六类缺口指标融合、工程重要度、构件排序 |
| 核心源码 | viewpoints | src/patent_gap/viewpoints/ranking.py | 候选视点生成、可见性筛选、价值计算、贪心降权 |
| 核心源码 | closed_loop | src/patent_gap/simulation/closed_loop.py | 补扫后观测证据更新与闭环迭代 |
| 核心源码 | registration | src/patent_gap/registration/__init__.py | CRAS粗平移和可选ICP刚性配准 |
| Web | webapp | src/patent_gap/webapp.py | Gradio数据源、模型导入、参数滑块、三维图、补扫按钮 |
| 脚本 | synthetic | scripts/run_synthetic_pipeline.py | Y>14m受控留出、定量评价、视点和闭环输出 |
| 脚本 | real | scripts/run_real_pipeline.py | CRAS真实诊断，不含Patch级二元真值 |
| 脚本 | raycast | scripts/gen_synthetic_scan.py | 9站虚拟扫描、1度射线、patch_stats输出 |
| 数据 | IFC | data/raw/craslabbim.ifc | 真实IFC；三角化后256构件、1197750三角面 |
| 数据 | 点云 | data/raw/craslabannotated.zip | CRAS真实室内点云；全量关联点数584701977 |
| 数据 | sample | data/processed/cras_point_sample*.parquet | 样本点20000，匹配率0.5077 |
| 输出 | synthetic_summary | outputs/reports/synthetic_summary.json | AUROC=0.9958, AUPRC=0.9866, 仅受控留出实验 |
| 输出 | real_summary | outputs/reports/real_data_summary.json | 诊断-only；Patch=9438，候选视点=810 |
