# 实现真实性表

| 技术点 | 当前实现 | 证据 | 限制 |
| --- | --- | --- | --- |
| 共享边区域生长 | 是 | patches.__init__.py:41-81, 186-187 | 仅在同一IFC构件内对三角面按共享边邻接生长 |
| 双法向约束 | 是 | patches.__init__.py:65-77 | 同时检查当前面法向和种子面法向与邻面夹角 |
| Patch属性 | 是 | patches.__init__.py:205-228 | 含面积、中心、法向、IFC类别、材料、工程重要度 |
| 点云配准 | 部分实现 | registration.__init__.py:18-20,134-175 | 当前有粗平移和可选ICP；全量关联输出使用open3d最近面 |
| 未匹配点保留 | 是 | cras_full_assoc_summary.json | unmatched_points=571866683 |
| 遮挡射线检测 | 当前视点评分未实现 | viewpoints/ranking.py:207-212 | 只按frontality、FOV、range筛选；遮挡只能写可选实施方式 |
| Web模型导入 | 是 | webapp.py:378-457 | IFC三角化；通用网格作为单Patch；评分CSV代理 |
| 真实CRAS监督精度 | 不支持 | real_data_summary.json | 无Patch级二元缺失真值，真实结果仅为工程诊断 |
