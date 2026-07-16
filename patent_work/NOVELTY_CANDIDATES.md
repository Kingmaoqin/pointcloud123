# 创造性候选审计

| 候选特征 | 真实性 | 建议位置 | 证据 |
| --- | --- | --- | --- |
| IFC构件内共享边区域生长 | 已实现 | 适合 | patches.__init__.py:41-81,162-187 |
| 双法向约束抑制链式合并 | 已实现 | 适合 | patches.__init__.py:65-77 |
| Patch含面积/中心/法向/语义/材料/重要度 | 已实现 | 适合 | patches.__init__.py:205-228 |
| 点云-IFC关联并保留未匹配点 | 已实现于数据输出 | 从属 | cras_full_assoc_summary.json; registration.__init__.py |
| 多源证据方向一致化 | 已实现 | 适合 | gap/scoring.py:140-224 |
| 可用指标集合重归一化 | 已实现 | 适合 | gap/scoring.py:54-66 |
| 缺口、面积和视点质量联合收益 | 已实现 | 适合 | viewpoints/ranking.py:236-260 |
| 基于中心法向生成多距离/方位/俯仰视点 | 已实现 | 适合 | viewpoints/ranking.py:59-130 |
| 闭环更新证据并重算下一视点 | 已实现 | 适合 | closed_loop.py:180-273 |
| 工程重要度加权 | 已实现 | 从属 | gap/scoring.py:225-231 |
| 顺序多视点冗余抑制 | 已实现为降权 | 从属 | viewpoints/ranking.py:270-285 |
| 遮挡射线检测 | 当前未实现 | 仅可选 | 不得进入独立权利要求 |
