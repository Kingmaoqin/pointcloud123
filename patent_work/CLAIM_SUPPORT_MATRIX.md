# 权利要求支持矩阵

| 技术特征 | 源码位置 | 配置 | 实验输出 | 当前实现 | 适合独权 | 从属权利要求 | 仅可选 | 风险说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 获取IFC模型和点云 | scripts/run_real_pipeline.py:417-422; configs/data/cras.yaml | CRAS IFC和点云zip | preprocess_summary.json, cras_full_assoc_summary.json | 是 | 是 | 否 | 否 | 常规输入步骤，创造性弱 |
| IFC三角化 | patent_gap.ifc.reader由webapp.py:22和preprocess_summary引用 | craslabbim.ifc | preprocess_summary.json | 是 | 是 | 否 | 否 | 三角化本身通用 |
| 构件内共享边区域生长 | patches.__init__.py:41-81,162-187 | normal_threshold_deg=20 | patches_real.parquet | 是 | 是 | 否 | 否 | 与双法向约束组合较强 |
| 双法向约束 | patches.__init__.py:65-77 | normal_threshold_deg=20 | patch数量9438 | 是 | 是 | 否 | 否 | 建议写入独权 |
| Patch属性结构 | patches.__init__.py:205-228 | importance_map默认 | patch_scores_real.csv | 是 | 是 | 否 | 否 | 与后续融合直接关联 |
| 点云-IFC最近表面关联 | registration.__init__.py:134-175; cras_full_assoc_summary | association_distance=0.05 | matched/unmatched统计 | 是 | 是 | 否 | 否 | 关联算法本身需与Patch证据结合 |
| 构造观测/角度/几何/语义/材料证据 | evidence, semantics, materials模块 | 见PARAMETER_REGISTRY.csv | patch_scores*.csv | 是 | 是 | 否 | 否 | 多源证据体系可支撑 |
| 方向一致缺口指标 | gap/scoring.py:140-213 | D越大缺口越强 | patch_scores*.csv | 是 | 是 | 否 | 否 | 建议写入独权 |
| 可用证据权重重归一化 | gap/scoring.py:54-66 | DEFAULT_ALPHAS | real中D_sem/D_ang NaN仍有G_gap | 是 | 是 | 否 | 否 | 最强候选之一 |
| 候选视点生成 | viewpoints/ranking.py:59-130 | distances/az/el | candidate_view_ranking_real.csv | 是 | 是 | 否 | 否 | 应避免锁死数值 |
| 视点价值计算 | viewpoints/ranking.py:133-267 | FOV/range/eta | top1_view_value | 是 | 是 | 否 | 否 | 与补扫收益相关 |
| 闭环补扫更新 | closed_loop.py:13-143,180-273 | recovery_per_view | closed_loop_results*.csv | 是 | 是 | 否 | 否 | 当前为仿真 |
| 工程重要度加权 | gap/scoring.py:225-231 | lambda=0.30 | G_task列 | 是 | 否 | 是 | 否 | 从属较稳妥 |
| 贪心多视点冗余抑制 | viewpoints/ranking.py:270-285 | 可见Patch降为25% | candidate_view_greedy_real.csv | 是 | 否 | 是 | 否 | 实现较简单，放从属 |
| 遮挡射线检测 | 无 | 无 | 无 | 否 | 否 | 否 | 是 | 当前不能主张 |
