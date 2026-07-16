# 公式到源码映射

| 公式/指标 | 表达式 | 源码路径:行号 | 输入字段 | 输出字段 | 调用入口 |
| --- | --- | --- | --- | --- | --- |
| 三角面法向 | n=normalize((v1-v0)x(v2-v0)) | patches.__init__.py:18-27 | vertices, faces | face_normals | build_patches |
| 三角面积 | A=0.5||(v1-v0)x(v2-v0)|| | patches.__init__.py:30-34 | vertices, faces | face_areas | build_patches |
| Patch中心 | c=sum(A_t c_t)/sum(A_t) | patches.__init__.py:195-197 | local_areas, local_centroids | centroid | build_patches |
| Patch法向 | n_p=normalize(sum(w_t n_t)) | patches.__init__.py:197-199 | local_normals, area weights | normal | build_patches |
| 区域生长约束 | dot(n_cur,n_nb)>=cosθ 且 dot(n_seed,n_nb)>=cosθ | patches.__init__.py:56-81 | local_normals, adjacency | patch labels | build_patches |
| JS散度 | D_JS=1/2 KL(p||m)+1/2 KL(q||m), m=(p+q)/2 | gap/scoring.py:20-41 | p_bim,p_obs | D_sem | compute_patch_scores |
| D_obs | clip(.35M+.25(2-N)/2+.20(1-Q)+.20(1-R)) | gap/scoring.py:156-173; closed_loop.py:110-117 | observed,Nvalid,Qproj,Rreg | D_obs | compute_patch_scores/closed_loop |
| D_ang | clip(1-(.50F+.30N/2+.20A)) | gap/scoring.py:176-190; closed_loop.py:118-127 | best_frontality,Nvalid,angular_diversity | D_ang | compute_patch_scores/closed_loop |
| D_geo | clip(.60(1-C)+.40(1-Rρ)) | gap/scoring.py:191-200; closed_loop.py:128-133 | coverage_ratio,density_ratio | D_geo | compute_patch_scores/closed_loop |
| 材料缺失 | D_mat_missing=0 if present else 1 | materials.__init__.py:159 | material_present | D_mat_missing | build_material_table |
| 材料冲突 | D_mat_conflict=1 if clear mismatch else 0 | materials.__init__.py:141-160 | IFC category, RGB category | D_mat_conflict | build_material_table |
| 可用权重融合 | G_gap=sum(a_k D_k)/sum(a_k), skip NaN | gap/scoring.py:54-66,215-224 | D components, alpha | G_gap | compute_patch_scores |
| 工程重要度 | G_task=clip(G_gap(1+λI),0,2) | gap/scoring.py:225-231 | G_gap,I,lambda | G_task | compute_patch_scores |
| 构件评分 | .25mean+.25max+.20p90+.20high_area+.10importance | gap/scoring.py:285-310 | patch scores by element | G_component | rank_components |
| 候选视点方向 | Rodrigues旋转，方位角绕bitangent，俯仰角绕tangent | viewpoints/ranking.py:35-56,97-110 | centroid,normal,az,el,distance | position,orientation | generate_candidates |
| 可见性 | frontality>fmin, alignment>=cos(FOV/2), range<=max | viewpoints/ranking.py:197-212 | patch, candidate | visible_mask | score_candidates |
| 视点质量 | q=frontality*exp(-(d-2.5)^2/8)*1/(1+.15d) | viewpoints/ranking.py:236-240 | frontality,distance | q | score_candidates |
| 视点价值 | V=sum(G_gap*q*area)-eta*overlap | viewpoints/ranking.py:241-260 | visible patches | value | score_candidates |
| 补扫更新 | x'=x+r_eff(1-x), r_eff=recovery(0.5+G_gap) | closed_loop.py:56-80 | visible_patch_ids,recovery,G_gap | updated observations | apply_supplemental_observation |
