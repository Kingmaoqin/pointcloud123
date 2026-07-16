"""PR5 验收: K_i 退出行为回归 + B1 数值行为不变 + 扩展权重表激活。"""

import pandas as pd

from patent_gap.gap.scoring import (
    DEFAULT_ALPHAS, EXTENDED_ALPHAS, compute_patch_scores, normalize_alphas,
)


def _scene(with_extended=False):
    patches = pd.DataFrame({
        "patch_id": [0, 1, 2],
        "element_guid": ["e0", "e0", "e1"],
        "ifc_class": ["breaker"] * 3,
        "name": ["p0", "p1", "p2"],
        "area": [1.0, 2.0, 0.5],
        "engineering_importance": [0.9, 0.9, 0.4],
    })
    obs = pd.DataFrame({
        "patch_id": [0, 1, 2],
        "directly_observed": [True, False, True],
        "number_of_valid_views": [2, 0, 1],
        "projected_resolution_quality": [0.8, 0.0, 0.5],
        "registration_confidence": [0.5, 0.5, 0.5],
        "best_frontality": [0.9, 0.0, 0.5],
        "angular_diversity": [0.5, 0.0, 0.25],
        "coverage_ratio": [0.9, 0.0, 0.5],
        "density_ratio": [0.8, 0.0, 0.4],
    })
    if with_extended:
        obs["D_occ"] = [0.1, 0.95, 0.4]
        obs["D_rng"] = [0.2, 1.0, 0.6]
        obs["D_regsup"] = [0.3, 1.0, 0.5]
    sem = pd.DataFrame({"patch_id": [0, 1, 2]})   # D_sem 不可算 → K_i 退出
    mat = pd.DataFrame({"patch_id": [0, 1, 2]})
    return {"patches": patches, "observations": obs, "semantics": sem, "materials": mat}


def test_b1_numeric_behavior_unchanged():
    """六证据路径的 G_gap 与母专利权重表手算一致(K_i 重归一)。"""
    df = compute_patch_scores(_scene(False)).sort_values("patch_id").reset_index(drop=True)
    w = normalize_alphas(None)
    assert w == normalize_alphas(None, extended=False)
    # patch 1: D_obs/D_ang/D_geo 可算, D_sem/D_mat NaN → 只有三项参与
    row = df.iloc[1]
    d_obs = 0.35 * 1 + 0.25 * 1 + 0.20 * 1 + 0.20 * 0.5
    d_ang = 1.0
    d_geo = 1.0
    expect = ((w["alpha_obs"] * d_obs + w["alpha_ang"] * d_ang + w["alpha_geo"] * d_geo)
              / (w["alpha_obs"] + w["alpha_ang"] + w["alpha_geo"]))
    assert abs(row["G_gap"] - expect) < 1e-9
    assert "D_occ" not in df.columns or df["D_occ"].isna().all()


def test_extended_activates_new_weights():
    df = compute_patch_scores(_scene(True)).sort_values("patch_id").reset_index(drop=True)
    w = normalize_alphas(None, extended=True)
    assert set(EXTENDED_ALPHAS) == set(w)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    row = df.iloc[1]
    d_obs = 0.35 * 1 + 0.25 * 1 + 0.20 * 1 + 0.20 * 0.5
    num = (w["alpha_obs"] * d_obs + w["alpha_ang"] * 1.0 + w["alpha_geo"] * 1.0
           + w["alpha_occ"] * 0.95 + w["alpha_rng"] * 1.0 + w["alpha_regsup"] * 1.0)
    den = (w["alpha_obs"] + w["alpha_ang"] + w["alpha_geo"]
           + w["alpha_occ"] + w["alpha_rng"] + w["alpha_regsup"])
    assert abs(row["G_gap"] - num / den) < 1e-9
    # 高遮挡+低密度 patch 的缺口分应高于观测充分 patch
    assert df.iloc[1]["G_gap"] > df.iloc[0]["G_gap"]


def test_default_alphas_untouched():
    """母专利权重表逐值回归(Agent 硬性纪律)。"""
    assert DEFAULT_ALPHAS == {
        "alpha_sem": 0.16, "alpha_mat_missing": 0.14, "alpha_mat_conflict": 0.10,
        "alpha_obs": 0.22, "alpha_ang": 0.16, "alpha_geo": 0.22,
    }
