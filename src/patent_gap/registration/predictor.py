"""公式(36)–(40) 配准感知模块。

(36) 双区域划分 L_v^new / L_v^ovl
(37) 预期重叠率 O^reg(v)(面积-质量加权)
(38) point-to-plane Fisher 信息 H(v) 与退化度 Degen(v)
(39) 配准支持增益 R^reg(v)
(40) 配准支持缺口证据 D_i^regsup
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAU_GAP_DEFAULT = 0.55
TAU_COV_DEFAULT = 0.5
O_MIN_DEFAULT = 0.30
GAMMA_DEFAULT = 1.0
EPS0 = 1e-6


def split_regions(gap_scores: np.ndarray, coverage: np.ndarray,
                  in_fov: np.ndarray,
                  tau_gap: float = TAU_GAP_DEFAULT,
                  tau_cov: float = TAU_COV_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """公式(36): 返回 (mask_new, mask_ovl), 两集合可交叠。"""
    in_fov = np.asarray(in_fov, dtype=bool)
    mask_new = in_fov & (np.asarray(gap_scores) >= tau_gap)
    mask_ovl = in_fov & (np.asarray(coverage) >= tau_cov)
    return mask_new, mask_ovl


def expected_overlap(areas: np.ndarray, quality: np.ndarray,
                     mask_ovl: np.ndarray, in_fov: np.ndarray) -> float:
    """公式(37)。"""
    aq = np.asarray(areas, dtype=np.float64) * np.asarray(quality, dtype=np.float64)
    denom = float(aq[np.asarray(in_fov, dtype=bool)].sum()) + EPS0
    return float(aq[np.asarray(mask_ovl, dtype=bool)].sum()) / denom


def degeneracy(points: np.ndarray, normals: np.ndarray,
               centroid: np.ndarray | None = None) -> float:
    """公式(38): Degen(v) = 1 − λ_min(H̃)/λ_max(H̃) ∈ [0,1)。

    H̃ 为平移/旋转对角块分别按 trace 归一后的 point-to-plane Fisher 信息。
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if len(points) < 6:
        return 1.0
    if centroid is None:
        centroid = points.mean(axis=0)
    lever = np.cross(points - centroid[None, :], normals)
    J = np.hstack([normals, lever])                      # (K,6)
    H = J.T @ J
    Ht, Hr = H[:3, :3], H[3:, 3:]
    tt, tr = float(np.trace(Ht)), float(np.trace(Hr))
    if tt <= 1e-12 or tr <= 1e-12:
        return 1.0
    Hn = np.zeros((6, 6))
    Hn[:3, :3] = Ht / tt
    Hn[3:, 3:] = Hr / tr
    # 块间耦合项保留会重引入量纲差, 按方案只归一对角块并置零交叉块
    lam = np.linalg.eigvalsh(Hn)
    lam_max = float(lam[-1])
    if lam_max <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - max(lam[0], 0.0) / lam_max, 0.0, 1.0))


def r_reg(overlap: float, degen: float,
          o_min: float = O_MIN_DEFAULT, gamma: float = GAMMA_DEFAULT) -> float:
    """公式(39) 配准支持增益 ∈ [0,1]。"""
    return float(np.clip(overlap / o_min, 0.0, 1.0) * (1.0 - degen) ** gamma)


@dataclass
class RegPrediction:
    overlap: float
    degen: float
    r_reg: float


def predict_registration(anchor_points: np.ndarray, anchor_normals: np.ndarray,
                         areas: np.ndarray, quality: np.ndarray,
                         mask_ovl: np.ndarray, in_fov: np.ndarray,
                         o_min: float = O_MIN_DEFAULT,
                         gamma: float = GAMMA_DEFAULT) -> RegPrediction:
    """一站式: (37)+(38)→(39)。anchor_* 为 L_v^ovl 的公式(26)采样点及法向。"""
    ov = expected_overlap(areas, quality, mask_ovl, in_fov)
    dg = degeneracy(anchor_points, anchor_normals)
    return RegPrediction(overlap=ov, degen=dg, r_reg=r_reg(ov, dg, o_min, gamma))


def regsup_gap_evidence(r_by_station: dict[str, dict[int, float]],
                        patch_ids: list[int]) -> dict[int, float]:
    """公式(40): D_i^regsup = 1 − max_v [ I[i∈L_v]·R^reg(v) ]。

    r_by_station: {station_id: {patch_id ∈ L_v: R^reg(v)}}。
    """
    out = {}
    for pid in patch_ids:
        best = 0.0
        for rmap in r_by_station.values():
            best = max(best, rmap.get(int(pid), 0.0))
        out[int(pid)] = 1.0 - best
    return out
