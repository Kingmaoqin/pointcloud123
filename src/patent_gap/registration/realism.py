"""让公式(36)–(40) 有配准误差可预测：按实际重叠与实际退化度判定配准成败。

原仿真器不做配准，也不施加位姿扰动，于是 R^reg(v) 预测的东西在世界里根本不
存在——候选间 Degen 的动态范围只有 0.02，配准感知项的取值差异无从检验。

这里补上被预测的量本身。设站后按该站**实际**落到已有覆盖区上的重叠点数与该
重叠区的实际退化度，给出配准位姿误差的量级：point-to-plane 配准的
Cramér–Rao 下界随有效观测数按 1/√K 收缩，随信息矩阵条件数 λ_max/λ_min 放大，
而公式(38) 的 Degen = 1 − λ_min/λ_max 正是该条件数的单调变换：

    σ_pose ≈ σ_meas / √K · √(1/(1 − Degen))

重叠率低于 O_min 时按配准失败处理——拼不进全局坐标系的一站，其数据不可用。
失败是可检测的（残差显著偏大），所以规划器与评价都应把该站剔除，代价是这一站
的路程与时间照付。这正是配准感知规划要避免的浪费。
"""

from __future__ import annotations

import numpy as np

SIGMA_POSE_FLOOR = 1e-4     # m: 再好的几何也有的系统性下限
DEGEN_CAP = 0.999           # 完全退化时条件数发散, 截断避免 inf


def pose_error_sigma(n_overlap: int, degen: float, sigma_meas: float) -> float:
    """配准位姿误差的 1σ 量级(米)。n_overlap 为重叠区有效点数。"""
    if n_overlap < 3:
        return float("inf")
    amp = 1.0 / max(1.0 - min(float(degen), DEGEN_CAP), 1e-3)
    return float(SIGMA_POSE_FLOOR
                 + sigma_meas / np.sqrt(float(n_overlap)) * np.sqrt(amp))


def register_station(overlap: float, n_overlap: int, degen: float,
                     sigma_meas: float, rng: np.random.Generator,
                     o_min: float = 0.30, tol_m: float = 0.05) -> tuple[bool, float]:
    """判定一站能否拼入全局坐标系, 返回 (是否成功, 实际位姿误差 m)。

    overlap 低于 O_min 直接判失败: 没有足够公共区域可配准, 与误差大小无关。
    """
    if overlap < o_min or n_overlap < 3:
        return False, float("inf")
    sigma = pose_error_sigma(n_overlap, degen, sigma_meas)
    err = float(abs(rng.normal(0.0, sigma)))
    return err <= tol_m, err
