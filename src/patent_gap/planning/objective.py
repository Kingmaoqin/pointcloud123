"""公式(44) 联合轨迹目标 J(π) 与时间预算约束。

J(π) = F/F_ub + λ_reg·min_k R^reg(v_k) − λ_len·Σdist/L_diag − λ_sta·M/M_max
T(π) = Σdist/v_move + M·t_scan ≤ B(约束, 非目标项)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .route import route_length
from .set_select import coverage_value


@dataclass
class ObjectiveParams:
    lambda_reg: float = 0.3
    lambda_len: float = 0.4
    lambda_sta: float = 0.2
    v_move: float = 0.5       # m/s
    t_scan: float = 180.0     # s/站
    m_max: int = 8


def trajectory_time(dist_matrix: np.ndarray, route: list[int], p: ObjectiveParams) -> float:
    n_scan = max(len(route) - 1, 0)  # route[0] = v_0 当前位姿, 不扫描
    return route_length(dist_matrix, route) / p.v_move + n_scan * p.t_scan


def trajectory_objective(c_jv: np.ndarray, gains_j: np.ndarray,
                         r_reg_v: np.ndarray, dist_matrix: np.ndarray,
                         route: list[int], l_diag: float,
                         params: ObjectiveParams | None = None) -> dict:
    """route 含起点 v_0(索引0对应距离矩阵首行), 其余为选中站(列索引=c_jv列)。"""
    p = params or ObjectiveParams()
    stations = [v - 1 for v in route[1:]]  # 距离矩阵中 0 = v_0, i+1 = 候选 i
    f = coverage_value(c_jv, gains_j, stations)
    f_ub = float(gains_j.sum())
    length = route_length(dist_matrix, route)
    m = len(stations)
    j = (f / max(f_ub, 1e-12)
         + p.lambda_reg * (float(np.min(r_reg_v[stations])) if stations else 0.0)
         - p.lambda_len * length / max(l_diag, 1e-9)
         - p.lambda_sta * m / max(p.m_max, 1))
    return {
        "J": float(j),
        "F": float(f),
        "F_norm": float(f / max(f_ub, 1e-12)),
        "path_length_m": float(length),
        "n_stations": m,
        "time_s": trajectory_time(dist_matrix, route, p),
    }
