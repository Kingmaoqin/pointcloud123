"""公式(44) 联合轨迹目标 J(π) 与时间预算约束。

J(π) = F/F_ub + λ_reg·min_k R^reg(v_k) − λ_len·Σdist/L_diag − λ_sta·M/M_max
T(π) = Σdist/v_move + M·t_scan ≤ B(约束, 非目标项)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .route import route_length
from .set_select import coverage_value


# 作业时间标定的**唯一来源**。此前 v_move/t_scan 硬编码在五处(闭环的
# scan_equiv、本文件、run_e2/e3/e5 各一处), 改一处不改另一处会让预算与评价
# 悄悄脱钩 —— 公式(42) 的 cost 与时间归一指标必须用同一组常数。
V_MOVE_DEFAULT = 0.5      # m/s 三脚架搬站步行速度
T_SCAN_DEFAULT = 180.0    # s   单站架设 + 扫描耗时


def episode_time_s(path_len_m: float, n_stations: int,
                   v_move: float = V_MOVE_DEFAULT,
                   t_scan: float = T_SCAN_DEFAULT) -> float:
    """T = 路径/v_move + 站数·t_scan。评价与预算共用。"""
    return float(path_len_m) / v_move + float(n_stations) * t_scan


@dataclass
class ObjectiveParams:
    # 与 V2Config.lambda_reg 保持一致。两处曾各取 0.5 / 0.3, 而同一次选站决策
    # 的两个环节(候选打分与单步 J)分别用其中一个, 表达的设计意图并不相同。
    lambda_reg: float = 0.5
    lambda_len: float = 0.4
    lambda_sta: float = 0.2
    v_move: float = V_MOVE_DEFAULT
    t_scan: float = T_SCAN_DEFAULT
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
