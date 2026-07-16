"""公式(31)–(34) 量程与点密度模块。

(31) ρ̂_{i,v} = cosα/(d²ΔθΔφ)·I[r_min≤d≤r_max]   [点/m²]
(32) η_inc(α) = max(cosα−cosα_max,0)/(1−cosα_max), α_max=80°(默认关闭)
(33) g_rng = clip(ρ̂/ρ_req, 0, 1),  ρ_req = ρ_0(1+λ_E·E_i)
(34) D_rng = 1 − clip(ρ_obs/ρ_req, 0, 1),  ρ_obs = N_pts/A
"""

from __future__ import annotations

import numpy as np

from ..sensors.model import SensorModel

RHO0_DEFAULT = 400.0     # 点/m²(【推断】pilot 标定)
LAMBDA_E_DEFAULT = 1.0
ALPHA_MAX_DEG = 80.0


def expected_density(d: np.ndarray, cos_inc: np.ndarray, sm: SensorModel,
                     use_eta_inc: bool = False) -> np.ndarray:
    """公式(31); use_eta_inc=True 时乘公式(32)掠射角因子。"""
    d = np.asarray(d, dtype=np.float64)
    cos_inc = np.clip(np.asarray(cos_inc, dtype=np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = cos_inc / (np.maximum(d, 1e-9) ** 2 * sm.dtheta * sm.dphi)
    rho = np.where((d >= sm.r_min) & (d <= sm.r_max), rho, 0.0)
    if use_eta_inc:
        rho = rho * eta_incidence(cos_inc)
    return rho


def eta_incidence(cos_inc: np.ndarray, alpha_max_deg: float = ALPHA_MAX_DEG) -> np.ndarray:
    """公式(32) 入射角有效回波因子。"""
    cmax = np.cos(np.deg2rad(alpha_max_deg))
    return np.maximum(np.asarray(cos_inc, dtype=np.float64) - cmax, 0.0) / (1.0 - cmax)


def rho_required(importance: np.ndarray, rho0: float = RHO0_DEFAULT,
                 lambda_e: float = LAMBDA_E_DEFAULT) -> np.ndarray:
    """公式(33) 右式: ρ_req = ρ_0(1+λ_E·E_i)。"""
    return rho0 * (1.0 + lambda_e * np.clip(np.asarray(importance, dtype=np.float64), 0, 1))


def g_rng(d: np.ndarray, cos_inc: np.ndarray, rho_req: np.ndarray | float,
          sm: SensorModel, use_eta_inc: bool = False) -> np.ndarray:
    """公式(33) 量程-密度质量项 ∈ [0,1]。"""
    return np.clip(expected_density(d, cos_inc, sm, use_eta_inc) / np.maximum(rho_req, 1e-9), 0.0, 1.0)


def range_density_gap(n_points: np.ndarray, areas: np.ndarray,
                      rho_req: np.ndarray | float) -> np.ndarray:
    """公式(34) 量程密度缺口证据 D_rng ∈ [0,1]。"""
    rho_obs = np.asarray(n_points, dtype=np.float64) / np.maximum(np.asarray(areas, dtype=np.float64), 1e-9)
    return 1.0 - np.clip(rho_obs / np.maximum(rho_req, 1e-9), 0.0, 1.0)
