"""公式(30) 传感器参数组 Θ = (r_min, r_opt, r_max, Δθ, Δφ, σ_r, f_pulse)。

单位: r 为米; dtheta/dphi 为弧度(角分辨率); sigma_r 为米(测距噪声1σ);
f_pulse 为 Hz。随 station NPZ 的 `sensor` 字段持久化, 禁止硬编码。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SensorModel:
    r_min: float
    r_opt: float
    r_max: float
    dtheta: float  # 弧度
    dphi: float    # 弧度
    sigma_r: float
    f_pulse: float

    def __post_init__(self) -> None:
        if not (0 <= self.r_min < self.r_max):
            raise ValueError(f"invalid range gates r_min={self.r_min} r_max={self.r_max}")
        if self.dtheta <= 0 or self.dphi <= 0:
            raise ValueError("angular resolution must be positive")
        if self.sigma_r < 0:
            raise ValueError("sigma_r must be non-negative")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "SensorModel":
        return cls(**json.loads(s))

    @classmethod
    def from_config(cls, cfg: dict) -> "SensorModel":
        """由 yaml 配置构造; 角分辨率以度给出(dtheta_deg)。"""
        dtheta = math.radians(float(cfg.get("dtheta_deg", 0.04)))
        dphi = math.radians(float(cfg.get("dphi_deg", cfg.get("dtheta_deg", 0.04))))
        r = cfg.get("r", [0.5, 10.0, 60.0])
        return cls(
            r_min=float(r[0]), r_opt=float(r[1]), r_max=float(r[2]),
            dtheta=dtheta, dphi=dphi,
            sigma_r=float(cfg.get("sigma_r", 0.005)),
            f_pulse=float(cfg.get("f_pulse", 300000.0)),
        )


# RIEGL VZ-400 量级预设(【推断】110kV 站内工作距离; 实验中作为变量)
DEFAULT_TLS = SensorModel(
    r_min=0.5, r_opt=10.0, r_max=60.0,
    dtheta=math.radians(0.04), dphi=math.radians(0.04),
    sigma_r=0.005, f_pulse=300000.0,
)
