"""1.1 节 station_{s}.npz 统一逐站中间格式 — 所有模块唯一入口。

Schema:
  points     float32 (N,3)  点云(frame 字段标明坐标系)
  origin     float64 (3,)   扫描仪原点(全局系, 米)
  quat_wxyz  float64 (4,)   扫描仪姿态四元数
  station_id str            站唯一 ID
  timestamp  float64        采集时间(0=未知)
  intensity  float32 (N,) 可选
  frame      str            "global" | "sensor"
  sensor     json str       传感器参数快照(公式30 Θ)
可选扩展键(仿真真值, 评价专用, 算法不得读取):
  hit_patch  int64 (N,)     每点命中 Patch id(-1=非Patch表面)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sensors.model import SensorModel

_VALID_FRAMES = ("global", "sensor")


@dataclass
class StationScan:
    points: np.ndarray
    origin: np.ndarray
    quat_wxyz: np.ndarray
    station_id: str
    sensor: SensorModel
    frame: str = "global"
    timestamp: float = 0.0
    intensity: np.ndarray | None = None
    hit_patch: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 3)
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        self.quat_wxyz = np.asarray(self.quat_wxyz, dtype=np.float64).reshape(4)
        if self.intensity is not None:
            self.intensity = np.asarray(self.intensity, dtype=np.float32).reshape(-1)
        if self.hit_patch is not None:
            self.hit_patch = np.asarray(self.hit_patch, dtype=np.int64).reshape(-1)
        validate_station(self)


def validate_station(scan: StationScan) -> None:
    if scan.frame not in _VALID_FRAMES:
        raise ValueError(f"frame must be one of {_VALID_FRAMES}, got {scan.frame!r}")
    if not np.isfinite(scan.points).all():
        raise ValueError("points contain non-finite values")
    if not np.isfinite(scan.origin).all():
        raise ValueError("origin contains non-finite values")
    qn = float(np.linalg.norm(scan.quat_wxyz))
    if not (0.99 < qn < 1.01):
        raise ValueError(f"quat_wxyz must be unit-norm, got |q|={qn:.4f}")
    if not scan.station_id:
        raise ValueError("station_id must be non-empty")
    n = len(scan.points)
    if scan.intensity is not None and len(scan.intensity) != n:
        raise ValueError("intensity length mismatch")
    if scan.hit_patch is not None and len(scan.hit_patch) != n:
        raise ValueError("hit_patch length mismatch")
    if not isinstance(scan.sensor, SensorModel):
        raise ValueError("sensor must be a SensorModel")


def save_station(path: str | Path, scan: StationScan) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = dict(
        points=scan.points,
        origin=scan.origin,
        quat_wxyz=scan.quat_wxyz,
        station_id=np.str_(scan.station_id),
        timestamp=np.float64(scan.timestamp),
        frame=np.str_(scan.frame),
        sensor=np.str_(scan.sensor.to_json()),
    )
    if scan.intensity is not None:
        payload["intensity"] = scan.intensity
    if scan.hit_patch is not None:
        payload["hit_patch"] = scan.hit_patch
    np.savez_compressed(path, **payload)


def load_station(path: str | Path) -> StationScan:
    with np.load(str(path), allow_pickle=False) as z:
        keys = set(z.files)
        required = {"points", "origin", "quat_wxyz", "station_id", "timestamp", "frame", "sensor"}
        missing = required - keys
        if missing:
            raise ValueError(f"{path}: missing schema keys {sorted(missing)}")
        unknown = keys - required - {"intensity", "hit_patch"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)} (schema is closed)")
        return StationScan(
            points=z["points"],
            origin=z["origin"],
            quat_wxyz=z["quat_wxyz"],
            station_id=str(z["station_id"]),
            timestamp=float(z["timestamp"]),
            frame=str(z["frame"]),
            sensor=SensorModel.from_json(str(z["sensor"])),
            intensity=z["intensity"] if "intensity" in keys else None,
            hit_patch=z["hit_patch"] if "hit_patch" in keys else None,
        )
