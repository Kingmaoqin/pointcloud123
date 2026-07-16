"""HELIOS++ 桥接与降级仿真器(1.2.1 / 3.7)。

HELIOS++ 未部署时(Gate 0 降级路径), FallbackSimulator 用 Open3D
RaycastingScene 做球面扫描: (Δθ,Δφ) 方位×俯仰网格射线 → 首命中点
+ N(0,σ_r) 距离噪声 → 可选公式(32)掠射角丢弃 → 统一 station NPZ。
与 HELIOS++ 输出 Schema 一致, 上层零改动。
"""

from __future__ import annotations

import shutil

import numpy as np

from ..io.station_npz import StationScan
from ..occlusion.raycast import OcclusionOracle
from ..sensors.model import SensorModel


def helios_available() -> bool:
    """【待核实】HELIOS++ 可执行文件是否在 PATH。当前环境未部署 → Fallback。"""
    return shutil.which("helios") is not None


class FallbackSimulator:
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray,
                 tri_to_patch: np.ndarray, sensor: SensorModel,
                 sim_dtheta_deg: float | None = None,
                 elev_range_deg: tuple[float, float] = (-35.0, 65.0),
                 use_eta_inc: bool = False):
        """sim_dtheta_deg: 仿真射线角步长(默认取传感器 Δθ; 可放粗以控制射线数,
        密度模型会用实际步长, 保持物理一致)。"""
        self.oracle = OcclusionOracle(vertices, triangles, tri_to_patch)
        self.sensor = sensor
        self.sim_dtheta = np.deg2rad(sim_dtheta_deg) if sim_dtheta_deg else sensor.dtheta
        self.elev_range = elev_range_deg
        self.use_eta_inc = use_eta_inc
        self._vertices = vertices
        self._triangles = triangles
        # 面法向用于入射角
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        self._face_normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def effective_sensor(self) -> SensorModel:
        """仿真实际使用的角步长(评价/密度预测须用它, 禁止硬编码)。"""
        s = self.sensor
        return SensorModel(r_min=s.r_min, r_opt=s.r_opt, r_max=s.r_max,
                           dtheta=self.sim_dtheta, dphi=self.sim_dtheta,
                           sigma_r=s.sigma_r, f_pulse=s.f_pulse)

    def scan(self, origin: np.ndarray, station_id: str, seed: int = 0) -> StationScan:
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        step = self.sim_dtheta
        az = np.arange(0.0, 2 * np.pi, step)
        el = np.arange(np.deg2rad(self.elev_range[0]), np.deg2rad(self.elev_range[1]), step)
        A, E = np.meshgrid(az, el, indexing="ij")
        dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)],
                        axis=-1).reshape(-1, 3)

        # 分批 cast, 控制内存
        rng = np.random.default_rng(seed)
        pts_out, patch_out = [], []
        s = self.sensor
        import open3d as o3d
        for chunk in np.array_split(dirs, max(len(dirs) // 500000, 1)):
            rays = np.hstack([np.tile(origin.astype(np.float32), (len(chunk), 1)),
                              chunk.astype(np.float32)])
            ans = self.oracle.scene.cast_rays(o3d.core.Tensor(rays))
            t = ans["t_hit"].numpy().astype(np.float64)
            prim = ans["primitive_ids"].numpy().astype(np.int64)
            ok = np.isfinite(t) & (t >= s.r_min) & (t <= s.r_max)
            if self.use_eta_inc:
                from ..density.expected import ALPHA_MAX_DEG
                fn = self._face_normals[np.clip(prim, 0, len(self._face_normals) - 1)]
                cos_inc = np.abs((fn * chunk).sum(axis=1))
                keep_p = np.clip((cos_inc - np.cos(np.deg2rad(ALPHA_MAX_DEG)))
                                 / (1 - np.cos(np.deg2rad(ALPHA_MAX_DEG))), 0, 1)
                ok &= rng.random(len(t)) < np.where(keep_p > 0, 1.0, 0.0)
            t_noisy = t[ok] + rng.normal(0.0, s.sigma_r, size=int(ok.sum()))
            pts_out.append(origin[None, :] + chunk[ok] * t_noisy[:, None])
            hp = np.full(len(prim), -1, dtype=np.int64)
            valid = prim >= 0
            hp[valid] = self.oracle.tri_to_patch[np.minimum(prim[valid],
                                                            len(self.oracle.tri_to_patch) - 1)]
            patch_out.append(hp[ok])

        return StationScan(
            points=np.vstack(pts_out) if pts_out else np.zeros((0, 3)),
            origin=origin,
            quat_wxyz=np.array([1.0, 0, 0, 0]),
            station_id=station_id,
            sensor=self.effective_sensor(),
            frame="global",
            hit_patch=np.concatenate(patch_out) if patch_out else np.zeros(0, dtype=np.int64),
        )
