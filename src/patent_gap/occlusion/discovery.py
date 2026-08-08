"""从已测点云中发现 BIM 未建模的遮挡物, 增量修正规划用遮挡模型。

规划阶段只有设计 BIM, 竣工现场却有车辆、料堆、脚手架这类 BIM 里查不到的东西。
公式(27)(28) 的 Vis 因此是会出错的预测: 规划器以为看得见, 扫描仪被挡住。

但这些遮挡物并非不可知 —— 它们本身会被扫到。凡是离所有 BIM 曲面都超过 ε 的
回波, 就是"测到了、但 BIM 里没有"的表面。把这些点体素化后并入规划几何, 后续
视点的可见性预测就不再重复同一个错误。这与公式(45) 闭环同源: 每执行一站, 既
更新缺口证据, 也更新遮挡模型。

与固定遮挡先验的区别在于, 这里不需要预先知道障碍在哪, 只需要知道"这块回波
无法归属于任何 BIM 构件" —— 真实流水线里正是点-构件关联(9)(10) 的残差。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from .raycast import OcclusionOracle

EPS_UNMODELED = 0.25    # m: 离 BIM 曲面多远算"未建模"(须大于配准误差+噪声)
VOXEL = 0.40            # m: 体素边长, 与 r_robot/2 同量级
Z_MIN = 0.35            # m: 低于此高度的回波按地面处理, 不作为遮挡物
MAX_POINTS = 200000     # 单站参与判定的点数上限(超出则均匀抽样)


class UnmodeledOccluders:
    """累积发现的未建模遮挡体素, 并据此重建规划用 OcclusionOracle。"""

    def __init__(self, bim_vertices: np.ndarray, bim_triangles: np.ndarray,
                 bim_tri_to_patch: np.ndarray, voxel: float = VOXEL,
                 eps: float = EPS_UNMODELED, z_min: float = Z_MIN,
                 seed: int = 0) -> None:
        self._verts = np.asarray(bim_vertices, dtype=np.float64)
        self._tris = np.asarray(bim_triangles, dtype=np.int64)
        self._t2p = np.asarray(bim_tri_to_patch, dtype=np.int64)
        self.voxel = float(voxel)
        self.eps = float(eps)
        self.z_min = float(z_min)
        self._rng = np.random.default_rng(seed)
        self.cells: set[tuple[int, int, int]] = set()
        # 只含 BIM 的距离场, 用于判定"这点属不属于某个 BIM 构件"
        self._dist_scene = o3d.t.geometry.RaycastingScene()
        self._dist_scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(self._verts.astype(np.float32)),
            o3d.core.Tensor(self._tris.astype(np.uint32))))

    def update(self, points: np.ndarray) -> int:
        """并入一站点云, 返回本站新增的体素数。"""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts[pts[:, 2] > self.z_min]
        if len(pts) == 0:
            return 0
        if len(pts) > MAX_POINTS:      # 抽样只为控开销, 体素化本身已高度冗余
            pts = pts[self._rng.choice(len(pts), MAX_POINTS, replace=False)]
        d = self._dist_scene.compute_distance(
            o3d.core.Tensor(pts.astype(np.float32))).numpy()
        far = pts[d > self.eps]
        if len(far) == 0:
            return 0
        keys = np.floor(far / self.voxel).astype(np.int64)
        before = len(self.cells)
        self.cells.update(map(tuple, keys))
        return len(self.cells) - before

    def n_cells(self) -> int:
        return len(self.cells)

    def build_oracle(self) -> OcclusionOracle:
        """BIM 几何 + 已发现遮挡体素(实心立方体)构成的规划用遮挡模型。

        发现出来的体素不携带 Patch 归属(tri_to_patch = −1): 它们是遮挡体, 不是
        待扫资产 —— 目标集始终由 BIM 构件清单定义, 不因现场堆了东西而改变。
        """
        if not self.cells:
            return OcclusionOracle(self._verts, self._tris, self._t2p)
        keys = np.array(sorted(self.cells), dtype=np.float64)
        centers = (keys + 0.5) * self.voxel
        cv, cf = _cube_soup(centers, self.voxel)
        verts = np.vstack([self._verts, cv])
        tris = np.vstack([self._tris, cf + len(self._verts)])
        t2p = np.concatenate([self._t2p, np.full(len(cf), -1, dtype=np.int64)])
        return OcclusionOracle(verts, tris, t2p)


_UNIT_CUBE_V = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64) - 0.5
_UNIT_CUBE_F = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)


def _cube_soup(centers: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """把体素中心展开成一堆独立立方体(不合并共面, BVH 不在乎)。"""
    n = len(centers)
    verts = (centers[:, None, :] + _UNIT_CUBE_V[None, :, :] * size).reshape(-1, 3)
    offs = (np.arange(n, dtype=np.int64) * 8)[:, None, None]
    faces = (_UNIT_CUBE_F[None, :, :] + offs).reshape(-1, 3)
    return verts, faces
