"""公式(43) 2.5D 可通行图与安全膨胀 + 栅格 A*。

trav(g) = I[slope≤s_max]·I[clear≥h_robot]·I[g∉Z_safe⊕r_infl]

本实现面向程序化场景: 地面取 z=0 平面(slope 恒 0), 障碍由构件足印
(xy 包围盒/多边形)给出, 安全禁入区为带电构件足印外扩 d_safe(class)。
r_infl = r_robot + d_safe(class), 形态学膨胀在栅格上以欧氏距离变换实现。
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage

R_ROBOT_DEFAULT = 0.6
H_ROBOT_DEFAULT = 1.8
RES_DEFAULT = 0.25


class TravGrid:
    def __init__(self, bounds_xy: tuple[float, float, float, float],
                 res: float = RES_DEFAULT, r_robot: float = R_ROBOT_DEFAULT):
        self.xmin, self.ymin, self.xmax, self.ymax = map(float, bounds_xy)
        self.res = float(res)
        self.r_robot = float(r_robot)
        self.nx = max(int(np.ceil((self.xmax - self.xmin) / self.res)), 1)
        self.ny = max(int(np.ceil((self.ymax - self.ymin) / self.res)), 1)
        # 每格分别记录: 障碍占据(物理不可达) 与 禁入区(安全约束)
        self._obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        self._forbid = np.zeros((self.nx, self.ny), dtype=bool)
        self._free: np.ndarray | None = None

    # ---- 构建 ----
    def _fill_box(self, grid: np.ndarray, xy_min, xy_max, margin: float) -> None:
        i0 = int(np.floor((xy_min[0] - margin - self.xmin) / self.res))
        i1 = int(np.ceil((xy_max[0] + margin - self.xmin) / self.res))
        j0 = int(np.floor((xy_min[1] - margin - self.ymin) / self.res))
        j1 = int(np.ceil((xy_max[1] + margin - self.ymin) / self.res))
        grid[max(i0, 0):min(i1, self.nx), max(j0, 0):min(j1, self.ny)] = True

    def add_obstacle_box(self, xy_min, xy_max, clearance_z: float = 0.0,
                         h_robot: float = H_ROBOT_DEFAULT) -> None:
        """构件足印; clearance_z 为构件底面离地净空(≥h_robot 时可从下方通过, 如管母)。"""
        if clearance_z >= h_robot:
            return
        self._fill_box(self._obstacle, xy_min, xy_max, 0.0)
        self._free = None

    def add_safety_zone(self, xy_min, xy_max, d_safe: float) -> None:
        """安全禁入区 = 足印外扩 d_safe(class)(公式43 的 Z_safe⊕r_infl 的 d_safe 部分)。"""
        self._fill_box(self._forbid, xy_min, xy_max, d_safe)
        self._free = None

    def _compute_free(self) -> np.ndarray:
        if self._free is None:
            blocked = self._obstacle | self._forbid
            # 机器人半径膨胀(欧氏距离变换)
            if blocked.any():
                dist = ndimage.distance_transform_edt(~blocked, sampling=self.res)
                blocked = dist <= self.r_robot
            self._free = ~blocked
        return self._free

    # ---- 查询 ----
    def to_ij(self, xy) -> tuple[int, int]:
        i = int((float(xy[0]) - self.xmin) / self.res)
        j = int((float(xy[1]) - self.ymin) / self.res)
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def to_xy(self, ij) -> tuple[float, float]:
        return (self.xmin + (ij[0] + 0.5) * self.res, self.ymin + (ij[1] + 0.5) * self.res)

    def is_free(self, xy) -> bool:
        free = self._compute_free()
        return bool(free[self.to_ij(xy)])

    def project_to_free(self, xy, max_dist: float = 2.0) -> tuple[float, float] | None:
        """候选点投影到最近可通行单元; 超过 max_dist 返回 None(公式43 末段)。"""
        free = self._compute_free()
        if free[self.to_ij(xy)]:
            return (float(xy[0]), float(xy[1]))
        ii, jj = np.where(free)
        if len(ii) == 0:
            return None
        cx = self.xmin + (ii + 0.5) * self.res
        cy = self.ymin + (jj + 0.5) * self.res
        d2 = (cx - float(xy[0])) ** 2 + (cy - float(xy[1])) ** 2
        k = int(np.argmin(d2))
        if np.sqrt(d2[k]) > max_dist:
            return None
        return (float(cx[k]), float(cy[k]))

    # ---- A* ----
    def astar(self, a_xy, b_xy) -> tuple[float, list[tuple[float, float]]]:
        """8 邻域栅格 A*; 返回 (路径长米, 途径点)。不可达返回 (inf, [])。"""
        free = self._compute_free()
        start, goal = self.to_ij(a_xy), self.to_ij(b_xy)
        if not free[start] or not free[goal]:
            return float("inf"), []
        if start == goal:
            return 0.0, [self.to_xy(start)]
        sq2 = np.sqrt(2.0)
        nbrs = [(-1, -1, sq2), (-1, 0, 1.0), (-1, 1, sq2), (0, -1, 1.0),
                (0, 1, 1.0), (1, -1, sq2), (1, 0, 1.0), (1, 1, sq2)]

        def h(ij):
            dx, dy = abs(ij[0] - goal[0]), abs(ij[1] - goal[1])
            return (max(dx, dy) + (sq2 - 1.0) * min(dx, dy))

        g = {start: 0.0}
        came: dict = {}
        pq = [(h(start), start)]
        closed = set()
        while pq:
            _, cur = heapq.heappop(pq)
            if cur in closed:
                continue
            if cur == goal:
                path = [cur]
                while path[-1] in came:
                    path.append(came[path[-1]])
                return g[cur] * self.res, [self.to_xy(ij) for ij in reversed(path)]
            closed.add(cur)
            for di, dj, w in nbrs:
                ni, nj = cur[0] + di, cur[1] + dj
                if not (0 <= ni < self.nx and 0 <= nj < self.ny) or not free[ni, nj]:
                    continue
                # 对角穿越需两正交邻格均可行, 防止贴角穿越障碍
                if di and dj and not (free[cur[0] + di, cur[1]] and free[cur[0], cur[1] + dj]):
                    continue
                ng = g[cur] + w
                nb = (ni, nj)
                if ng < g.get(nb, float("inf")):
                    g[nb] = ng
                    came[nb] = cur
                    heapq.heappush(pq, (ng + h(nb), nb))
        return float("inf"), []

    def distance_matrix(self, points_xy: list[tuple[float, float]]) -> np.ndarray:
        n = len(points_xy)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d, _ = self.astar(points_xy[i], points_xy[j])
                D[i, j] = D[j, i] = d
        return D
