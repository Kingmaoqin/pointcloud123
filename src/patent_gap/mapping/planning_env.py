"""规划环境模型 M_plan：由实测观测建立、逐轮更新的现场环境表示。

与目标参考模型 M_ref 的分工：

    M_ref  回答"我要扫什么"——目标构件、目标表面及其工程属性。
           不要求完整、准确地描述现场全部障碍物与非目标环境几何。
    M_plan 回答"我现在认为现场哪里被占据、哪些视线会被挡住"——
           由当前已获得的观测（加可选的可信安全先验）形成，随扫描逐轮更新。

在此模块之前，二者是同一份几何：可通行空间由完整参考模型的全部构件栅格化得到，
可见性求交也对该完整几何进行。于是规划器在执行第一站之前就已知道现场每个障碍的
位置——这正是"环境先验过强"。

本模块只提供 M_plan 的**占据侧**（三态栅格与可通行性）。其**遮挡侧**（射线求交用
的三角网）沿用既有的 UnmodeledOccluders：把基准网格由"完整参考模型"改为"仅目标
表面"后，同一套判据的语义即由"模型外临时异物"上位为"参考模型未描述的环境几何"。

三态的含义与处置：

    OCCUPIED  已由观测确认被占据 —— 障碍
    FREE      已由观测确认为空 —— 可通行（还需按平台半径膨胀）
    UNKNOWN   没有证据 —— **不得默认当作可通行**

UNKNOWN 的处置由 `unknown_is_free` 显式控制，缺省为 False（保守）。把它设为 True
即退回"未知即可走"的旧行为，仅供对照实验使用。
"""

from __future__ import annotations

import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class PlanningEnvGrid:
    """观测驱动的三态平面栅格，兼作可通行图。

    与 TravGrid 的接口保持一致（to_ij/to_xy/is_free/project_to_free/astar/
    distance_field/distance_matrix），使 S3 与 S6 无需改动即可换用。
    """

    def __init__(self, bounds_xy, res: float = 0.25, r_robot: float = 0.6,
                 unknown_is_free: bool = False) -> None:
        from .traversability import TravGrid

        self.xmin, self.ymin, self.xmax, self.ymax = map(float, bounds_xy)
        self.res = float(res)
        self.r_robot = float(r_robot)
        self.unknown_is_free = bool(unknown_is_free)
        self.nx = max(int(np.ceil((self.xmax - self.xmin) / self.res)), 1)
        self.ny = max(int(np.ceil((self.ymax - self.ymin) / self.res)), 1)
        self.state = np.full((self.nx, self.ny), UNKNOWN, dtype=np.uint8)
        # 复用 TravGrid 的几何工具（膨胀、A*、距离场），只替换其占据来源
        self._tg = TravGrid((self.xmin, self.ymin, self.xmax, self.ymax),
                            res=self.res, r_robot=self.r_robot)
        self._cache_key: tuple | None = None
        self._cache_free: np.ndarray | None = None

    # ---------------------------------------------------------------- 建图
    def integrate_scan(self, origin, points: np.ndarray,
                       z_lo: float = 0.20, z_hi: float = 1.80) -> tuple[int, int]:
        """把一站观测并入 M_plan，返回本次新增的 (占据格数, 自由格数)。

        占据：落在移动平台身高带 [z_lo, z_hi] 内的回波点所在栅格。带外的回波
        （地面、高处管母）不构成平面通行障碍，不标记。
        自由：自站位到每个回波点之间的连线所经栅格——射线打到那里才停下，
        沿途必然是空的。这是唯一能**由观测证明**某处可通行的依据。
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        if not len(pts):
            return 0, 0
        band = (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
        n_occ0 = int((self.state == OCCUPIED).sum())
        n_free0 = int((self.state == FREE).sum())

        # 先标自由（沿途），再标占据（终点），使终点不被自身射线抹成 FREE
        self._carve_free(o, pts)
        ij = self._to_ij_array(pts[band])
        if len(ij):
            self.state[ij[:, 0], ij[:, 1]] = OCCUPIED

        self._cache_key = None
        return (int((self.state == OCCUPIED).sum()) - n_occ0,
                int((self.state == FREE).sum()) - n_free0)

    def seed_free(self, xy, radius: float = 1.0) -> None:
        """把站位自身及其邻域标为自由——平台确实站在那里，是最直接的观测证据。"""
        i0, j0 = self.to_ij(xy)
        r = int(np.ceil(radius / self.res))
        ii, jj = np.meshgrid(np.arange(i0 - r, i0 + r + 1),
                             np.arange(j0 - r, j0 + r + 1), indexing="ij")
        ok = ((ii >= 0) & (ii < self.nx) & (jj >= 0) & (jj < self.ny)
              & ((ii - i0) ** 2 + (jj - j0) ** 2 <= r * r))
        sel = self.state[ii[ok], jj[ok]]
        self.state[ii[ok], jj[ok]] = np.where(sel == OCCUPIED, OCCUPIED, FREE)
        self._cache_key = None

    def add_safety_prior(self, xy_min, xy_max, d_safe: float) -> None:
        """可选的可信安全先验：带电体禁入区等**必须先验已知**的约束。

        这类信息不是"现场几何长什么样"，而是作业安全规程，与是否已经扫描无关，
        因此允许在无观测时即写入。它只增加禁入，不增加可通行。
        """
        self._tg.add_safety_zone(xy_min, xy_max, d_safe)
        self._tg._free = None
        self._cache_key = None

    # ---------------------------------------------------------------- 查询
    def _compute_free(self) -> np.ndarray:
        key = (int((self.state == OCCUPIED).sum()), int((self.state == FREE).sum()),
               bool(self.unknown_is_free))
        if self._cache_key == key and self._cache_free is not None:
            return self._cache_free
        passable = (self.state == FREE)
        if self.unknown_is_free:
            passable |= (self.state == UNKNOWN)
        self._tg._obstacle[:] = ~passable
        self._tg._free = None      # TravGrid 缓存 _free, 改 _obstacle 后须失效
        free = self._tg._compute_free()
        self._cache_key, self._cache_free = key, free
        return free

    def known_ratio(self) -> float:
        """已确认（非 UNKNOWN）栅格占全部栅格的比例，用于观察 M_plan 的收敛。"""
        return float((self.state != UNKNOWN).mean())

    def frontier_cells(self) -> np.ndarray:
        """frontier：已确认自由且四邻域中存在 UNKNOWN 的栅格。供探索基线使用。"""
        f = (self.state == FREE)
        u = (self.state == UNKNOWN)
        nb = np.zeros_like(u)
        nb[1:, :] |= u[:-1, :]; nb[:-1, :] |= u[1:, :]
        nb[:, 1:] |= u[:, :-1]; nb[:, :-1] |= u[:, 1:]
        return np.argwhere(f & nb)

    def expected_unknown_gain(self, xy, r_max: float, n_rays: int = 180) -> int:
        """从该位置架站, 预计能把多少个 UNKNOWN 栅格变为已确认。

        探索类方法的收益口径。二维等角射线自站位射出, 遇第一个 OCCUPIED 即止,
        沿途 UNKNOWN 栅格计入(去重)。这是 frontier / NBV 探索里通行的地图信息
        增益估计, 与本方法按目标缺口计价的收益是两个不同的量 —— 对照实验必须
        两个都报, 否则会把"地图覆盖得快"读成"任务缺口补得好"。
        """
        i0, j0 = self.to_ij(xy)
        n_step = max(int(np.ceil(r_max / self.res)), 1)
        ang = np.linspace(0.0, 2 * np.pi, n_rays, endpoint=False)
        t = (np.arange(1, n_step + 1) * self.res)[None, :]
        ii = np.rint(i0 + np.cos(ang)[:, None] * t / self.res).astype(np.int64)
        jj = np.rint(j0 + np.sin(ang)[:, None] * t / self.res).astype(np.int64)
        inside = (ii >= 0) & (ii < self.nx) & (jj >= 0) & (jj < self.ny)
        st = np.zeros(ii.shape, dtype=np.uint8)
        st[inside] = self.state[ii[inside], jj[inside]]
        # 射线在第一个占据格(或出界)处截断: 该步之前才算看得见
        blocked = (st == OCCUPIED) | (~inside)
        first = np.where(blocked.any(axis=1), blocked.argmax(axis=1), n_step)
        live = np.arange(n_step)[None, :] < first[:, None]
        sel = live & (st == UNKNOWN) & inside
        if not sel.any():
            return 0
        return int(len(np.unique(ii[sel] * self.ny + jj[sel])))

    # ---------------------------------------------------------------- 委托
    def to_ij(self, xy):
        return self._tg.to_ij(xy)

    def to_xy(self, ij):
        return self._tg.to_xy(ij)

    def is_free(self, xy) -> bool:
        i, j = self.to_ij(xy)
        return bool(self._compute_free()[i, j])

    def project_to_free(self, xy, max_dist: float = 2.0):
        self._compute_free()
        return self._tg.project_to_free(xy, max_dist=max_dist)

    def astar(self, a_xy, b_xy):
        self._compute_free()
        return self._tg.astar(a_xy, b_xy)

    def distance_field(self, src_xy) -> np.ndarray:
        self._compute_free()
        return self._tg.distance_field(src_xy)

    def distance_matrix(self, points_xy) -> np.ndarray:
        self._compute_free()
        return self._tg.distance_matrix(points_xy)

    # ---------------------------------------------------------------- 内部
    def _to_ij_array(self, pts: np.ndarray) -> np.ndarray:
        if not len(pts):
            return np.zeros((0, 2), dtype=np.int64)
        i = np.floor((pts[:, 0] - self.xmin) / self.res).astype(np.int64)
        j = np.floor((pts[:, 1] - self.ymin) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        return np.stack([i[ok], j[ok]], axis=1)

    def _carve_free(self, origin: np.ndarray, pts: np.ndarray,
                    step_frac: float = 0.5) -> None:
        """沿站位到各回波点的连线标记自由栅格（终点前一步为止）。

        按栅格分辨率的 step_frac 倍取样；采样点数按最长射线统一，短射线的多余
        采样被其自身终点截断，故不会越过终点。
        """
        d = pts[:, :2] - origin[None, :2]
        rng = np.linalg.norm(d, axis=1)
        keep = rng > self.res
        if not keep.any():
            return
        d, rng = d[keep], rng[keep]
        n = int(np.ceil(float(rng.max()) / (self.res * step_frac)))
        n = max(min(n, 4000), 1)
        # t 沿各自射线归一，末端留出一格不标，避免把终点障碍抹成自由
        back = np.clip(1.0 - self.res / np.maximum(rng, 1e-9), 0.0, 1.0)
        t = np.linspace(0.0, 1.0, n)[None, :] * back[:, None]
        xs = origin[0] + d[:, 0:1] * t
        ys = origin[1] + d[:, 1:2] * t
        i = np.floor((xs.ravel() - self.xmin) / self.res).astype(np.int64)
        j = np.floor((ys.ravel() - self.ymin) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        i, j = i[ok], j[ok]
        cur = self.state[i, j]
        # 已确认占据的栅格不被射线抹回自由（占据证据强于穿越证据）
        self.state[i, j] = np.where(cur == OCCUPIED, OCCUPIED, FREE)
