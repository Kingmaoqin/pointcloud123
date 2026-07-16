"""公式(26) Patch 确定性表面采样。

M_i^s = clip(ceil(A_i/a_0), M_min, M_max); 面内分层重心坐标由 Halton 序列驱动,
种子 = hash(patch_id) → 同一 Patch 任意两次评估采样一致(可复现要求)。
"""

from __future__ import annotations

import hashlib

import numpy as np

A0_DEFAULT = 0.01   # m²/点
M_MIN_DEFAULT = 4
M_MAX_DEFAULT = 64


def _halton(index: np.ndarray, base: int) -> np.ndarray:
    """确定性 Halton 序列(向量化)。index 从 1 开始。"""
    result = np.zeros(len(index), dtype=np.float64)
    f = 1.0 / base
    i = index.astype(np.int64).copy()
    while np.any(i > 0):
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _stable_seed(patch_id: int) -> int:
    """跨进程稳定的种子(python hash() 不稳定, 用 sha1)。"""
    return int(hashlib.sha1(str(int(patch_id)).encode()).hexdigest()[:8], 16)


def sample_patch_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    patch_area: float,
    patch_id: int,
    a0: float = A0_DEFAULT,
    m_min: int = M_MIN_DEFAULT,
    m_max: int = M_MAX_DEFAULT,
) -> np.ndarray:
    """对单个 Patch(其三角面集合)做面积加权确定性采样, 返回 (M,3)。"""
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if len(triangles) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = float(tri_areas.sum())
    if total <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    m = int(np.clip(np.ceil(patch_area / a0), m_min, m_max))
    offset = _stable_seed(patch_id) % 100003  # Halton 起始偏移 = 确定性"种子"
    idx = np.arange(1, m + 1, dtype=np.int64) + offset
    u1 = _halton(idx, 2)
    u2 = _halton(idx, 3)

    # 面积加权分配三角形(按累计面积逆采样, 用第三条 Halton 维度)
    u3 = _halton(idx, 5)
    cdf = np.cumsum(tri_areas) / total
    tri_sel = np.searchsorted(cdf, u3, side="left").clip(0, len(triangles) - 1)

    # 分层重心坐标(折叠法保证均匀)
    su1 = np.sqrt(u1)
    b0 = 1.0 - su1
    b1 = u2 * su1
    b2 = 1.0 - b0 - b1
    pts = (b0[:, None] * v0[tri_sel] + b1[:, None] * v1[tri_sel] + b2[:, None] * v2[tri_sel])
    return pts


class PatchSampler:
    """按 patch → 三角面映射批量采样并缓存(确定性, 可反复复用)。"""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray,
                 tri_to_patch: np.ndarray, patch_areas: dict[int, float] | None = None,
                 a0: float = A0_DEFAULT, m_min: int = M_MIN_DEFAULT, m_max: int = M_MAX_DEFAULT):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.tri_to_patch = np.asarray(tri_to_patch, dtype=np.int64)
        self.a0, self.m_min, self.m_max = a0, m_min, m_max
        self._patch_tris: dict[int, np.ndarray] = {}
        for pid in np.unique(self.tri_to_patch):
            if pid >= 0:
                self._patch_tris[int(pid)] = self.triangles[self.tri_to_patch == pid]
        if patch_areas is None:
            patch_areas = {}
            for pid, tris in self._patch_tris.items():
                v0, v1, v2 = (self.vertices[tris[:, k]] for k in range(3))
                patch_areas[pid] = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
        self.patch_areas = patch_areas
        self._cache: dict[int, np.ndarray] = {}

    def patch_ids(self) -> list[int]:
        return sorted(self._patch_tris)

    def samples(self, patch_id: int) -> np.ndarray:
        pid = int(patch_id)
        if pid not in self._cache:
            tris = self._patch_tris.get(pid)
            if tris is None:
                self._cache[pid] = np.zeros((0, 3))
            else:
                self._cache[pid] = sample_patch_surface(
                    self.vertices, tris, self.patch_areas.get(pid, 0.0), pid,
                    self.a0, self.m_min, self.m_max)
        return self._cache[pid]
