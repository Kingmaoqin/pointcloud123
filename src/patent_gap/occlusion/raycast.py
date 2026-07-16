"""公式(27)(28)(29) first-hit 可见函数 / 视点-Patch 遮挡可见率 / 遮挡缺口证据。

基于 Open3D t.geometry.RaycastingScene(Embree BVH, CPU)。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from .sampling import PatchSampler


class OcclusionOracle:
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, tri_to_patch: np.ndarray):
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(triangles, dtype=np.uint32)),
        )
        self.geom_id = self.scene.add_triangles(mesh)
        self.tri_to_patch = np.asarray(tri_to_patch, dtype=np.int64)

    def first_hits(self, origin: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """从单一原点批量发射线; 返回 (t_hit, hit_patch)。未命中: t=inf, patch=-1。"""
        origin = np.asarray(origin, dtype=np.float32).reshape(3)
        directions = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        rays = np.hstack([np.tile(origin, (len(directions), 1)), directions])
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = ans["t_hit"].numpy().astype(np.float64)
        prim = ans["primitive_ids"].numpy().astype(np.int64)
        hit_patch = np.full(len(prim), -1, dtype=np.int64)
        ok = np.isfinite(t_hit) & (prim >= 0) & (prim < len(self.tri_to_patch))
        hit_patch[ok] = self.tri_to_patch[prim[ok]]
        return t_hit, hit_patch

    def visibility(self, origin: np.ndarray, samples: np.ndarray,
                   target_patch: int, eps: float = 0.01) -> float:
        """公式(27)(28): Vis_{i,v} = 命中自身 Patch 的采样点比例。"""
        samples = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
        if len(samples) == 0:
            return 0.0
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        d = samples - origin[None, :]
        dist = np.linalg.norm(d, axis=1)
        dist_safe = np.where(dist > 1e-9, dist, 1.0)
        t_hit, hit_patch = self.first_hits(origin, d / dist_safe[:, None])
        ok = (hit_patch == int(target_patch)) & (t_hit >= dist - eps)
        return float(ok.mean())

    def visibility_batch(self, origin: np.ndarray, sampler: PatchSampler,
                         patch_ids: list[int] | np.ndarray | None = None,
                         eps: float = 0.01) -> dict[int, float]:
        """一个候选站对全部目标 Patch 的采样点拼成单次 cast_rays(批量化)。"""
        if patch_ids is None:
            patch_ids = sampler.patch_ids()
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        all_samples, owners, counts = [], [], []
        for pid in patch_ids:
            s = sampler.samples(pid)
            counts.append(len(s))
            if len(s):
                all_samples.append(s)
                owners.append(np.full(len(s), int(pid), dtype=np.int64))
        if not all_samples:
            return {int(p): 0.0 for p in patch_ids}
        samples = np.vstack(all_samples)
        owner = np.concatenate(owners)
        d = samples - origin[None, :]
        dist = np.linalg.norm(d, axis=1)
        dist_safe = np.where(dist > 1e-9, dist, 1.0)
        t_hit, hit_patch = self.first_hits(origin, d / dist_safe[:, None])
        ok = (hit_patch == owner) & (t_hit >= dist - eps)
        out: dict[int, float] = {}
        pos = 0
        for pid, c in zip(patch_ids, counts):
            out[int(pid)] = float(ok[pos:pos + c].mean()) if c else 0.0
            pos += c
        return out


def occlusion_gap_evidence(vis_by_station: dict[str, dict[int, float]],
                           patch_ids: list[int]) -> dict[int, float]:
    """公式(29): D_i^occ = 1 - max_{v∈S_obs} Vis_{i,v}。

    vis_by_station: {station_id: {patch_id: Vis}}。空站集 → 全部 1.0。
    """
    out = {}
    for pid in patch_ids:
        best = 0.0
        for vis in vis_by_station.values():
            best = max(best, vis.get(int(pid), 0.0))
        out[int(pid)] = 1.0 - best
    return out
