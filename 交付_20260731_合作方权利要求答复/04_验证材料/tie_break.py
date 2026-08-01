"""说明书规定的"最近三角面唯一归属规则"的实现（2026-07-31 合作方批注 疑问2）。

背景：点到不同三角面的最近距离可能严格相等——在两表面相交且夹角小于180度的凸棱
外侧，以该棱为脊的三维楔形区域内处处如此，且该棱通常正是两个表面分块的公共边界。
底层最近点查询库在并列时的取舍随三角面存储顺序而变，故必须由本模块按说明书规定的
确定性规则裁决，使归属唯一、可复现、且不随三角面存储顺序改变。

规则（逐级适用直至唯一）：
  第一级  取最近距离最小者（并列判定含数值容差 tol）；
  第二级  取“由该三角面上最近表面点指向该点的方向”与该三角面法向所成夹角的
          余弦绝对值最大者（取绝对值使判定不受顶点绕序即法向朝向影响）；
          当最近距离为零、该方向无定义时，视为本级并列，直接进入第三级；
  第三级  取“三个顶点坐标按字典序排序后所得序列”在字典序上最小者。

第三级采用三角面自身的几何量而非其存储索引，因此整条规则的结果不依赖三角面在
文件中的存储顺序——实测（见 tests/test_tie_break.py）：若第三级改用“索引最小者”，
同一等距点在 24 种三角面排列下会被判给不同的表面分块；改用几何字典序后，24 种
排列全部给出同一归属。仅当两个候选三角面的顶点集合完全相同（几何重合）时，
第三级无法区分，此时退回按索引择一；该情形下两者几何等同，归属差异无实际影响。
"""

from __future__ import annotations

import numpy as np

# 距离并列的数值容差（米）。工程实现为浮点运算，理论等距点在不同精度下可能
# 得到 0.050000000 与 0.050000003 之类的差异，若按严格相等判定，确定性规则
# 根本不会被触发。取值依据：在真实 IFC 网格（1,197,750 个三角面）与 20,000 点
# 的实测中，同一点由单精度后端与双精度闭式解得到的最近距离最大相差 8.3e-7 m，
# 故容差取 1e-6 m —— 既能覆盖该量级的浮点差异，又远小于距离阈值 tau_d（0.05 m），
# 不会把几何上真正不同的三角面误判为并列。
DEFAULT_TIE_TOL = 1e-6


def closest_point_on_triangle(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """点 p 到单个三角形 tri(3x3) 的最近表面点（闭式解，含退化面处理）。"""
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    # 退化面（三顶点共线或重合，面积为零）：所在平面无定义，退化到三条闭线段
    n = np.cross(ab, ac)
    if float(n @ n) <= 1e-24 * max(float(ab @ ab) * float(ac @ ac), 1e-30):
        best, best_d = a, np.inf
        for u, v in ((a, b), (b, c), (a, c)):
            q = _closest_on_segment(p, u, v)
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best, best_d = q, d
        return best
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    den = 1.0 / (va + vb + vc)
    return a + ab * (vb * den) + ac * (vc * den)


def _closest_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 0.0:
        return a
    t = min(max(float((p - a) @ ab) / denom, 0.0), 1.0)
    return a + t * ab


def _unit_normal(tri: np.ndarray) -> np.ndarray:
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    ln = float(np.linalg.norm(n))
    return n / ln if ln > 0 else np.array([0.0, 0.0, 1.0])


def resolve_tie(p: np.ndarray, vertices: np.ndarray, faces: np.ndarray,
                candidates: np.ndarray, tol: float = DEFAULT_TIE_TOL) -> int:
    """在并列的候选三角面中按说明书规则择一，返回三角面索引。

    candidates 为取得（容差内）相同最近距离的三角面索引数组。
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) == 1:
        return int(candidates[0])

    # 第二级：|cos(方向, 法向)| 最大者；方向由最近表面点指向 p
    scores = np.full(len(candidates), -1.0)
    for k, t in enumerate(candidates):
        tri = vertices[faces[t]]
        q = closest_point_on_triangle(p, tri)
        d = p - q
        ln = float(np.linalg.norm(d))
        if ln <= 0.0:
            continue                      # 方向无定义 → 本级并列，留 -1
        scores[k] = abs(float((d / ln) @ _unit_normal(tri)))
    best = scores.max()
    if best >= 0.0:
        keep = candidates[scores >= best - tol]
    else:
        keep = candidates                 # 全部方向无定义

    # 第三级：几何字典序最小者（与存储顺序无关）
    keys = [(_geometric_key(vertices[faces[t]]), int(t)) for t in keep]
    keys.sort()
    return keys[0][1]


def _geometric_key(tri: np.ndarray) -> tuple:
    """三角面的几何字典序键：三个顶点各自坐标取整到容差网格后排序。

    仅由三角面自身的几何位置决定，不含任何存储顺序信息，故据此裁决的结果
    在三角面重新排列后保持不变。取整是为避免浮点末位差异影响排序。
    """
    q = np.round(np.asarray(tri, dtype=np.float64), 9)
    return tuple(sorted(tuple(v) for v in q))


class _BucketedTriangleIndex:
    """按外接球半径分桶的三角面空间索引，用于精确的“距离不超过 D”范围查询。

    三角面 t 能取得 d(p,t) ≤ D，必有 ‖p − c_t‖ ≤ D + r_t（c_t、r_t 为其外接球心与
    半径）。若对全体三角面统一用 r_max 作界，在尺寸悬殊的建筑网格上会退化为近乎
    全表扫描（本项目真实网格 r 的中位数 0.0055 m，最大 16.17 m，相差三个数量级）。
    故按 r 分桶，每桶用该桶的 r 上界作查询半径，既保持精确又把候选数压到局部量级。
    """

    def __init__(self, tri_pts: np.ndarray, n_buckets: int = 12):
        from scipy.spatial import cKDTree

        centers = tri_pts.mean(axis=1)
        radii = np.linalg.norm(tri_pts - centers[:, None, :], axis=2).max(axis=1)
        self.buckets: list[tuple] = []
        if len(radii) == 0:
            return
        r_pos = radii[radii > 0]
        lo = float(r_pos.min()) if len(r_pos) else 1e-9
        hi = float(radii.max()) if radii.max() > 0 else 1e-9
        # 边界必须至少有两个：当全部三角面尺寸相同时 geomspace 会塌缩成单值，
        # 若不显式补上下界，会一个桶都建不出来，规则将永远不被触发。
        edges = np.unique(np.concatenate((
            [0.0],
            np.geomspace(max(lo, 1e-12), max(hi, 1e-12), n_buckets + 1),
            [max(hi, 1e-12) * (1 + 1e-9)],
        )))
        for k in range(len(edges) - 1):
            sel = np.where((radii >= edges[k]) & (radii < edges[k + 1]))[0]
            if len(sel):
                self.buckets.append((cKDTree(centers[sel]), sel, float(radii[sel].max())))

    def query(self, p: np.ndarray, d: float) -> np.ndarray:
        """返回所有可能满足 d(p,t) ≤ d 的三角面索引（精确超集）。"""
        out: list[np.ndarray] = []
        for tree, sel, r_max in self.buckets:
            hit = tree.query_ball_point(p, d + r_max)
            if hit:
                out.append(sel[np.asarray(hit, dtype=np.int64)])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def nearest_triangle_deterministic(points: np.ndarray, vertices: np.ndarray,
                                   faces: np.ndarray, base_distances: np.ndarray,
                                   base_indices: np.ndarray,
                                   tol: float = DEFAULT_TIE_TOL,
                                   max_distance: float | None = None,
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """在底层最近点查询结果之上施加说明书规定的确定性规则。

    base_distances / base_indices 为任一最近点查询后端给出的最近距离与三角面索引；
    本函数在**全部**三角面中找出与最小距离在容差 tol 内并列者，再按规则择一。
    返回 (最近距离, 确定性裁决后的三角面索引)。

    max_distance 给定时（通常取距离阈值 tau_d），最近距离已超过 max_distance + tol
    的点一律跳过：这类点无论归属哪个三角面都判为未匹配，其归属不影响任何统计量。

    注意：候选集必须在全部三角面上取，不能只取与基准三角面相邻者——两片互不相邻
    的平行表面，其正中的点到两者距离相等，若只搜相邻面会漏判（见 tests/）。
    """
    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    out_idx = np.asarray(base_indices, dtype=np.int64).copy()
    dists = np.asarray(base_distances, dtype=np.float64).copy()

    tri_pts = vertices[faces]                                   # (T,3,3)
    index = _BucketedTriangleIndex(tri_pts)

    for i, p in enumerate(points):
        d0 = dists[i]
        t0 = int(out_idx[i])
        if not np.isfinite(d0) or t0 < 0 or t0 >= len(faces):
            continue
        if max_distance is not None and d0 > max_distance + tol:
            continue
        cand = index.query(p, d0 + tol)
        if len(cand) <= 1:
            continue
        dd = np.array([np.linalg.norm(p - closest_point_on_triangle(p, tri_pts[t]))
                       for t in cand])
        m = float(dd.min())
        tied = cand[dd <= m + tol]
        out_idx[i] = (resolve_tie(p, vertices, faces, tied, tol)
                      if len(tied) > 1 else int(tied[0]))
        dists[i] = m
    return dists, out_idx
