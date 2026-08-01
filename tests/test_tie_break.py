"""确定性归属规则的单测（2026-07-31 合作方批注 疑问2 的实现验证）。

验证四件事：
  1. 新规则已进入实际关联程序；
  2. 改变三角面存储顺序后，同一点的归属不变（旧行为会变）；
  3. 各表面分块的匹配点数之和不超过点云总点数（每点至多计入一个分块）；
  4. 闭式最近点在退化面、面内、边、顶点各分支上均正确。
"""

from itertools import permutations

import numpy as np
import pytest

from patent_gap.data.tie_break import (
    closest_point_on_triangle, nearest_triangle_deterministic, resolve_tie,
)

o3d = pytest.importorskip("open3d")


# 90°凸棱：竖直面(x=0)与水平面(z=0)交于 y 轴；双法向约束会把二者分为两个表面分块
RIDGE_V = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1],
                    [1, 0, 0], [1, 1, 0]], dtype=float)
RIDGE_F = [[0, 1, 2], [1, 3, 2],      # 表面分块 0（竖直面）
           [0, 4, 1], [4, 5, 1]]      # 表面分块 1（水平面）
PATCH_OF = {0: 0, 1: 0, 2: 1, 3: 1}


def _query(V, F, pts):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)),
                     o3d.core.Tensor(np.asarray(F).astype(np.uint32)))
    a = sc.compute_closest_points(o3d.core.Tensor(pts.astype(np.float32)))
    d = np.linalg.norm(pts - a["points"].numpy().astype(np.float64), axis=1)
    return d, a["primitive_ids"].numpy().astype(np.int64)


def _patch_under_all_orders(p, use_rule):
    """在 24 种三角面存储顺序下，返回该点被判给的表面分块集合。"""
    out = set()
    for perm in permutations(range(4)):
        F = np.array([RIDGE_F[i] for i in perm])
        d0, i0 = _query(RIDGE_V, F, p[None])
        if use_rule:
            _, i0 = nearest_triangle_deterministic(p[None], RIDGE_V, F, d0, i0)
        out.add(PATCH_OF[perm[int(i0[0])]])
    return out


@pytest.mark.parametrize("t", [0.05, 0.20, 0.50])
def test_tie_point_order_independent(t):
    """等距点（棱外45°方向）在任意存储顺序下归属唯一 —— 规则的核心目的。"""
    p = np.array([-t / np.sqrt(2), 0.5, -t / np.sqrt(2)])
    assert len(_patch_under_all_orders(p, use_rule=True)) == 1


def test_tie_point_unstable_without_rule():
    """对照：不加规则时，同一等距点在不同存储顺序下会被判给不同分块。"""
    p = np.array([-0.2 / np.sqrt(2), 0.5, -0.2 / np.sqrt(2)])
    assert len(_patch_under_all_orders(p, use_rule=False)) > 1


@pytest.mark.parametrize("p,expect_patch", [
    (np.array([-0.30, 0.5, -0.10]), 0),   # 更正对竖直面
    (np.array([-0.10, 0.5, -0.30]), 1),   # 更正对水平面
])
def test_non_tie_follows_frontality(p, expect_patch):
    """非等距点：归属由最近距离决定，且在任意存储顺序下稳定。"""
    got = _patch_under_all_orders(p, use_rule=True)
    assert got == {expect_patch}


def test_rule_is_wired_into_pipeline():
    """确定性规则确实接入了实际关联流程（而非只写在说明书里）。"""
    import inspect

    from patent_gap.data import pointcloud

    src = inspect.getsource(pointcloud.associate_points_to_ifc)
    assert "nearest_triangle_deterministic" in src
    assert "apply_tie_break" in inspect.signature(
        pointcloud.associate_points_to_ifc).parameters


def test_counts_never_exceed_total_points():
    """每点至多计入一个表面分块 → 各分块匹配点数之和 ≤ 点云总点数。"""
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.6, 0.6, size=(600, 3))
    pts[:, 1] = rng.uniform(0.1, 0.9, size=600)
    F = np.array(RIDGE_F)
    d0, i0 = _query(RIDGE_V, F, pts)
    d, idx = nearest_triangle_deterministic(pts, RIDGE_V, F, d0, i0)
    matched = d <= 0.05
    counts = {}
    for tri in idx[matched]:
        counts[PATCH_OF[int(tri)]] = counts.get(PATCH_OF[int(tri)], 0) + 1
    assert sum(counts.values()) == int(matched.sum()) <= len(pts)
    # 每个点只出现一次：索引数组长度与点数一致
    assert len(idx) == len(pts)


# 两片互不相邻的平行面：正中的点到两者等距。候选集若只取"与基准面共顶点者"
# 会完全漏掉对面那片 —— 该缺陷曾真实存在，此测试用于防止回归。
PARALLEL_V = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
                       [0, 0, 1], [2, 0, 1], [0, 2, 1], [2, 2, 1]], dtype=float)
PARALLEL_F = [[0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7]]
PARALLEL_PATCH = {0: 0, 1: 0, 2: 1, 3: 1}


@pytest.mark.parametrize("p", [
    np.array([1.0, 1.0, 0.5]),      # 正中，四面全并列
    np.array([0.6, 0.7, 0.5]),      # 正中平面上另一点
])
def test_non_adjacent_tie_order_independent(p):
    """不相邻的等距三角面也必须被裁决，且与存储顺序无关。"""
    got = set()
    for perm in permutations(range(4)):
        F = np.array([PARALLEL_F[i] for i in perm])
        d0, i0 = _query(PARALLEL_V, F, p[None])
        _, i1 = nearest_triangle_deterministic(p[None], PARALLEL_V, F, d0, i0)
        got.add(PARALLEL_PATCH[perm[int(i1[0])]])
    assert len(got) == 1, f"不相邻等距面归属不稳定: {got}"


def test_candidate_set_matches_brute_force():
    """裁决结果须与"在全部三角面上暴力枚举"一致（规范一致性）。"""
    from patent_gap.data.tie_break import closest_point_on_triangle as cpt

    rng = np.random.default_rng(11)
    pts = rng.uniform(-0.3, 2.3, size=(200, 3))
    F = np.array(PARALLEL_F)
    d0, i0 = _query(PARALLEL_V, F, pts)
    d1, i1 = nearest_triangle_deterministic(pts, PARALLEL_V, F, d0, i0)
    for k, p in enumerate(pts):
        dd = np.array([np.linalg.norm(p - cpt(p, PARALLEL_V[f])) for f in F])
        assert abs(d1[k] - dd.min()) < 1e-9          # 最小距离正确
        assert dd[i1[k]] <= dd.min() + 1e-6          # 选中者确在并列集内


def test_bucketing_handles_uniform_triangle_sizes():
    """全部三角面尺寸相同时分桶不得塌缩（曾导致一个桶都建不出、规则从不触发）。"""
    from patent_gap.data.tie_break import _BucketedTriangleIndex

    tri_pts = PARALLEL_V[np.array(PARALLEL_F)]
    idx = _BucketedTriangleIndex(tri_pts)
    assert len(idx.buckets) >= 1
    assert len(idx.query(np.array([1.0, 1.0, 0.5]), 0.6)) == 4


def test_resolve_tie_returns_single_candidate():
    F = np.array(RIDGE_F)
    assert resolve_tie(np.zeros(3), RIDGE_V, F, np.array([2])) == 2


@pytest.mark.parametrize("tri,p,expect", [
    # 退化面：两顶点重合 → 退化为线段，不应返回 NaN
    (np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], float), np.array([0.5, 1.0, 0.0]), 1.0),
    # 退化面：三点共线
    (np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], float), np.array([0, 0, 0], float), 0.0),
])
def test_degenerate_triangles(tri, p, expect):
    q = closest_point_on_triangle(p, tri)
    assert np.isfinite(q).all()
    assert abs(float(np.linalg.norm(p - q)) - expect) < 1e-9


def test_closest_point_branches():
    """面内 / 边 / 顶点三个分支。"""
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    np.testing.assert_allclose(
        closest_point_on_triangle(np.array([0.25, 0.25, 1.0]), tri), [0.25, 0.25, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        closest_point_on_triangle(np.array([0.5, -1.0, 0.0]), tri), [0.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        closest_point_on_triangle(np.array([-1.0, -1.0, 0.0]), tri), [0.0, 0.0, 0.0], atol=1e-12)
