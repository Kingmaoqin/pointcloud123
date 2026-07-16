"""PR4 验收: 退化专项(平行圆柱阵列 Degen>0.8) + 公式(37)(39)(40)行为。"""

import numpy as np

from patent_gap.registration.predictor import (
    degeneracy, expected_overlap, r_reg, regsup_gap_evidence, split_regions,
)


def _cylinder_points(axis="x", n_cyl=4, n_pts=400, radius=0.1):
    """平行圆柱阵列表面点+法向(轴向平移不可观)。"""
    rng = np.random.default_rng(0)
    pts, nrms = [], []
    for k in range(n_cyl):
        t = rng.uniform(0, 20, n_pts)          # 沿轴
        th = rng.uniform(0, 2 * np.pi, n_pts)  # 圆周
        y0 = k * 2.0
        if axis == "x":
            p = np.stack([t, y0 + radius * np.cos(th), radius * np.sin(th)], axis=1)
            n = np.stack([np.zeros(n_pts), np.cos(th), np.sin(th)], axis=1)
        pts.append(p)
        nrms.append(n)
    return np.vstack(pts), np.vstack(nrms)


def test_degeneracy_parallel_cylinders():
    pts, nrms = _cylinder_points()
    assert degeneracy(pts, nrms) > 0.8   # 沿轴不可观 → 高退化


def test_degeneracy_well_constrained():
    """三正交平面 → 六自由度可观, 退化度明显低于圆柱阵列。"""
    rng = np.random.default_rng(1)
    pts, nrms = [], []
    for ax in range(3):
        p = rng.uniform(-1, 1, size=(500, 3))
        p[:, ax] = 1.0
        n = np.zeros((500, 3))
        n[:, ax] = 1.0
        pts.append(p)
        nrms.append(n)
    d_plane = degeneracy(np.vstack(pts), np.vstack(nrms))
    pts_c, nrms_c = _cylinder_points()
    assert d_plane < degeneracy(pts_c, nrms_c) - 0.15


def test_degeneracy_too_few_points():
    assert degeneracy(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0


def test_split_and_overlap():
    gap = np.array([0.9, 0.2, 0.7, 0.1])
    cov = np.array([0.1, 0.9, 0.6, 0.9])
    fov = np.array([True, True, True, False])
    m_new, m_ovl = split_regions(gap, cov, fov)
    np.testing.assert_array_equal(m_new, [True, False, True, False])
    np.testing.assert_array_equal(m_ovl, [False, True, True, False])
    areas = np.ones(4)
    q = np.ones(4)
    ov = expected_overlap(areas, q, m_ovl, fov)
    assert abs(ov - 2.0 / 3.0) < 1e-5


def test_r_reg_gating():
    assert r_reg(overlap=0.30, degen=0.0) == 1.0
    assert r_reg(overlap=0.15, degen=0.0) == 0.5
    assert r_reg(overlap=0.60, degen=1.0) == 0.0
    assert abs(r_reg(overlap=0.30, degen=0.5) - 0.5) < 1e-12


def test_regsup_evidence():
    d = regsup_gap_evidence({"a": {1: 0.4}, "b": {1: 0.7, 2: 0.2}}, [1, 2, 3])
    assert abs(d[1] - 0.3) < 1e-12
    assert abs(d[2] - 0.8) < 1e-12
    assert d[3] == 1.0
