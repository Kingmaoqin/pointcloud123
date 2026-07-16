"""Gate 3 判据(3.5): 膨胀正确性; 随机场景 A* 路径逐段栅格碰撞=0; 不可达剔除。"""

import numpy as np

from patent_gap.mapping.traversability import TravGrid


def test_inflation_blocks_near_obstacle():
    g = TravGrid((0, 0, 20, 20), res=0.25, r_robot=0.6)
    g.add_obstacle_box((8, 8), (12, 12))
    assert not g.is_free((10, 10))          # 障碍内部
    assert not g.is_free((12.4, 10))        # 膨胀带(<0.6m)
    assert g.is_free((14.0, 10))            # 膨胀带外


def test_safety_zone_inflation():
    g = TravGrid((0, 0, 20, 20), res=0.25, r_robot=0.6)
    g.add_safety_zone((8, 8), (12, 12), d_safe=1.5)
    # 禁入区 = 足印外扩1.5 再加机器人半径0.6 → 13.5+0.6 之内不可行
    assert not g.is_free((13.9, 10))
    assert g.is_free((14.5, 10))


def test_overhead_busbar_passable():
    g = TravGrid((0, 0, 20, 20), res=0.25)
    g.add_obstacle_box((0, 9), (20, 11), clearance_z=5.0, h_robot=1.8)  # 高空管母
    assert g.is_free((10, 10))


def test_astar_zero_collision_random_scenes():
    """50 随机场景: A* 路径逐点必须落在可通行单元。"""
    rng = np.random.default_rng(42)
    for k in range(50):
        g = TravGrid((0, 0, 30, 30), res=0.5, r_robot=0.5)
        for _ in range(rng.integers(3, 8)):
            x0, y0 = rng.uniform(3, 24, 2)
            w, h = rng.uniform(1, 5, 2)
            g.add_obstacle_box((x0, y0), (x0 + w, y0 + h))
        a, b = (1.0, 1.0), (29.0, 29.0)
        if not (g.is_free(a) and g.is_free(b)):
            continue
        d, path = g.astar(a, b)
        if not np.isfinite(d):
            continue
        for xy in path:
            assert g.is_free(xy), f"scene {k}: path point {xy} in collision"
        assert d >= np.hypot(28, 28) - 1.5  # 不短于欧氏距离(容差=栅格离散)


def test_astar_unreachable():
    g = TravGrid((0, 0, 10, 10), res=0.25, r_robot=0.3)
    g.add_obstacle_box((4, 0), (6, 10))  # 全宽墙
    d, path = g.astar((1, 5), (9, 5))
    assert not np.isfinite(d) and path == []


def test_project_to_free():
    g = TravGrid((0, 0, 10, 10), res=0.25, r_robot=0.3)
    g.add_obstacle_box((4, 4), (6, 6))
    p = g.project_to_free((5, 5), max_dist=2.0)
    assert p is not None and g.is_free(p)
    assert g.project_to_free((5, 5), max_dist=0.1) is None
