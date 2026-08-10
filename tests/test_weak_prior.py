"""目标参考模型 M_ref 与规划环境模型 M_plan 的解耦。

在此之前二者是同一份几何：可通行空间由完整参考模型的全部构件栅格化得到，可见性
求交也对该完整几何进行——规划器在执行第一站之前就知道现场每个障碍在哪。
"""

from __future__ import annotations

import numpy as np
import pytest

from patent_gap.mapping.planning_env import FREE, OCCUPIED, UNKNOWN, PlanningEnvGrid
from patent_gap.sensors.model import SensorModel
import patent_gap.simulation.closed_loop_v2 as cl
from patent_gap.simulation.scene_gen import generate_scene

SENSOR = {"dtheta_deg": 0.8, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005}


def _scene():
    return generate_scene(seed=0, family="S", density="low")


def test_full_prior_is_bit_identical_to_previous_behaviour():
    """env_prior_frac=1.0 且不启用观测建图时，必须与既有实现完全相同。

    这是"现有实验全部保留、不得作废"的前提：P100 就是既有条件。
    """
    scene = _scene()
    assert (scene.prior_tri_mask(1.0) == scene.bim_tri_mask()).all()
    w = cl.SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8)
    assert w.plan_oracle is w.oracle      # 无偏离时不额外建 BVH
    assert w.env is None                  # 不启用观测建图
    assert cl.planning_map(w) is w.grid   # 可通行图仍来自完整模型


def test_environment_geometry_is_removed_by_object_not_by_triangle():
    """按对象整取整舍，不得随机删三角面留下半张皮。"""
    scene = _scene()
    envs = [c for c in scene.components if not c.is_target and c.cls != "ground"]
    assert envs, "该场景没有非目标环境构件，无法调节环境先验"
    for frac in (0.0, 0.5, 1.0):
        m = scene.prior_tri_mask(frac, seed=0)
        for c in scene.components:
            seg = m[c.tri_start:c.tri_end]
            assert seg.all() or (~seg).all(), f"构件 {c.comp_id} 被部分保留"
        # 目标构件与地面恒含
        for c in scene.components:
            if c.is_target or c.cls == "ground":
                assert m[c.tri_start:c.tri_end].all()


def test_weak_prior_planner_cannot_see_unobserved_environment():
    """P0 下规划用遮挡模型不得包含未观测的非目标环境几何。"""
    scene = _scene()
    w = cl.SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8,
                          env_prior_frac=0.0, observed_env=True)
    assert w.plan_oracle is not w.oracle
    n_plan = w.plan_oracle.tri_to_patch.shape[0]
    n_full = w.oracle.tri_to_patch.shape[0]
    assert n_plan < n_full, "M_plan 仍含全部几何"
    for c in scene.components:
        if not c.is_target and c.cls != "ground":
            assert not w.ref_tris[c.tri_start], f"{c.cls} 不应进入 M_plan⁰"


def test_unknown_is_not_free_by_default():
    """未知区域不得默认当作可通行——这是弱先验下最容易偷跑的一处。"""
    g = PlanningEnvGrid((0, 0, 10, 10), res=0.5, r_robot=0.0)
    assert (g.state == UNKNOWN).all()
    assert not g._compute_free().any(), "初始全未知时不得有任何可通行栅格"
    g.seed_free((5, 5), radius=1.0)
    assert g._compute_free().any()
    # 显式放开时才退回旧行为
    g2 = PlanningEnvGrid((0, 0, 10, 10), res=0.5, r_robot=0.0, unknown_is_free=True)
    assert g2._compute_free().all()


def test_observation_marks_occupied_and_carves_free():
    """回波点所在处为占据，站位到回波点之间为自由；占据不被射线抹回自由。"""
    g = PlanningEnvGrid((0, 0, 10, 10), res=0.25, r_robot=0.0)
    origin = np.array([1.0, 5.0, 1.0])
    pts = np.array([[8.0, 5.0, 1.0], [8.0, 5.2, 1.0]])
    n_occ, n_free = g.integrate_scan(origin, pts)
    assert n_occ > 0 and n_free > 0
    assert g.state[g.to_ij((8.0, 5.0))] == OCCUPIED
    assert g.state[g.to_ij((4.0, 5.0))] == FREE      # 途中
    assert g.state[g.to_ij((2.0, 9.0))] == UNKNOWN   # 射线未及
    # 再打一条穿过已知占据格的射线，占据不得被抹掉
    g.integrate_scan(origin, np.array([[9.5, 5.0, 1.0]]))
    assert g.state[g.to_ij((8.0, 5.0))] == OCCUPIED


def test_new_observation_grows_the_planning_map():
    """闭环二：新增观测必须实际扩大 M_plan 的已确认区域。"""
    scene = _scene()
    w = cl.SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8,
                          env_prior_frac=0.0, observed_env=True)
    before = w.env.known_ratio()
    assert before == 0.0
    init = cl.default_init_stations(w, n=3)
    obs = cl.ObsState(world=w)
    for k, o in enumerate(init):
        sc = obs.add_station(o, f"i{k}", seed=k)
        w.env.seed_free(o[:2], radius=1.0)
        w.env.integrate_scan(o, sc.points)
    assert w.env.known_ratio() > 0.5


def test_discovered_geometry_never_becomes_a_scan_target():
    """在线发现的占据几何只作遮挡体，不得成为新的待扫目标。"""
    from patent_gap.occlusion.discovery import UnmodeledOccluders

    scene = _scene()
    w = cl.SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8,
                          env_prior_frac=0.0, observed_env=True)
    disc = UnmodeledOccluders(scene.vertices, scene.triangles[w.ref_tris],
                              w.tri_to_patch[w.ref_tris])
    obs = cl.ObsState(world=w)
    for k, o in enumerate(cl.default_init_stations(w, n=3)):
        obs.add_station(o, f"i{k}", seed=k)
        disc.update(obs.scans[-1].points)
    assert disc.n_cells() > 0, "P0 下应当发现未被参考模型描述的环境几何"
    orc = disc.build_oracle()
    n_base = int(w.ref_tris.sum())
    assert (orc.tri_to_patch[n_base:] == -1).all()


def test_planner_traversability_excludes_geometry_absent_from_the_model():
    """规划器的可通行图不得含 in_bim=False 的竣工态临时占位物。

    此前 build_trav_grid 遍历全部 components 而不看 in_bim，于是规划器绕开了
    一批"设计模型里根本查不到"的施工车辆——它不可能知道它们在哪。n_temp=0 时
    该泄漏不激活，既有 E2/E5/E6 因此逐比特不受影响。
    """
    sensor = SensorModel.from_config(SENSOR)
    w0 = cl.SimWorld.build(generate_scene(seed=0, family="S", density="low"),
                           sensor, sim_dtheta_deg=0.8)
    assert w0.plan_grid is w0.grid, "无临时占位物时不得多建一张图"

    scene = generate_scene(seed=0, family="S", density="low", n_temp=6)
    w = cl.SimWorld.build(scene, sensor, sim_dtheta_deg=0.8)
    assert w.plan_grid is not w.grid
    free_plan = cl.planning_map(w)._compute_free()
    free_real = w.grid._compute_free()
    assert free_plan.sum() > free_real.sum(), "规划器反而比实景知道得更多"
    # 临时占位物的中心在实景中不可通行，在规划图中却应当"看着能走"
    temp = [c for c in scene.components if not c.in_bim]
    assert temp
    hit = 0
    for c in temp:
        xy = ((c.bbox_min[0] + c.bbox_max[0]) / 2, (c.bbox_min[1] + c.bbox_max[1]) / 2)
        if not w.grid.is_free(xy) and cl.planning_map(w).is_free(xy):
            hit += 1
    assert hit > 0, "没有任何临时占位物体现出'规划器看不见'"


def test_vis_audit_is_off_by_default_and_records_when_on():
    """审计字段默认不出现——既有结果集必须逐比特可复现。"""
    assert cl.EpisodeConfig().vis_audit is False
    scene = generate_scene(seed=0, family="S", density="low", n_temp=6)
    w = cl.SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8)
    init = cl.default_init_stations(w, n=2)
    cfg = cl.EpisodeConfig(stations_max=1, rounds_max=1, seed=0,
                           method="B10_full", vis_audit=True)
    r = cl.run_episode(w, init, cfg)
    last = r["history"][-1]
    for k in ("vis_mae", "vis_over", "invalid_view_frac", "mplan_known_ratio"):
        assert k in last, k
    assert last["vis_over"] >= 0.0
    # 单向高估不得超过双向总误差
    assert last["vis_over"] <= last["vis_mae"] + 1e-12


def test_frontier_is_boundary_of_free_and_unknown():
    g = PlanningEnvGrid((0, 0, 10, 10), res=0.5, r_robot=0.0)
    assert len(g.frontier_cells()) == 0      # 全未知时没有 frontier
    g.seed_free((5, 5), radius=1.0)
    f = g.frontier_cells()
    assert len(f) > 0
    for i, j in f:
        assert g.state[i, j] == FREE
