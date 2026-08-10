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


def test_every_target_surface_is_present_in_the_planning_model():
    """凡 M_ref 声明为目标的表面，M_plan⁰ 必须含其几何——各先验档都成立。

    否则规划器会被要求扫一个它拿不到几何的面，并在求交时以为自己能透视过去。
    先验强度只允许调节**非目标**环境几何。
    """
    scene = generate_scene(seed=0, family="S", density="low", n_temp=6)
    sensor = SensorModel.from_config(SENSOR)
    for frac in (1.0, 0.5, 0.0):
        w = cl.SimWorld.build(scene, sensor, sim_dtheta_deg=0.8,
                              env_prior_frac=frac, observed_env=True)
        is_target_tri = w.tri_to_patch >= 0
        assert is_target_tri.any()
        assert (w.ref_tris | ~is_target_tri).all(), \
            f"P{int(frac*100)}: 有目标分块的三角面不在 M_plan⁰ 中"


def test_non_target_environment_is_not_a_scan_target():
    """厂房/围墙/杂物遮挡视线但不是验收资产，不得进入目标分块表。"""
    from patent_gap.simulation.scene_gen import ENVIRONMENT_CLASSES

    scene = generate_scene(seed=0, family="S", density="low", n_temp=6)
    w = cl.SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8)
    env = [c for c in scene.components if c.cls in ENVIRONMENT_CLASSES]
    assert env
    for c in env:
        assert (w.tri_to_patch[c.tri_start:c.tri_end] == -1).all(), \
            f"{c.cls} 不应是待扫目标"
    # 但它们仍必须参与遮挡求交(评测用的完整几何里在)
    assert w.oracle.tri_to_patch.shape[0] == len(scene.triangles)


def test_s1_is_unaffected_by_how_much_environment_the_model_describes():
    """S1(目标分块与工程属性)只依赖目标本身，不随环境先验强度变化。"""
    scene = generate_scene(seed=0, family="S", density="low", n_temp=6)
    sensor = SensorModel.from_config(SENSOR)
    ref = None
    for frac in (1.0, 0.5, 0.0):
        w = cl.SimWorld.build(scene, sensor, sim_dtheta_deg=0.8,
                              env_prior_frac=frac, observed_env=True)
        cols = ["patch_id", "element_guid", "centroid_x", "centroid_y", "centroid_z",
                "area", "engineering_importance"]
        cur = w.patches[cols].to_numpy(dtype=object)
        if ref is None:
            ref = cur
        else:
            assert np.array_equal(ref, cur), f"P{int(frac*100)} 的 S1 结果变了"


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


def test_frontier_baseline_actually_moves_and_grows_the_map():
    """探索基线必须真能用掉预算 —— 一个走不动的基线不构成对照。

    frontier 栅格按平台半径膨胀后一个都不可通行(未知区按障碍处理, 而 frontier
    的定义就是紧贴未知区), 直接拿边界格当目标会让它 0 站收场。此处锁住"投影到
    邻近可站位置"这一修复。
    """
    scene = generate_scene(seed=0, family="S", density="low", n_temp=6)
    w = cl.SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8,
                          env_prior_frac=0.0, observed_env=True)
    init = cl.default_init_stations(w, n=3)
    cfg = cl.EpisodeConfig(stations_max=3, rounds_max=3, seed=0,
                           method="Bfrontier", vis_audit=True)
    r = cl.run_episode(w, init, cfg)
    assert r["final"]["n_stations"] >= 1, "探索基线一站也没走出去"
    kr = [h["mplan_known_ratio"] for h in r["history"] if "mplan_known_ratio" in h]
    assert kr and kr[-1] >= kr[0], "探索基线没有扩大地图已知区"


def test_frontier_baseline_reads_only_the_observed_map():
    """探索基线只许读当前 M_plan，不得碰完整模型栅格图。

    把 world.grid / world.plan_grid 换成一读就炸的哨兵：真值与初始站已在外面
    算好，闭环内若还有哪一处摸了完整模型，这里就会抛出来。
    """
    class Poison:
        def __getattr__(self, name):
            raise AssertionError(f"探索基线读了完整模型栅格图: .{name}")

    scene = generate_scene(seed=0, family="S", density="low", n_temp=6)
    sensor = SensorModel.from_config(SENSOR)
    w0 = cl.SimWorld.build(scene, sensor, sim_dtheta_deg=0.8)
    init = cl.default_init_stations(w0, n=3)
    probe = cl.ObsState(world=w0)
    for k, o in enumerate(init):
        probe.add_station(o, f"i{k}", seed=k)
    gt = cl.build_ground_truth(w0, list(probe.masks))
    del w0, probe

    w = cl.SimWorld.build(scene, sensor, sim_dtheta_deg=0.8,
                          env_prior_frac=0.0, observed_env=True)
    w.grid = Poison()
    w.plan_grid = Poison()
    cfg = cl.EpisodeConfig(stations_max=2, rounds_max=2, seed=0,
                           method="Bfrontier", vis_audit=True)
    r = cl.run_episode(w, init, cfg, gt=gt)
    assert r["final"]["n_stations"] >= 1


def test_frontier_baseline_requires_an_observation_driven_map():
    """完整先验下不存在未知区, 探索基线无定义 —— 必须报错而不是静默退化。"""
    scene = generate_scene(seed=0, family="S", density="low")
    w = cl.SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8)
    init = cl.default_init_stations(w, n=2)
    cfg = cl.EpisodeConfig(stations_max=1, rounds_max=1, seed=0, method="Bfrontier")
    with pytest.raises(ValueError, match="observed_env"):
        cl.run_episode(w, init, cfg)


def test_frontier_is_boundary_of_free_and_unknown():
    g = PlanningEnvGrid((0, 0, 10, 10), res=0.5, r_robot=0.0)
    assert len(g.frontier_cells()) == 0      # 全未知时没有 frontier
    g.seed_free((5, 5), radius=1.0)
    f = g.frontier_cells()
    assert len(f) > 0
    for i, j in f:
        assert g.state[i, j] == FREE
