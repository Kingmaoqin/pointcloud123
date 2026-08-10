"""未建模遮挡物在线发现(occlusion/discovery.py)。"""

from __future__ import annotations

import numpy as np
import pytest

from patent_gap.occlusion.discovery import UnmodeledOccluders
from patent_gap.sensors.model import SensorModel
from patent_gap.simulation.closed_loop_v2 import (
    EpisodeConfig, ObsState, SimWorld, build_ground_truth,
    default_init_stations, run_episode,
)
from patent_gap.simulation.scene_gen import generate_scene

SENSOR = {"dtheta_deg": 0.4, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005}


def _world(n_temp: int):
    scene = generate_scene(seed=0, family="S", density="low", n_temp=n_temp)
    return scene, SimWorld.build(scene, SensorModel.from_config(SENSOR),
                                 sim_dtheta_deg=0.4)


def test_nothing_discovered_when_bim_matches_reality():
    """BIM 与实景一致时不得"发现"任何东西 —— 否则是把噪声当障碍。"""
    scene, world = _world(n_temp=0)
    bim = scene.bim_tri_mask()
    disc = UnmodeledOccluders(scene.vertices, scene.triangles[bim],
                             world.tri_to_patch[bim])
    obs = ObsState(world=world)
    for k, o in enumerate(default_init_stations(world, n=2)):
        obs.add_station(o, f"i{k}", seed=k)
        disc.update(obs.scans[-1].points)
    assert disc.n_cells() == 0
    assert disc.build_oracle().tri_to_patch.shape[0] == int(bim.sum())


def test_discovers_the_occluders_bim_lacks():
    """临时占位物应被发现, 且发现的体素确实落在它们身上。"""
    scene, world = _world(n_temp=6)
    bim = scene.bim_tri_mask()
    disc = UnmodeledOccluders(scene.vertices, scene.triangles[bim],
                             world.tri_to_patch[bim])
    obs = ObsState(world=world)
    for k, o in enumerate(default_init_stations(world, n=3)):
        obs.add_station(o, f"i{k}", seed=k)
        disc.update(obs.scans[-1].points)
    assert disc.n_cells() > 0

    boxes = [(c.bbox_min, c.bbox_max) for c in scene.components if not c.in_bim]
    centers = (np.array(sorted(disc.cells), dtype=float) + 0.5) * disc.voxel
    inside = np.zeros(len(centers), dtype=bool)
    for lo, hi in boxes:                      # 体素边长带来的外扩容差
        inside |= ((centers >= lo - disc.voxel).all(axis=1)
                   & (centers <= hi + disc.voxel).all(axis=1))
    assert inside.mean() > 0.9, f"仅 {inside.mean():.0%} 的发现体素落在临时占位物上"

    # 发现体是遮挡体而非待扫资产: 不得引入新的 Patch 归属
    oracle = disc.build_oracle()
    assert oracle.tri_to_patch.shape[0] > int(bim.sum())
    assert (oracle.tri_to_patch[int(bim.sum()):] == -1).all()


@pytest.mark.parametrize("n_temp", [0])
def test_b11_equals_b10_without_divergence(n_temp):
    """无 BIM 失配时 B11 不得偏离 B10 —— 发现机制不能自己引入扰动。"""
    _, world = _world(n_temp)
    init = default_init_stations(world, n=3)
    probe = ObsState(world=world)
    for k, o in enumerate(init):
        probe.add_station(o, f"i{k}", seed=k)
    gt = build_ground_truth(world, list(probe.masks))

    def run(method):
        return run_episode(world, init, EpisodeConfig(
            stations_max=3, length_max_m=400.0, rounds_max=3,
            rho0=50.0, seed=0, method=method), gt=gt)["final"]

    a, b = run("B10_full"), run("B11_disc")
    for k in ("awc_gap_recovery", "asset_recovery", "crit_recall", "path_len_m"):
        assert a[k] == pytest.approx(b[k], abs=1e-12), k


def test_registration_realism_penalises_ill_conditioned_overlap():
    """配准误差须随重叠点数与退化度变化, 且默认关闭时不改变任何行为。"""
    import numpy as np

    from patent_gap.registration.realism import pose_error_sigma, register_station

    # 1/√K 收缩: 点数增至 100 倍, 高于系统性下限的部分应降至 1/10
    from patent_gap.registration.realism import SIGMA_POSE_FLOOR
    s_few = pose_error_sigma(50, 0.5, 0.005) - SIGMA_POSE_FLOOR
    s_many = pose_error_sigma(5000, 0.5, 0.005) - SIGMA_POSE_FLOOR
    assert s_few == pytest.approx(10 * s_many, rel=1e-6)

    # 条件数放大: 同样点数下退化度越高误差越大
    assert pose_error_sigma(500, 0.99, 0.005) > 5 * pose_error_sigma(500, 0.5, 0.005)

    # 重叠率低于 O_min 直接判失败, 与误差大小无关
    rng = np.random.default_rng(0)
    ok, _ = register_station(0.10, 10000, 0.0, 0.005, rng, o_min=0.30)
    assert not ok

    # 关闭时 run_episode 不得出现配准字段, 也不得改变结果
    _, world = _world(n_temp=0)
    init = default_init_stations(world, n=3)
    probe = ObsState(world=world)
    for k, o in enumerate(init):
        probe.add_station(o, f"i{k}", seed=k)
    gt = build_ground_truth(world, list(probe.masks))
    base = EpisodeConfig(stations_max=2, length_max_m=400.0, rounds_max=2,
                         rho0=50.0, seed=0, method="B10_full")
    off = run_episode(world, init, base, gt=gt)
    assert off["registration"] == []
    assert "n_reg_failed" not in off["final"]


def test_failed_registration_does_not_trap_the_planner():
    """配准失败的站必须进入去重集, 否则规划器在原地空转。

    失败站会从 obs.scans 弹出。若不另行记住, 规划器看到"这里没人去过"且 A*
    距离为 0(机器人就站在那儿), 成本必然最低 → 每轮重选同一点。实测该缺陷
    会让全部轮次停在一个位置、awc 归零。
    """
    import numpy as np

    import patent_gap.simulation.closed_loop_v2 as cl

    _, world = _world(n_temp=0)
    init = default_init_stations(world, n=3)
    probe = ObsState(world=world)
    for k, o in enumerate(init):
        probe.add_station(o, f"i{k}", seed=k)
    gt = build_ground_truth(world, list(probe.masks))

    picks: list[tuple] = []
    original = cl.ObsState.add_station
    original_reg = cl.register_station
    calls = {"n": 0}

    def spy(self, origin, station_id, seed=0):
        if station_id.startswith("B10"):
            picks.append(tuple(np.round(origin[:2], 2)))
        return original(self, origin, station_id, seed=seed)

    def fail_first(*a, **kw):
        # 直接注入一次失败, 不依赖某个场景恰好重叠不足 —— 要锁的是失败之后的
        # 控制流, 而重叠率会随采样密度等实现细节变动。
        calls["n"] += 1
        return (False, float("inf")) if calls["n"] == 1 else original_reg(*a, **kw)

    cl.ObsState.add_station = spy
    cl.register_station = fail_first
    try:
        res = cl.run_episode(world, init, cl.EpisodeConfig(
            stations_max=6, length_max_m=400.0, rounds_max=6, rho0=50.0,
            seed=0, method="B10_full", registration_realism=True), gt=gt)
    finally:
        cl.ObsState.add_station = original
        cl.register_station = original_reg

    assert res["registration"] and not res["registration"][0]["accepted"]
    # 注入的那一次必然失败; 之后是否还有**真实**的配准失败取决于站间重叠, 而
    # 重叠随目标集与采样密度变动 —— 目标集改为只含验收资产后实测会多出一次。
    # 本用例要锁的是失败之后的控制流(不得原地重选、任务不得因此归零), 不是失败
    # 次数, 后者不该被写死。
    assert res["final"]["n_reg_failed"] >= 1
    assert len(set(picks)) == len(picks), f"重复选中同一站位: {picks}"
    assert res["final"]["awc_gap_recovery"] > 0.5
