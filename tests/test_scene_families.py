"""9 类场景(S/M/L × low/mid/high)全部可执行。

E2/E3 只跑过 S/low、S/mid、M/mid 三种; L 家族与 high 密度从未被执行过, 而它们
的规模差出三倍(L/high 有 1606 个 Patch、145 个构件、275 个缺口)。这里不检验
效果好坏, 只确保没有随规模崩掉的路径 —— 效果由 E5 负责。
"""

from __future__ import annotations

import numpy as np
import pytest

from patent_gap.sensors.model import SensorModel
from patent_gap.simulation.closed_loop_v2 import (
    EpisodeConfig, ObsState, SimWorld, build_ground_truth,
    default_init_stations, run_episode,
)
from patent_gap.simulation.scene_gen import generate_scene

SENSOR = {"dtheta_deg": 0.8, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005}


@pytest.mark.slow
@pytest.mark.parametrize("family", ["S", "M", "L"])
@pytest.mark.parametrize("density", ["low", "mid", "high"])
def test_every_scene_family_runs(family, density):
    scene = generate_scene(seed=0, family=family, density=density)
    world = SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8)
    assert len(world.patches) > 0

    init = default_init_stations(world, n=3)
    assert len(init) >= 2, f"{family}/{density}: 初始站投影到可通行图后只剩 {len(init)} 站"
    probe = ObsState(world=world)
    for k, o in enumerate(init):
        probe.add_station(o, f"init_{k}", seed=k)
    gt = build_ground_truth(world, list(probe.masks))
    assert int(gt["y"].sum()) > 0, f"{family}/{density}: 初始站集没留下任何缺口, 该场景测不出东西"

    res = run_episode(world, init, EpisodeConfig(
        stations_max=2, length_max_m=200.0, rounds_max=2,
        rho0=50.0, seed=0, method="B10_full"), gt=gt)
    assert res["status"] == "ok"
    fin = res["final"]
    # 预算内至少要走出一站; 一站都选不出说明候选生成或硬约束在该规模下失效
    assert fin["n_stations"] >= 1, f"{family}/{density}: 一站都没选出来"
    for key in ("awc_gap_recovery", "asset_recovery", "crit_recall", "dens_ok"):
        v = fin[key]
        assert np.isfinite(v) and 0.0 <= v <= 1.0, f"{family}/{density}: {key}={v}"
