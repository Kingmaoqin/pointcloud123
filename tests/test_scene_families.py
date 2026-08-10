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


@pytest.mark.parametrize("family", ["S", "M", "L"])
@pytest.mark.parametrize("density", ["low", "mid", "high"])
def test_equipment_stays_inside_the_site(family, density):
    """间隔排必须放得进场地。

    bay_pitch 原先只从 U(9,14) 抽而不与场地宽度 W 约束, 于是 L/high(n_bay=8,
    上限 8.9 m) 20 个种子里 16 个、M/high(上限 9.2 m) 15 个把设备摆到围栏外,
    最远 14 m —— 那里没有可通行格, 永远扫不到, 等于给 awc 压一个人为天花板,
    而 E5 用种子 0/1/2 恰好躲开了 L/high。扩种子是统计功效所必需的, 所以这条
    必须在扩样本之前锁住。
    """
    for seed in range(20):
        scene = generate_scene(seed=seed, family=family, density=density)
        xmin, ymin, xmax, ymax = scene.bounds_xy
        for c in scene.components:
            if c.cls == "ground":
                continue
            over = max(xmin - c.bbox_min[0], c.bbox_max[0] - xmax,
                       ymin - c.bbox_min[1], c.bbox_max[1] - ymax)
            # 容差必须紧到能抓住嵌在围栏里的构件。此前取 0.5 m, 恰好放过 S 族
            # CT/PT 那 0.143 m 的越界, 而参数表又漏了 E2 实际使用的 S/low、
            # S/mid —— 测试宣称锁住这条却没锁住。
            assert over <= 0.05, (f"{family}/{density} seed{seed}: {c.cls} "
                                  f"越出场地 {over:.3f} m")


def test_every_component_can_be_targeted():
    """候选目标集必须覆盖全部构件, 否则整类设备永远没有为它生成的视点。

    原先目标集固定截断到 60 行, 而排序键 G_task·A 里 G_task 只跨 1.0-1.3、
    Patch 面积跨约 200 倍, 重要度抬不过面积。实测即使在最小的 S/low 上,
    隔离开关/CT-PT/避雷器/绝缘子四类**一个候选都没有**, 其中两类计入
    crit_recall —— 那些指标于是主要反映候选生成而非规划。
    """
    import numpy as np
    import patent_gap.simulation.closed_loop_v2 as cl

    for family, density in (("S", "low"), ("L", "high")):
        scene = generate_scene(seed=0, family=family, density=density)
        world = SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8)
        obs = ObsState(world=world)
        for k, o in enumerate(default_init_stations(world, n=3)):
            obs.add_station(o, f"i{k}", seed=k)
        s = cl.compute_scores(obs, extended=True, rho0=50.0)
        s = s.copy()
        s["_gain"] = s["G_task"] * s["area"]
        s = s.sort_values("_gain", ascending=False)
        n = int(np.clip(2 * s["element_guid"].nunique(), 60, 400))
        tgt = s.groupby("element_guid", sort=False).head(2).head(n)

        assert tgt["element_guid"].nunique() == world.patches["element_guid"].nunique()
        classes = set(world.patches["ifc_class"]) - {"ground"}
        missing = classes - set(tgt["ifc_class"])
        assert not missing, f"{family}/{density}: 这些类没有任何候选目标 {sorted(missing)}"


def test_tight_budget_degrades_instead_of_collapsing():
    """预算收紧时站数应单调减少, 而不是塌成零站。

    公式(42) 的 cost 是时间当量(dist + t_scan·v_move), 而 length_max_m 是纯路径
    预算, 懒惰贪心的预算又被抬高过 n_left·scan_equiv —— 两者只在恰好选满时等价。
    选少了等式就松, 整个站集可能没有一站走得到, 原实现直接终止 episode。实测
    S/low 在 length_max=25 m 下站数=0、awc=0, 而预算内的候选有两百多个。
    """
    import patent_gap.simulation.closed_loop_v2 as cl

    scene = generate_scene(seed=0, family="S", density="low")
    world = SimWorld.build(scene, SensorModel.from_config(SENSOR), sim_dtheta_deg=0.8)
    init = default_init_stations(world, n=3)
    probe = ObsState(world=world)
    for k, o in enumerate(init):
        probe.add_station(o, f"i{k}", seed=k)
    gt = cl.build_ground_truth(world, list(probe.masks))

    seen = {}
    for length_max in (400.0, 60.0, 25.0, 12.0):
        fin = cl.run_episode(world, init, cl.EpisodeConfig(
            stations_max=6, length_max_m=length_max, rounds_max=6,
            rho0=50.0, seed=0, method="B10_full"), gt=gt)["final"]
        assert fin["n_stations"] >= 1, f"length_max={length_max}: 一站都没选出来"
        assert fin["path_len_m"] <= length_max + 1e-9
        assert fin["stop_reason"] in ("rounds", "stations", "length", "no_candidate")
        seen[length_max] = fin["awc_gap_recovery"]
    # 站数不要求单调 —— 预算紧时贪心会挑更近的站, 因而可能挤进更多站。
    #
    # 恢复率也**不是严格单调**的: 每轮重规划的性价比贪心在纯路径预算下没有这个
    # 保证, 预算放宽会改变首站选择, 截断视野内的收益可能反而略低。此处实测
    # S/low seed0 在 25 m→60 m 之间有一次 0.009 的反转(0.131→0.122)。该断言
    # 曾经成立, 是因为当时目标集里含厂房与围墙 —— 那是面积大、边缘、两站就能
    # 覆盖的分块, 把小预算档的恢复率整个抬了起来(25 m 档 0.82 对 0.13)。目标集
    # 改为只含验收资产后, 剩下的设备才是真正难扫的, 这个由易分块撑起来的单调性
    # 也就不在了。
    #
    # 真正要锁的是本修复针对的性质: 预算收紧时**逐级退化而不是坍塌成零**, 且
    # 宽预算显著优于紧预算。局部反转限定在容差内。
    vals = [seen[b] for b in (12.0, 25.0, 60.0, 400.0)]
    tol = 0.02
    for a, b in zip(vals, vals[1:]):
        assert b >= a - tol, f"恢复率随预算显著倒退: {vals}"
    assert vals[-1] > vals[0] + 0.3, f"宽预算未显著优于紧预算: {vals}"
