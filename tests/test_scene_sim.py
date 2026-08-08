"""PR8 验收: 场景生成器 / Fallback 仿真器 / 闭环冒烟(公式45)。"""

import os
from pathlib import Path

import numpy as np
import pytest

from patent_gap.sensors.model import SensorModel
from patent_gap.simulation.scene_gen import build_trav_grid, generate_scene
from patent_gap.simulation.scene_patches import build_scene_patches


def _sensor():
    return SensorModel(r_min=0.5, r_opt=10, r_max=60,
                       dtheta=np.deg2rad(0.3), dphi=np.deg2rad(0.3),
                       sigma_r=0.005, f_pulse=3e5)


def test_scene_generation_deterministic():
    s1 = generate_scene(seed=3, family="S", density="mid")
    s2 = generate_scene(seed=3, family="S", density="mid")
    np.testing.assert_array_equal(s1.triangles, s2.triangles)
    np.testing.assert_allclose(s1.vertices, s2.vertices)
    s3 = generate_scene(seed=4, family="S", density="mid")
    assert len(s3.vertices) != len(s1.vertices) or not np.allclose(s3.vertices, s1.vertices)


def test_scene_has_required_classes():
    s = generate_scene(seed=0, family="M", density="mid")
    classes = {c.cls for c in s.components}
    for need in ("transformer", "breaker", "disconnector", "ct_pt",
                 "busbar", "gantry", "fence", "building"):
        assert need in classes, need
    live = [c for c in s.components if c.live]
    assert live and all(c.d_safe >= 0 for c in live)


def test_scene_patches_and_grid():
    s = generate_scene(seed=1, family="S", density="low")
    patches, tri_to_patch = build_scene_patches(s)
    assert len(patches) > 50
    assert (tri_to_patch >= -1).all()
    assert patches["area"].min() > 0
    grid = build_trav_grid(s)
    free = grid._compute_free()
    assert free.any()
    # 道路中心可通行
    assert grid.is_free((0.0, s.road_y)) or grid.project_to_free((0.0, s.road_y), 3.0)


def test_fallback_scan_and_association():
    from patent_gap.simulation.helios_bridge import FallbackSimulator

    s = generate_scene(seed=1, family="S", density="low")
    patches, tri_to_patch = build_scene_patches(s)
    sim = FallbackSimulator(s.vertices, s.triangles, tri_to_patch, _sensor(),
                            sim_dtheta_deg=0.5)
    scan = sim.scan(np.array([0.0, s.road_y, 2.0]), "t0", seed=0)
    assert len(scan.points) > 1000
    assert scan.hit_patch is not None and len(scan.hit_patch) == len(scan.points)
    assert (scan.hit_patch >= 0).sum() > 100     # 有点关联到构件 Patch
    d = np.linalg.norm(scan.points - scan.origin[None, :], axis=1)
    assert d.min() >= 0.4 and d.max() <= 60.5    # 量程门(含噪声容差)


@pytest.mark.slow
def test_closed_loop_smoke():
    """公式(45) 闭环冒烟: 一轮真实重扫后 awc 恢复率非负且不下降。"""
    from patent_gap.simulation.closed_loop_v2 import (
        EpisodeConfig, SimWorld, default_init_stations, run_episode,
    )

    s = generate_scene(seed=2, family="S", density="low")
    world = SimWorld.build(s, _sensor(), sim_dtheta_deg=0.6)
    init = default_init_stations(world, n=2)
    assert len(init) >= 1
    out = run_episode(world, init, EpisodeConfig(stations_max=2, rounds_max=2,
                                                 seed=0, method="B10_full"))
    assert out["status"] == "ok"
    h = out["history"]
    assert len(h) >= 2, "闭环未执行任何补扫站"
    awc = [m["awc_gap_recovery"] for m in h]
    assert awc[-1] >= awc[0] - 1e-9
    assert h[-1]["n_stations"] >= 1


def test_scene_generation_reproducible_across_processes():
    """同一 (seed, family, density) 在不同进程必须生成同一场景。

    曾用内置 hash(family+density) 播种，而 Python 的字符串哈希每进程随机化
    (PYTHONHASHSEED)，导致同一 seed 在每次运行中生成完全不同的场景——E2 基准
    的全部结果因此无法复现。此测试在子进程中重新生成并比对几何指纹。
    """
    import hashlib
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import hashlib, sys
        import numpy as np
        from patent_gap.simulation.scene_gen import generate_scene
        s = generate_scene(seed=2, family='S', density='low')
        print(hashlib.md5(np.ascontiguousarray(s.vertices).tobytes()).hexdigest())
    """)
    here = generate_scene(seed=2, family="S", density="low")
    mine = hashlib.md5(np.ascontiguousarray(here.vertices).tobytes()).hexdigest()
    seen = set()
    for salt in ("0", "1", "random"):          # 显式改变 PYTHONHASHSEED
        env = {**os.environ, "PYTHONHASHSEED": salt,
               "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, timeout=300)
        assert r.returncode == 0, r.stderr[-500:]
        seen.add(r.stdout.strip())
    assert seen == {mine}, f"场景随进程而变: 本进程 {mine}, 子进程 {seen}"


def test_planner_sees_bim_only_evaluation_sees_as_built():
    """规划几何与评测几何必须分离, 否则遮挡感知无法被证伪。

    公式(27)(28) 的 Vis 若与评测真值同源, 预测误差恒为 0, "遮挡感知有效"
    就成了同义反复。本测试锁定三件事:
      1. n_temp=0 时两套几何完全一致(既有基准逐比特兼容);
      2. n_temp>0 时临时占位物只进竣工几何, 不进 BIM, 也不产生新的待扫目标;
      3. 此时规划器的可见性预测确实会高估(实际被挡)。
    """
    import numpy as np
    from patent_gap.sensors.model import SensorModel
    from patent_gap.simulation.closed_loop_v2 import SimWorld
    from patent_gap.simulation.scene_gen import generate_scene

    sensor = SensorModel.from_config(
        {"dtheta_deg": 0.4, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005})

    clean = generate_scene(seed=0, family="S", density="low", n_temp=0)
    assert clean.bim_tri_mask().all()
    w0 = SimWorld.build(clean, sensor, sim_dtheta_deg=0.4)
    assert w0.plan_oracle is w0.oracle          # 无偏离时不额外建 BVH

    dirty = generate_scene(seed=0, family="S", density="low", n_temp=6)
    assert sum(not c.in_bim for c in dirty.components) == 6
    w6 = SimWorld.build(dirty, sensor, sim_dtheta_deg=0.4)
    assert w6.plan_oracle is not w6.oracle
    # 临时占位物不是 BIM 资产 → 不得改变待扫目标集, 否则 awc 跨 n_temp 不可比
    assert len(w6.patches) == len(w0.patches)
    assert not (w6.tri_to_patch[~dirty.bim_tri_mask()] >= 0).any()

    pids = [int(x) for x in w6.patches["patch_id"].to_numpy()]
    origin = np.array([0.0, -2.0, 2.0])
    pred = w6.plan_oracle.visibility_batch(origin, w6.sampler, pids, sensor=w6.sensor)
    real = w6.oracle.visibility_batch(origin, w6.sampler, pids, sensor=w6.sensor)
    diff = np.array([pred[p] - real[p] for p in pids])
    assert (diff >= -1e-9).all(), "BIM 几何是竣工几何的子集, 预测只可能高估"
    assert (diff > 1e-6).sum() > 0, "临时占位物没挡住任何目标, 该场景测不出预测误差"
