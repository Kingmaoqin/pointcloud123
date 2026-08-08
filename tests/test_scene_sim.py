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
