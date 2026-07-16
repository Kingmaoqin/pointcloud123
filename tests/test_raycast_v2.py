"""Gate 1 判据(3.2): 遮挡三单测 + 公式(26)确定性 + 公式(31)数值 + 平板校准。"""

import numpy as np

from patent_gap.density.expected import expected_density
from patent_gap.occlusion.raycast import OcclusionOracle, occlusion_gap_evidence
from patent_gap.occlusion.sampling import sample_patch_surface
from patent_gap.sensors.model import SensorModel


def _quad(z, half=1.0, hole=None):
    """z 平面上边长 2*half 的方形(两三角), hole=(hx0,hx1) 时挖竖条孔。"""
    if hole is None:
        v = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]])
        f = np.array([[0, 1, 2], [0, 2, 3]])
        return v, f
    hx0, hx1 = hole
    vs, fs = [], []
    for x0, x1 in [(-half, hx0), (hx1, half)]:
        base = len(vs)
        vs += [[x0, -half, z], [x1, -half, z], [x1, half, z], [x0, half, z]]
        fs += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return np.array(vs, dtype=float), np.array(fs)


def _stack(*parts):
    verts, faces, patch = [], [], []
    nv = 0
    for pid, (v, f) in enumerate(parts):
        verts.append(v)
        faces.append(f + nv)
        patch.append(np.full(len(f), pid))
        nv += len(v)
    return np.vstack(verts), np.vstack(faces), np.concatenate(patch)


def test_gate1_blocker_and_target():
    """前挡板(z=1)+后目标(z=2), 站在 z=0: 目标 Vis=0, 挡板 Vis=1。"""
    V, F, P = _stack(_quad(1.0, half=2.0), _quad(2.0, half=1.0))
    oracle = OcclusionOracle(V, F, P)
    origin = np.array([0.0, 0.0, 0.0])
    tgt = sample_patch_surface(V, F[P == 1], 4.0, 1)
    blk = sample_patch_surface(V, F[P == 0], 16.0, 0)
    assert oracle.visibility(origin, tgt, target_patch=1) == 0.0
    assert oracle.visibility(origin, blk, target_patch=0) == 1.0


def test_gate1_half_hole():
    """挡板开 50% 孔洞: |Vis−0.5| < 0.05。"""
    # 目标 x∈[-1,1]; 挡板留孔 x∈[-1,0](恰遮 x∈[0,1] 一半)
    blocker = _quad(1.0, half=8.0, hole=(-1.0, 0.0))
    target = _quad(2.0, half=1.0)
    V, F, P = _stack(blocker, target)
    oracle = OcclusionOracle(V, F, P)
    origin = np.array([0.0, 0.0, 0.0])
    # 目标平面均匀采样(用足够多点降低采样误差)
    xs, ys = np.meshgrid(np.linspace(-0.99, 0.99, 60), np.linspace(-0.99, 0.99, 60))
    samples = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 2.0)], axis=1)
    # 由原点看目标: 视线过 z=1 处 x 坐标 = 样本x/2 → 被遮当 样本x/2 ∈ [0, 8]即样本x>0
    vis = oracle.visibility(origin, samples, target_patch=1)
    assert abs(vis - 0.5) < 0.05


def test_gate1_box_backface():
    """盒体背面 Patch: Vis=0。"""
    import trimesh
    box = trimesh.creation.box(extents=[2, 2, 2])
    box.apply_translation([0, 0, 5.0])
    V = np.asarray(box.vertices, dtype=float)
    F = np.asarray(box.faces)
    # 顶面(z=6)为 patch 1, 其余 patch 0; 站在原点下方看不到顶面
    fc = V[F].mean(axis=1)
    P = np.where(fc[:, 2] > 5.9, 1, 0)
    oracle = OcclusionOracle(V, F, P)
    top_tris = F[P == 1]
    samples = sample_patch_surface(V, top_tris, 4.0, 1)
    assert oracle.visibility(np.array([0.0, 0.0, 0.0]), samples, target_patch=1) == 0.0


def test_sampling_deterministic():
    V, F = _quad(0.0, half=1.0)
    s1 = sample_patch_surface(V, F, 4.0, patch_id=7)
    s2 = sample_patch_surface(V, F, 4.0, patch_id=7)
    np.testing.assert_array_equal(s1, s2)
    assert 4 <= len(s1) <= 64
    # 落在面内
    assert np.all(np.abs(s1[:, :2]) <= 1.0 + 1e-9)
    assert np.allclose(s1[:, 2], 0.0)


def test_occlusion_gap_evidence():
    d = occlusion_gap_evidence({"s0": {1: 0.3}, "s1": {1: 0.8, 2: 0.1}}, [1, 2, 3])
    assert abs(d[1] - 0.2) < 1e-12
    assert abs(d[2] - 0.9) < 1e-12
    assert d[3] == 1.0  # 无站可见


def test_expected_density_formula31():
    """Δθ=Δφ=0.04°, d=10 m, 正视 → ρ̂≈2.05e4 点/m²; d 加倍 → ρ̂/4 (±1%)。"""
    sm = SensorModel(r_min=0.5, r_opt=10, r_max=60,
                     dtheta=np.deg2rad(0.04), dphi=np.deg2rad(0.04),
                     sigma_r=0.005, f_pulse=3e5)
    rho10 = float(expected_density(np.array([10.0]), np.array([1.0]), sm)[0])
    expected = 1.0 / (100 * np.deg2rad(0.04) ** 2)
    assert abs(rho10 - expected) / expected < 0.01
    assert abs(rho10 - 2.05e4) / 2.05e4 < 0.02
    rho20 = float(expected_density(np.array([20.0]), np.array([1.0]), sm)[0])
    assert abs(rho20 - rho10 / 4) / (rho10 / 4) < 0.01
    # 量程门
    assert float(expected_density(np.array([100.0]), np.array([1.0]), sm)[0]) == 0.0


def test_plate_calibration_within_15pct():
    """Gate 1 附加判据: 仿真扫平板, 实测密度与 ρ̂ 相对误差 < 15%。"""
    from patent_gap.simulation.helios_bridge import FallbackSimulator

    V, F = _quad(0.0, half=2.0)
    # 平板放到 x=8 处, 面向原点(法向 -x): 旋转 quad 到 yz 平面
    Vp = np.stack([np.full(len(V), 8.0), V[:, 0], V[:, 1] + 2.0], axis=1)
    P = np.zeros(len(F), dtype=np.int64)
    sm = SensorModel(r_min=0.5, r_opt=10, r_max=60,
                     dtheta=np.deg2rad(0.2), dphi=np.deg2rad(0.2),
                     sigma_r=0.0, f_pulse=3e5)
    sim = FallbackSimulator(Vp, F, P, sm, sim_dtheta_deg=0.2)
    scan = sim.scan(np.array([0.0, 0.0, 2.0]), "cal", seed=0)
    # 中心 1×1 m 区域实测密度
    pts = scan.points
    sel = (np.abs(pts[:, 1]) < 0.5) & (np.abs(pts[:, 2] - 2.0) < 0.5)
    measured = sel.sum() / 1.0
    predicted = float(expected_density(np.array([8.0]), np.array([1.0]),
                                       sim.effective_sensor())[0])
    assert abs(measured - predicted) / predicted < 0.15
