"""留存产物: 遮挡发现对可见性预测误差的影响(专利技术效果依据)。

此前该效果只在对话里跑过、无任何留存产物, 无法被复核。输出写入
results/evidence/ 并带 commit 戳。
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from patent_gap.occlusion.discovery import UnmodeledOccluders  # noqa: E402
from patent_gap.sensors.model import SensorModel  # noqa: E402
import patent_gap.simulation.closed_loop_v2 as cl  # noqa: E402
from patent_gap.simulation.scene_gen import generate_scene  # noqa: E402

commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                 cwd=ROOT, text=True).strip()
sm = SensorModel.from_config({"dtheta_deg": 0.4, "r": [0.5, 10, 60], "sigma_r": 0.005})
print(f"# 遮挡发现对可见性预测误差的影响  commit={commit}", flush=True)
rows = []
for seed in (0, 1, 2):
    sc = generate_scene(seed=seed, family="S", density="low", n_temp=6)
    w = cl.SimWorld.build(sc, sm, sim_dtheta_deg=0.4)
    bim = sc.bim_tri_mask()
    disc = UnmodeledOccluders(sc.vertices, sc.triangles[bim], w.tri_to_patch[bim], seed=seed)
    obs = cl.ObsState(world=w)
    for k, o in enumerate(cl.default_init_stations(w, n=3)):
        obs.add_station(o, f"i{k}", seed=k)
        disc.update(obs.scans[-1].points)
    o2 = disc.build_oracle()
    pids = [int(v) for v in w.patches["patch_id"].to_numpy()]
    xmin, ymin, xmax, ymax = sc.bounds_xy
    vps = [np.array([x, y, 2.0])
           for x in np.linspace(xmin + 6, xmax - 6, 3)
           for y in np.linspace(ymin + 6, ymax - 6, 3) if w.grid.is_free((x, y))]

    def err(orc):
        over = under = 0
        maes = []
        for v in vps:
            pr = orc.visibility_batch(v, w.sampler, pids, sensor=w.sensor)
            re = w.oracle.visibility_batch(v, w.sampler, pids, sensor=w.sensor)
            d = np.array([pr[p] - re[p] for p in pids])
            over += int((d > 1e-6).sum())
            under += int((d < -1e-6).sum())
            maes.append(float(np.abs(d).mean()))
        return over, under, float(np.mean(maes))

    a, b = err(w.plan_oracle), err(o2)
    rows.append({"seed": seed, "n_viewpoints": len(vps), "n_patches": len(pids),
                 "voxels": disc.n_cells(),
                 "bim_only": {"over": a[0], "under": a[1], "mae": a[2]},
                 "with_discovery": {"over": b[0], "under": b[1], "mae": b[2]},
                 "mae_ratio": a[2] / max(b[2], 1e-12)})
    print(f"seed{seed}: 视点{len(vps)}×分块{len(pids)}, 发现体素{disc.n_cells()}", flush=True)
    print(f"  仅BIM       : 高估{a[0]:5d} 低估{a[1]:5d} MAE={a[2]:.4f}", flush=True)
    print(f"  并入发现体素 : 高估{b[0]:5d} 低估{b[1]:5d} MAE={b[2]:.4f} "
          f"(降 {a[2]/max(b[2],1e-12):.1f} 倍)", flush=True)

print(f"\n三种子 MAE 降低倍数: {[round(r['mae_ratio'], 1) for r in rows]}", flush=True)
(ROOT / "results/evidence").mkdir(parents=True, exist_ok=True)
(ROOT / "results/evidence/discovery_vis_error.json").write_text(
    json.dumps({"commit": commit, "rows": rows}, indent=1))
