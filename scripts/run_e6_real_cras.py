"""E6: 升级后的流水线第一次在真实 IFC 与真实点云上运行（OPEN_ISSUES #11）。

数据：CRAS labs@FEUP。真实 IFC 模型（604187 顶点 / 1197750 三角面 / 256 构件）
与 584701977 个实测点，后者按 5 cm 阈值关联到构件（12835294 点匹配）。

两种缺口真值，分开跑，因为它们回答不同的问题：

  measured  缺口由**实测点云**定义：构件级匹配点数按面积分摊到分块，密度低于
            阈值即判为缺口。这是真实的"哪里没扫到"，但构件内分摊是近似。
  synthetic 缺口由仿真初始站定义（与 E2 同口径），用于与合成场景的结果对照，
            隔离出"真实几何本身带来的差异"。

**必须随结果一并声明的限制**：
1. 数据集只提供一份融合后的 ASC，无逐站位姿（docs/data_audit.md），因此补扫站的
   重新采集由射线仿真器在真实 IFC 网格上完成——几何与缺口是真的，重扫是仿真的。
2. CRAS 是实验楼，不是变电站。本实验检验的是"流水线能否吃真实 IFC 与真实点云"，
   不能替代变电站场景下的效果验证。
3. 工程重要度按建筑口径映射（real_scene.IFC_IMPORTANCE），为【推断】值。

输出：results/exp6/{gt_mode}/{method}/run.json 与 summary.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from patent_gap.planning.objective import (  # noqa: E402
    T_SCAN_DEFAULT as T_SCAN, V_MOVE_DEFAULT as V_MOVE,
)
from patent_gap.sensors.model import SensorModel  # noqa: E402
from patent_gap.simulation.closed_loop_v2 import (  # noqa: E402
    EpisodeConfig, ObsState, SimWorld, build_ground_truth,
    default_init_stations, run_episode,
)
from patent_gap.simulation.real_scene import (  # noqa: E402
    load_real_scene, real_initial_coverage, real_scene_report,
)

METHODS = ["Bdisp_maxmin", "Bbim_offline", "B1_patent", "B5_occ_rng",
           "B10_full", "B11_disc"]
GT_MODES = ["measured", "synthetic"]
PROCESSED = ROOT / "data" / "processed"
# 实测点密度达到该值即视为该分块已被覆盖。CRAS 匹配点为 5 cm 邻域内的回波,
# 1 点/m² 是很宽松的门槛 —— 取严会把大量弱覆盖分块也判成缺口, 掩盖方法差异。
RHO_FLOOR = 1.0


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/exp6"))
    ap.add_argument("--stations", type=int, default=0,
                    help="0 = 按构件数缩放")
    ap.add_argument("--dtheta", type=float, default=0.6)
    # 室内标定: 参考站 8 m 网格在被墙分隔的房间里只落得下 3 站(C_gt 均值 0.048,
    # 真值本身退化); 候选环 (4,8,14) m 全部落进墙里。以下为 CRAS 尺度取值。
    ap.add_argument("--full-spacing", type=float, default=2.5)
    ap.add_argument("--cand-distances", nargs=3, type=float, default=[1.5, 3.0, 5.0])
    ap.add_argument("--cand-grid", type=float, default=3.0)
    # 室内三脚架半径约 0.35 m; 0.6 m(户外轮式平台)会多吃掉 65 m² 自由空间
    ap.add_argument("--r-robot", type=float, default=0.35)
    # 障碍按三角面真实投影栅格化, 而非构件包围盒。实测把最大连通域从 28% 提到
    # 79% —— 包围盒对变电站方箱设备够用, 对建筑的 L 形墙与门洞会连通路一起封死。
    ap.add_argument("--aabb-obstacles", action="store_true",
                    help="退回包围盒栅格化(仅用于与旧结果对照)")
    # 排除半径 3.0 m 在 142 m² 的室内自由空间里六站就超过全部空间
    ap.add_argument("--r-dup", type=float, default=1.0)
    # 预算按构件数缩放(与 E5 同口径): 256 构件 / 9.3 ≈ 28 站。原先固定 6 站只有
    # 参考普查(26 站)的 23%, "追不回缺口"与"预算不够"混在一起。
    ap.add_argument("--stations-per-components", type=float, default=9.3)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--gt-modes", nargs="*", default=GT_MODES)
    ap.add_argument("--lambda-reg", type=float, default=0.5)
    ap.add_argument("--exec-policy", default="tsp_first",
                    choices=["tsp_first", "greedy_first", "j_step"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out_root = Path(args.out)
    commit = git_commit()

    scene, _ = load_real_scene(PROCESSED)
    sensor_cfg = {"dtheta_deg": args.dtheta, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005}
    sensor = SensorModel.from_config(sensor_cfg)
    t0 = time.time()
    world = SimWorld.build(scene, sensor, sim_dtheta_deg=args.dtheta,
                           r_robot=args.r_robot,
                           by_triangle=not args.aabb_obstacles)
    report = real_scene_report(scene, world.patches)
    print(f"[e6] commit={commit}  真实场景 {json.dumps(report, ensure_ascii=False)}",
          flush=True)
    print(f"[e6] SimWorld 构建 {time.time()-t0:.0f}s", flush=True)

    # 室内无"主干道路", 必须用自由空间最大最小离散撒初始站
    init = default_init_stations(world, n=3, spread=True)
    probe = ObsState(world=world)
    for k, o in enumerate(init):
        probe.add_station(o, f"init_{k}", seed=k)
    gt_syn = build_ground_truth(world, list(probe.masks),
                                full_spacing=args.full_spacing)

    C_meas, n_meas = real_initial_coverage(PROCESSED, world.patches,
                                           world.tri_to_patch, scene,
                                           rho_floor=RHO_FLOOR)
    gt_meas = dict(gt_syn)
    gt_meas["C_init"] = C_meas
    gt_meas["y"] = C_meas < 0.5 * gt_syn["C_gt"]
    print(f"[e6] 缺口数: measured={int(gt_meas['y'].sum())} "
          f"synthetic={int(gt_syn['y'].sum())} / {len(world.patches)} 分块; "
          f"实测覆盖均值 {C_meas.mean():.3f}", flush=True)

    n_comp = int(world.patches["element_guid"].nunique())
    n_st = args.stations or max(3, int(round(n_comp / args.stations_per_components)))
    print(f"[e6] 预算 {n_st} 站 (构件 {n_comp}, 参考普查 {gt_syn['n_full_stations']} 站)",
          flush=True)
    cfg = {"data": "CRAS labs@FEUP", "stations_budget": n_st, "r_robot": args.r_robot,
           "r_dup": args.r_dup, "sensor": sensor_cfg, "rho_floor": RHO_FLOOR,
           "full_spacing": args.full_spacing,
           "cand_distances": list(args.cand_distances),
           "cand_grid_spacing": args.cand_grid,
           "methods": args.methods, "tag": args.tag,
           "lambda_reg": args.lambda_reg, "exec_policy": args.exec_policy,
           "gt_modes": args.gt_modes, "scene": report}
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]

    for mode in args.gt_modes:
        gt = gt_meas if mode == "measured" else gt_syn
        for method in args.methods:
            run_dir = out_root / mode / (method + args.tag)
            run_path = run_dir / "run.json"
            if run_path.exists():
                prev = json.loads(run_path.read_text()).get("git_commit")
                if prev != commit:
                    raise SystemExit(f"拒绝续跑: {run_path} 产自 {prev}, 当前 {commit}")
                continue
            t1 = time.time()
            try:
                ep = EpisodeConfig(stations_max=n_st,
                                   length_max_m=30.0 * n_st,
                                   rounds_max=n_st, rho0=RHO_FLOOR,
                                   seed=0, method=method,
                                   cand_distances=tuple(args.cand_distances),
                                   cand_grid_spacing=args.cand_grid,
                                   cand_r_dup=args.r_dup,
                                   lambda_reg=args.lambda_reg,
                                   exec_policy=args.exec_policy)
                res = run_episode(world, init, ep, gt=gt)
                res["status"] = "ok"
            except Exception as e:
                res = {"status": "failed", "error": repr(e), "history": [], "final": None}
            res.update({"method": method, "gt_mode": mode, "config_hash": cfg_hash,
                        "git_commit": commit, "runtime_s": round(time.time() - t1, 1)})
            res.pop("gt", None)
            run_dir.mkdir(parents=True, exist_ok=True)
            run_path.write_text(json.dumps(res, indent=1))
            fin = res.get("final") or {}
            print(f"[e6] {mode:9s} {method:14s} awc={fin.get('awc_gap_recovery', float('nan')):.3f} "
                  f"asset={fin.get('asset_recovery', float('nan')):.3f} "
                  f"crit={fin.get('crit_recall', float('nan')):.3f} "
                  f"stop={fin.get('stop_reason', '?')} ({res['runtime_s']}s)", flush=True)

    summarize(out_root, cfg, cfg_hash, commit, report)


def summarize(out_root: Path, cfg: dict, cfg_hash: str, commit: str, report: dict) -> None:
    rows = []
    for mode in cfg["gt_modes"]:
        for method in cfg["methods"]:
            p = out_root / mode / (method + cfg.get("tag", "")) / "run.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            fin = dict(r.get("final") or {})
            if fin:
                T = fin.get("path_len_m", 0.0) / V_MOVE + fin.get("n_stations", 0) * T_SCAN
                fin["T_total_s"] = T
            rows.append({"gt_mode": mode, "method": method,
                         "status": r.get("status"), **fin})
    if not rows:
        return
    lines = ["# E6: 真实 IFC + 真实点云（CRAS labs@FEUP）", "",
             f"- config_hash `{cfg_hash}`, commit `{commit}`, runs={len(rows)}",
             f"- 场景: {report['vertices']} 顶点 / {report['triangles']} 三角面 / "
             f"{report['components']} 构件 / {report['patches']} 分块, "
             f"{report['extent_m'][0]}×{report['extent_m'][1]} m, "
             f"表面 {report['total_area_m2']} m²",
             "",
             "> **限制**：数据集只有一份融合 ASC、无逐站位姿，补扫站的重新采集由射线",
             "> 仿真器在真实 IFC 网格上完成——几何与缺口是真的，重扫是仿真的。CRAS 是",
             "> 实验楼而非变电站，本实验检验流水线能否吃真实数据，不替代变电站效果验证。",
             "> `measured` 模式下构件级匹配点数按面积分摊到分块，是一处近似。", "",
             "| 缺口真值 | 方法 | awc恢复率 | 构件等权 | 关键设备召回 | 路径(m) | 站数 | 终止原因 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.0f} | {} | {} |".format(
            r["gt_mode"], r["method"], r.get("awc_gap_recovery", float("nan")),
            r.get("asset_recovery", float("nan")), r.get("crit_recall", float("nan")),
            r.get("path_len_m", 0.0), r.get("n_stations", 0),
            r.get("stop_reason", "?")))
    (out_root / "summary.md").write_text("\n".join(lines))
    (out_root / "summary.json").write_text(json.dumps(
        {"config_hash": cfg_hash, "git_commit": commit, "scene": report, "rows": rows},
        indent=1, default=str))
    print(f"[e6] summary → {out_root/'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
