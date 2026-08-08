"""闭环执行策略 A/B：公式(45) 第3步"执行哪一站"的三种取法对比。

背景：公式(45) 规定"选站(42) → 定序(TSP) → 执行首站 → 闭环更新"。但 TSP 是为
"走完整条路线"排序的，而闭环每轮只执行一站即重新规划，取 TSP 首站等于系统性地
挑最近而非最有价值的站。本脚本在同一批场景上比较：

  tsp_first     公式(45) 原定：TSP 路线首站
  greedy_first  懒惰贪心的首选（边际增益/成本最大者）
  j_step        单步 J 最大者（公式(44) 的单步形式）。信息项与路径项按**步内
                min-max** 归一后再加权：ΔF/max ΔF + λ_reg·R_reg − λ_len·dist/max dist。
                若沿用公式(44) 的 F_ub / L_diag 归一，单步下距离项会比信息项大
                一到两个数量级，取极大即退化为"挑最近"，与 tsp_first 同解。

输出：results/exec_policy_ab/{policy}/{scene}/{seed}/run.json 与 summary.md
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

from patent_gap.evaluation.stats import bootstrap_ci, holm, paired_permutation  # noqa: E402
from patent_gap.sensors.model import SensorModel  # noqa: E402
from patent_gap.simulation.closed_loop_v2 import (  # noqa: E402
    EpisodeConfig, ObsState, SimWorld, build_ground_truth,
    default_init_stations, run_episode,
)
from patent_gap.simulation.scene_gen import generate_scene  # noqa: E402

POLICIES = ["tsp_first", "greedy_first", "j_step"]
SCENES = [("S", "low"), ("S", "mid"), ("M", "mid")]
# 策略选择必须在与 E2 基准（seed 0,1,2）不相交的场景上做，否则等于在评测集上
# 调参，会系统性抬高后续报告中的优势。
SEEDS = [10, 11, 12, 13, 14]
METRICS = ["awc_gap_recovery", "crit_recall", "dens_ok", "path_len_m", "ig_per_m"]


def git_commit() -> str:
    try:
        # 必须带 --dirty：脏工作树跑出的结果若只记 HEAD，会被盖上一个"该代码
        # 当时并不存在"的 commit 戳（E2 pilot 即因此标成了 7a91f13）。
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/exec_policy_ab"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out)

    cfg = {"scenes": SCENES, "seeds": SEEDS, "policies": POLICIES,
           "sensor": {"dtheta_deg": 0.4, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005},
           "budget": {"stations_max": 6, "length_max_m": 400.0, "rounds_max": 6},
           "rho0": 50.0, "init_stations": 3}
    if args.quick:
        cfg["scenes"], cfg["seeds"] = [("S", "low")], [2]
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    commit = git_commit()
    print(f"[ab] config_hash={cfg_hash} commit={commit}")

    for family, density in cfg["scenes"]:
        for seed in cfg["seeds"]:
            scene_name = f"scene_{family}_{density}"
            scene = generate_scene(seed=seed, family=family, density=density)
            sensor = SensorModel.from_config(cfg["sensor"])
            world = SimWorld.build(scene, sensor, sim_dtheta_deg=cfg["sensor"]["dtheta_deg"])
            init = default_init_stations(world, n=cfg["init_stations"])
            probe = ObsState(world=world)
            for k, o in enumerate(init):
                probe.add_station(o, f"init_{k}", seed=seed * 100 + k)
            gt = build_ground_truth(world, list(probe.masks))
            print(f"[ab] {scene_name}/seed{seed}: patches={len(world.patches)} "
                  f"gaps={int(np.sum(gt['y']))}")

            for policy in cfg["policies"]:
                run_dir = out_root / policy / scene_name / str(seed)
                run_path = run_dir / "run.json"
                if run_path.exists():
                    print(f"[ab]   {policy}: exists, skip")
                    continue
                t0 = time.time()
                try:
                    ep = EpisodeConfig(
                        stations_max=cfg["budget"]["stations_max"],
                        length_max_m=cfg["budget"]["length_max_m"],
                        rounds_max=cfg["budget"]["rounds_max"],
                        rho0=cfg["rho0"], seed=seed, method="B10_full",
                        exec_policy=policy)
                    res = run_episode(world, init, ep, gt=gt)
                    res["status"] = "ok"
                except Exception as e:
                    res = {"status": "failed", "error": repr(e), "history": [], "final": None}
                res.update({"policy": policy, "scene": scene_name, "scene_seed": seed,
                            "config_hash": cfg_hash, "git_commit": commit,
                            "runtime_s": round(time.time() - t0, 1)})
                res.pop("gt", None)
                run_dir.mkdir(parents=True, exist_ok=True)
                run_path.write_text(json.dumps(res, indent=1))
                fin = res.get("final") or {}
                print(f"[ab]   {policy:13s} awc={fin.get('awc_gap_recovery', float('nan')):.3f} "
                      f"crit={fin.get('crit_recall', float('nan')):.3f} "
                      f"len={fin.get('path_len_m', 0):.0f}m ({res['runtime_s']}s)")

    summarize(out_root, cfg, cfg_hash, commit)


def summarize(out_root: Path, cfg: dict, cfg_hash: str, commit: str) -> None:
    rows = []
    for policy in cfg["policies"]:
        for family, density in cfg["scenes"]:
            for seed in cfg["seeds"]:
                p = out_root / policy / f"scene_{family}_{density}" / str(seed) / "run.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                rows.append({"policy": policy, "scene": r["scene"], "seed": seed,
                             **(r.get("final") or {})})
    if not rows:
        return
    by = {}
    for row in rows:
        by.setdefault(row["policy"], {m: [] for m in METRICS})
        for m in METRICS:
            by[row["policy"]][m].append(row.get(m, float("nan")))

    lines = ["# 闭环执行策略 A/B 结果", "",
             f"- config_hash `{cfg_hash}`, commit `{commit}`, runs={len(rows)}",
             "- 三种策略只改公式(45)第3步『本轮执行哪一站』，其余完全相同", "",
             "| 策略 | awc恢复率 | 关键设备召回 | dens_ok | 路径(m) | 单位路径增益 |",
             "|---|---|---|---|---|---|"]
    name = {"tsp_first": "tsp_first（公式45原定）", "greedy_first": "greedy_first",
            "j_step": "j_step（本次升级）"}
    for pol in cfg["policies"]:
        if pol not in by:
            continue
        v = by[pol]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.0f} | {:.4f} |".format(
            name.get(pol, pol), *[float(np.nanmean(v[m])) for m in METRICS]))

    def paired(metric, a, b):
        idx = {(r["policy"], r["scene"], r["seed"]): r for r in rows}
        xs, ys = [], []
        for family, density in cfg["scenes"]:
            for seed in cfg["seeds"]:
                ka = (a, f"scene_{family}_{density}", seed)
                kb = (b, f"scene_{family}_{density}", seed)
                if ka in idx and kb in idx:
                    xs.append(idx[ka].get(metric, float("nan")))
                    ys.append(idx[kb].get(metric, float("nan")))
        return np.asarray(xs, float), np.asarray(ys, float)

    tests = []
    for base in ("tsp_first", "greedy_first"):
        for metric in ("awc_gap_recovery", "crit_recall", "ig_per_m"):
            x, y = paired(metric, "j_step", base)
            if len(x) == 0:
                continue
            tests.append({"cmp": f"j_step vs {base}", "metric": metric,
                          "diff": float(np.nanmean(x - y)),
                          "p_raw": paired_permutation(x, y, n=10000, seed=0),
                          "ci95": bootstrap_ci(x - y, n=5000)})
    if tests:
        adj = holm([t["p_raw"] for t in tests])
        for t, a in zip(tests, adj):
            t["p_holm"] = float(a) if np.isfinite(a) else None
        lines += ["", "## 配对置换检验（Holm 校正）", "",
                  "| 对比 | 指标 | Δ均值 | 95%CI | p_raw | p_holm |", "|---|---|---|---|---|---|"]
        for t in tests:
            lines.append("| {} | {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {:.4f} | {:.4f} |".format(
                t["cmp"], t["metric"], t["diff"], t["ci95"][0], t["ci95"][1],
                t["p_raw"], t["p_holm"]))

    (out_root / "summary.md").write_text("\n".join(lines))
    (out_root / "summary.json").write_text(json.dumps(
        {"config_hash": cfg_hash, "git_commit": commit, "rows": rows, "tests": tests},
        indent=1, default=str))
    print(f"[ab] summary → {out_root/'summary.md'}")


if __name__ == "__main__":
    main()
