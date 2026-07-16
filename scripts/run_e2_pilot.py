"""E2 仿真基准 pilot(4.3 节): 场景 × 种子 × 方法 闭环基准 + 配对统计。

用法:
  python scripts/run_e2_pilot.py [--quick]
输出:
  results/exp2/{method}/{scene}/{seed}/run.json   (禁止覆盖: 已存在则跳过)
  results/exp2/summary.json / summary.md
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
    EpisodeConfig, SimWorld, build_ground_truth, default_init_stations, run_episode,
)
from patent_gap.simulation.scene_gen import generate_scene  # noqa: E402

METHODS = ["B0_random", "B1_patent", "B5_occ_rng", "B10_full"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="缩小规模冒烟")
    ap.add_argument("--out", default=str(ROOT / "results/exp2"))
    args = ap.parse_args()

    config = {
        "scenes": [("S", "low"), ("S", "mid"), ("M", "mid")],
        "seeds": [0, 1, 2],
        "methods": METHODS,
        "sensor": {"dtheta_deg": 0.4, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005},
        "sim_dtheta_deg": 0.4,
        "budget": {"stations_max": 6, "length_max_m": 400.0, "rounds_max": 6},
        "init_stations": 3,
        "rho0": 50.0,   # pilot 标定值: 0.4°仿真步长下 r_opt 可达密度的 1/4(OPEN_ISSUES #18)
        "lambda_e": 1.0,
    }
    if args.quick:
        config["scenes"] = [("S", "low")]
        config["seeds"] = [0]
        config["budget"]["stations_max"] = 2
        config["budget"]["rounds_max"] = 2

    cfg_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    commit = git_commit()
    out_root = Path(args.out)
    print(f"[e2] config_hash={cfg_hash} commit={commit}")

    for family, density in config["scenes"]:
        for seed in config["seeds"]:
            scene_name = f"scene_{family}_{density}"
            t0 = time.time()
            scene = generate_scene(seed=seed, family=family, density=density)
            sensor = SensorModel.from_config(config["sensor"])
            world = SimWorld.build(scene, sensor,
                                   sim_dtheta_deg=config["sim_dtheta_deg"])
            init = default_init_stations(world, n=config["init_stations"])
            # 真值与方法无关, 每 (场景,种子) 只算一次
            from patent_gap.simulation.closed_loop_v2 import ObsState
            probe = ObsState(world=world)
            for k, o in enumerate(init):
                probe.add_station(o, f"init_{k}", seed=seed * 100 + k)
            gt = build_ground_truth(world, list(probe.masks))
            print(f"[e2] {scene_name}/seed{seed}: patches={len(world.patches)} "
                  f"init={len(init)} gt_gaps={int(np.sum(gt['y']))} "
                  f"full_st={gt['n_full_stations']} setup={time.time()-t0:.1f}s")

            for method in config["methods"]:
                run_dir = out_root / method / scene_name / str(seed)
                run_path = run_dir / "run.json"
                if run_path.exists():
                    print(f"[e2]   {method}: exists, skip (禁止覆盖)")
                    continue
                t1 = time.time()
                try:
                    ep = EpisodeConfig(
                        stations_max=config["budget"]["stations_max"],
                        length_max_m=config["budget"]["length_max_m"],
                        rounds_max=config["budget"]["rounds_max"],
                        rho0=config["rho0"], lambda_e=config["lambda_e"],
                        seed=seed, method=method)
                    result = run_episode(world, init, ep, gt=gt)
                    result["status"] = "ok"
                except Exception as e:  # 缺失结果显式登记, 禁止静默删除
                    result = {"method": method, "seed": seed, "status": "failed",
                              "error": repr(e), "history": [], "final": None}
                result.update({
                    "config_hash": cfg_hash, "git_commit": commit,
                    "scene": scene_name, "scene_seed": seed,
                    "runtime_s": round(time.time() - t1, 2),
                })
                result.pop("gt", None)  # 真值另存, run.json 保持精简
                run_dir.mkdir(parents=True, exist_ok=True)
                run_path.write_text(json.dumps(result, indent=1))
                fin = result.get("final") or {}
                print(f"[e2]   {method}: status={result['status']} "
                      f"awc={fin.get('awc_gap_recovery', float('nan')):.3f} "
                      f"len={fin.get('path_len_m', 0):.0f}m "
                      f"st={fin.get('n_stations', 0)} ({result['runtime_s']}s)")
            gt_path = out_root / "gt" / scene_name / f"{seed}.json"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.write_text(json.dumps(
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gt.items()}))

    summarize(out_root, config, cfg_hash, commit)


def summarize(out_root: Path, config: dict, cfg_hash: str, commit: str) -> None:
    rows = []
    for method in config["methods"]:
        for family, density in config["scenes"]:
            for seed in config["seeds"]:
                p = out_root / method / f"scene_{family}_{density}" / str(seed) / "run.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                fin = r.get("final") or {}
                rows.append({"method": method, "scene": f"scene_{family}_{density}",
                             "seed": seed, "status": r["status"], **fin})
    by_method: dict[str, dict[str, list]] = {}
    keys = ["awc_gap_recovery", "dens_ok", "crit_recall", "path_len_m",
            "n_stations", "ig_per_m"]
    for row in rows:
        m = by_method.setdefault(row["method"], {k: [] for k in keys})
        for k in keys:
            m[k].append(row.get(k, float("nan")))

    summary = {"config": config, "config_hash": cfg_hash, "git_commit": commit,
               "n_runs": len(rows), "methods": {}}
    for m, vals in by_method.items():
        entry = {}
        for k in keys:
            arr = np.asarray(vals[k], dtype=float)
            lo, hi = bootstrap_ci(arr, n=5000)
            entry[k] = {"mean": float(np.nanmean(arr)), "ci95": [lo, hi],
                        "n": int(np.isfinite(arr).sum())}
        summary["methods"][m] = entry

    # 配对置换检验: B10 vs 各基线, awc 主指标 + 次指标, Holm 校正
    def paired(metric, m1, m2):
        pairs1, pairs2 = [], []
        idx = {(r["method"], r["scene"], r["seed"]): r for r in rows}
        for family, density in config["scenes"]:
            for seed in config["seeds"]:
                k1 = (m1, f"scene_{family}_{density}", seed)
                k2 = (m2, f"scene_{family}_{density}", seed)
                if k1 in idx and k2 in idx:
                    pairs1.append(idx[k1].get(metric, float("nan")))
                    pairs2.append(idx[k2].get(metric, float("nan")))
        return np.asarray(pairs1, float), np.asarray(pairs2, float)

    tests = []
    for baseline in ["B0_random", "B1_patent", "B5_occ_rng"]:
        for metric in ["awc_gap_recovery", "ig_per_m", "crit_recall"]:
            x, y = paired(metric, "B10_full", baseline)
            pv = paired_permutation(x, y, n=10000, seed=0)
            tests.append({"comparison": f"B10_full vs {baseline}", "metric": metric,
                          "mean_diff": float(np.nanmean(x - y)), "p_raw": pv})
    adj = holm([t["p_raw"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_holm"] = float(a) if np.isfinite(a) else None
    summary["paired_tests"] = tests

    (out_root / "summary.json").write_text(json.dumps(summary, indent=1))

    lines = ["# E2 pilot 结果汇总", "",
             f"- config_hash `{cfg_hash}`, commit `{commit}`, runs={len(rows)}", "",
             "| 方法 | awc恢复率 | dens_ok | crit_recall | 路径(m) | 站数 | 单位路径增益 |",
             "|---|---|---|---|---|---|---|"]
    for m in config["methods"]:
        e = summary["methods"].get(m)
        if not e:
            continue
        lines.append(
            f"| {m} | {e['awc_gap_recovery']['mean']:.3f} | {e['dens_ok']['mean']:.3f} "
            f"| {e['crit_recall']['mean']:.3f} | {e['path_len_m']['mean']:.0f} "
            f"| {e['n_stations']['mean']:.1f} | {e['ig_per_m']['mean']:.4f} |")
    lines += ["", "## 配对置换检验(Holm 校正)", "",
              "| 对比 | 指标 | Δ均值 | p_raw | p_holm |", "|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['comparison']} | {t['metric']} | {t['mean_diff']:+.3f} "
                     f"| {t['p_raw']:.4f} | {t['p_holm']:.4f} |")
    (out_root / "summary.md").write_text("\n".join(lines))
    print(f"[e2] summary → {out_root/'summary.md'}")


if __name__ == "__main__":
    main()
