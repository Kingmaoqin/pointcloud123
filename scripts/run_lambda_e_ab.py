"""A/B: 价值通路的 ρ_req 是否该按工程重要度缩放(公式(33) 的 λ_E 取值)。

(41) 的收益权含 G_task ∝ (1+E_i), (35) 的 g^rng = clip(ρ̂/(ρ_0(1+λ_E·E_i)))。
两处取同一个重要度系数时, 在未饱和区(ρ̂ < ρ_req)二者相乘精确抵消, 重要度对
视点价值的影响严格为零 —— 实测 15 m 外或斜入射处主变与杂物权重之比恰为 1.00。

  lambda_e_value=1.0  升级前行为: 抵消发生, 远处/斜视目标不分重要度
  lambda_e_value=0.0  价值通路用统一参考密度 ρ_0, 重要度只经 G_task 生效

主指标是关键设备召回 crit_recall —— 该改动针对的正是"重要资产被平等对待",
awc 是次指标(它由少数大平面主导, 对重要度不敏感)。

种子取 10–14, 与 E2 基准(0–7)不相交: 在评测集上选参数会系统性抬高优势。

输出: results/lambda_e_ab/{lev}/{scene}/{seed}/run.json 与 summary.md
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

LEVELS = {"cancel_1.0": 1.0, "decoupled_0.0": 0.0}
SCENES = [("S", "low"), ("S", "mid"), ("M", "mid")]
SEEDS = [10, 11, 12, 13, 14]
METRICS = ["crit_recall", "awc_gap_recovery", "dens_ok", "path_len_m"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/lambda_e_ab"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out)

    cfg = {"scenes": SCENES, "seeds": SEEDS, "levels": LEVELS,
           "sensor": {"dtheta_deg": 0.4, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005},
           "budget": {"stations_max": 6, "length_max_m": 400.0, "rounds_max": 6},
           "rho0": 50.0, "init_stations": 3, "method": "B10_full"}
    if args.quick:
        cfg["scenes"], cfg["seeds"] = [("S", "low")], [10]
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    commit = git_commit()
    print(f"[lam] config_hash={cfg_hash} commit={commit}", flush=True)

    for family, density in cfg["scenes"]:
        for seed in cfg["seeds"]:
            scene_name = f"scene_{family}_{density}"
            scene = generate_scene(seed=seed, family=family, density=density)
            sensor = SensorModel.from_config(cfg["sensor"])
            world = SimWorld.build(scene, sensor,
                                   sim_dtheta_deg=cfg["sensor"]["dtheta_deg"])
            init = default_init_stations(world, n=cfg["init_stations"])
            probe = ObsState(world=world)
            for k, o in enumerate(init):
                probe.add_station(o, f"init_{k}", seed=seed * 100 + k)
            gt = build_ground_truth(world, list(probe.masks))
            print(f"[lam] {scene_name}/seed{seed}: patches={len(world.patches)} "
                  f"gaps={int(np.sum(gt['y']))}", flush=True)

            for lev, lam in cfg["levels"].items():
                run_dir = out_root / lev / scene_name / str(seed)
                run_path = run_dir / "run.json"
                if run_path.exists():
                    continue
                t0 = time.time()
                try:
                    ep = EpisodeConfig(
                        stations_max=cfg["budget"]["stations_max"],
                        length_max_m=cfg["budget"]["length_max_m"],
                        rounds_max=cfg["budget"]["rounds_max"],
                        rho0=cfg["rho0"], seed=seed, method=cfg["method"],
                        lambda_e_value=lam)
                    res = run_episode(world, init, ep, gt=gt)
                    res["status"] = "ok"
                except Exception as e:
                    res = {"status": "failed", "error": repr(e), "history": [], "final": None}
                res.update({"level": lev, "lambda_e_value": lam, "scene": scene_name,
                            "scene_seed": seed, "config_hash": cfg_hash,
                            "git_commit": commit, "runtime_s": round(time.time() - t0, 1)})
                res.pop("gt", None)
                run_dir.mkdir(parents=True, exist_ok=True)
                run_path.write_text(json.dumps(res, indent=1))
                fin = res.get("final") or {}
                print(f"[lam]   {lev:14s} crit={fin.get('crit_recall', float('nan')):.3f} "
                      f"awc={fin.get('awc_gap_recovery', float('nan')):.3f} "
                      f"({res['runtime_s']}s)", flush=True)

    summarize(out_root, cfg, cfg_hash, commit)


def summarize(out_root: Path, cfg: dict, cfg_hash: str, commit: str) -> None:
    rows = []
    for lev in cfg["levels"]:
        for family, density in cfg["scenes"]:
            for seed in cfg["seeds"]:
                p = out_root / lev / f"scene_{family}_{density}" / str(seed) / "run.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                rows.append({"level": lev, "scene": r["scene"], "seed": seed,
                             **(r.get("final") or {})})
    if not rows:
        return

    lines = ["# 价值通路 ρ_req 重要度系数 A/B", "",
             f"- config_hash `{cfg_hash}`, commit `{commit}`, runs={len(rows)}",
             "- 主指标 crit_recall(关键设备召回); 场景种子 10–14, 与 E2 基准不相交", "",
             "| 取值 | 关键设备召回 | awc恢复率 | dens_ok | 路径(m) |", "|---|---|---|---|---|"]
    for lev in cfg["levels"]:
        sub = [r for r in rows if r["level"] == lev]
        if not sub:
            continue
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.0f} |".format(
            lev, *[float(np.nanmean([r.get(m, float("nan")) for r in sub]))
                   for m in METRICS]))

    idx = {(r["level"], r["scene"], r["seed"]): r for r in rows}
    tests = []
    for metric in METRICS[:3]:
        a, b = [], []
        for family, density in cfg["scenes"]:
            for seed in cfg["seeds"]:
                k1 = ("decoupled_0.0", f"scene_{family}_{density}", seed)
                k0 = ("cancel_1.0", f"scene_{family}_{density}", seed)
                if k0 in idx and k1 in idx:
                    a.append(idx[k1].get(metric, float("nan")))
                    b.append(idx[k0].get(metric, float("nan")))
        if not a:
            continue
        x, y = np.asarray(a, float), np.asarray(b, float)
        tests.append({"metric": metric, "diff": float(np.nanmean(x - y)),
                      "ci95": bootstrap_ci(x - y, n=5000),
                      "p_raw": paired_permutation(x, y, seed=0)})
    if tests:
        for t, adj in zip(tests, holm([t["p_raw"] for t in tests])):
            t["p_holm"] = float(adj) if np.isfinite(adj) else float("nan")
        lines += ["", "## 配对置换检验 (decoupled_0.0 − cancel_1.0, Holm 校正)", "",
                  "| 指标 | Δ均值 | 95%CI | p_raw | p_holm |", "|---|---|---|---|---|"]
        for t in tests:
            lines.append("| {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {:.4f} | {:.4f} |".format(
                t["metric"], t["diff"], t["ci95"][0], t["ci95"][1],
                t["p_raw"], t["p_holm"]))

    (out_root / "summary.md").write_text("\n".join(lines))
    (out_root / "summary.json").write_text(json.dumps(
        {"config_hash": cfg_hash, "git_commit": commit, "rows": rows, "tests": tests},
        indent=1, default=str))
    print(f"[lam] summary → {out_root/'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
