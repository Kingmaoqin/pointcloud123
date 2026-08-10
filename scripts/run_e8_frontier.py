"""E8: 与 frontier 探索基线的对照 —— 两个不同的收益口径。

弱先验条件下"下一站去哪"这个问题, 探索类方法已有成熟答案: 往 frontier(已知
自由区与未知区的边界)走, 把地图未知区变已知。合作方会问的正是: 既然 M_plan 现在
也由观测建立、逐轮更新, 本方法与 frontier 探索到底差在哪。

公平性按"只有目标函数不同"设定。二者共用:

    平台与半径、传感器与量程、初始站集、路径预算、站数上限、轮次上限、
    带电体禁入区、A* 与距离场、去重半径、成本折算(路径 + 单站架设当量)、
    观测驱动的同一个 M_plan 实现。

差别只有一处:

    Bfrontier  收益 = 预计新揭示的未知栅格数(二维射线估计)
    B10/B11    收益 = 目标缺口的加权补测量(公式 26-45)

候选集也各按各自的口径取: frontier 取 frontier 栅格本身(Yamauchi 的做法), 本方法
取按目标分块生成的候选环。把本方法的候选环拿给 frontier 用, 等于替它挑好了地方;
反之亦然。二者都不受对方候选口径的限制。

必须两个轴一起报, 不得只报一个:

    mplan_known_ratio  地图覆盖 —— frontier 的目标函数, 它理应领先
    awc_gap_recovery / asset_recovery  任务缺口 —— 本方法的目标函数

"地图覆盖得快"不是"任务缺口补得好"。若只报后者, 就是拿自己的评价标准去判对方;
若把前者的领先说成本方法的劣势, 同样是错的 —— 那本就不是本方法要优化的量。

输出: results/exp8/{level}/{method}/{scene}/{seed}/run.json 与 summary.md
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

LEVELS = [("P50", 0.5, True), ("P0", 0.0, True)]
METHODS = ["Bfrontier", "B10_full", "B11_disc"]

METRICS = [
    ("mplan_known_ratio", "地图覆盖(frontier 的目标)"),
    ("awc_gap_recovery", "任务缺口恢复(本方法的目标)"),
    ("asset_recovery", "资产级恢复"),
    ("crit_recall", "关键设备召回"),
    ("dens_ok", "密度达标率"),
    ("path_len_m", "路径长度 m"),
    ("n_stations", "站数"),
    ("vis_mae", "可见性预测 MAE"),
    ("vis_over", "可见性单向高估"),
    ("invalid_view_frac", "无效视点占比"),
]
PER_ROUND = {"vis_mae", "vis_over", "invalid_view_frac"}


def git_commit() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                        cwd=ROOT, text=True).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def episode_mean(hist: list[dict], key: str) -> float:
    v = [h[key] for h in hist if key in h]
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp8")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = {
        "scenes": [("S", "low"), ("S", "mid"), ("M", "mid")],
        "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
        "n_temp": 6,
        "levels": [l[0] for l in LEVELS],
        "methods": METHODS,
        "sensor": {"dtheta_deg": 0.4, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005},
        "sim_dtheta_deg": 0.4,
        "budget": {"stations_max": 6, "length_max_m": 400.0, "rounds_max": 6},
        "init_stations": 3,
        "rho0": 50.0,
        "lambda_e": 1.0,
        "prior_seed_base": 1000,   # 与 E7 同基, 使同名档的先验抽样完全一致
    }
    levels = list(LEVELS)
    if args.quick:
        cfg["scenes"] = [("M", "mid")]
        cfg["seeds"] = [0]
        cfg["budget"] = {"stations_max": 2, "length_max_m": 400.0, "rounds_max": 2}
        levels = [LEVELS[1]]
        cfg["levels"] = [l[0] for l in levels]

    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    commit = git_commit()
    out_root = ROOT / args.out
    print(f"[e8] config_hash={cfg_hash} commit={commit}", flush=True)

    for family, density in cfg["scenes"]:
        for seed in cfg["seeds"]:
            scene_name = f"scene_{family}_{density}"
            scene = generate_scene(seed=seed, family=family, density=density,
                                   n_temp=cfg["n_temp"])
            sensor = SensorModel.from_config(cfg["sensor"])
            w0 = SimWorld.build(scene, sensor, sim_dtheta_deg=cfg["sim_dtheta_deg"])
            init = default_init_stations(w0, n=cfg["init_stations"])
            probe = ObsState(world=w0)
            for k, o in enumerate(init):
                probe.add_station(o, f"init_{k}", seed=seed * 100 + k)
            gt = build_ground_truth(w0, list(probe.masks))
            print(f"[e8] {scene_name}/seed{seed}: gaps={int(np.sum(gt['y']))}",
                  flush=True)
            del w0, probe

            for level, frac, obs_env in levels:
                for method in cfg["methods"]:
                    run_dir = out_root / level / method / scene_name / str(seed)
                    run_path = run_dir / "run.json"
                    if run_path.exists():
                        prev = json.loads(run_path.read_text()).get("git_commit")
                        if prev != commit:
                            raise SystemExit(
                                f"拒绝续跑: {run_path} 产自 {prev}, 当前 {commit}。"
                                f"断点续跑只判文件存在会拼出版本混合的结果集 —— "
                                f"请先清空该输出目录再重跑。")
                        continue
                    t0 = time.time()
                    world = SimWorld.build(
                        scene, sensor, sim_dtheta_deg=cfg["sim_dtheta_deg"],
                        env_prior_frac=frac, observed_env=obs_env,
                        prior_seed=cfg["prior_seed_base"] + seed)
                    try:
                        ep = EpisodeConfig(
                            stations_max=cfg["budget"]["stations_max"],
                            length_max_m=cfg["budget"]["length_max_m"],
                            rounds_max=cfg["budget"]["rounds_max"],
                            rho0=cfg["rho0"], lambda_e=cfg["lambda_e"],
                            seed=seed, method=method, vis_audit=True)
                        res = run_episode(world, init, ep, gt=gt)
                        res["status"] = "ok"
                    except Exception as e:
                        res = {"status": "failed", "error": repr(e),
                               "history": [], "final": None}
                    res.update({"level": level, "env_prior_frac": frac,
                                "observed_env": obs_env, "method": method,
                                "scene": scene_name, "scene_seed": seed,
                                "config_hash": cfg_hash, "git_commit": commit,
                                "runtime_s": round(time.time() - t0, 1)})
                    res.pop("gt", None)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    run_path.write_text(json.dumps(res, indent=1))
                    fin = res.get("final") or {}
                    hist = res.get("history") or []
                    print(f"[e8]   {level:5s} {method:10s} "
                          f"known={fin.get('mplan_known_ratio', float('nan')):.3f} "
                          f"awc={fin.get('awc_gap_recovery', float('nan')):.3f} "
                          f"asset={fin.get('asset_recovery', float('nan')):.3f} "
                          f"inval={episode_mean(hist, 'invalid_view_frac'):.3f} "
                          f"st={fin.get('n_stations', -1)} "
                          f"stop={fin.get('stop_reason')} ({res['runtime_s']}s)",
                          flush=True)
                    del world
            summarize(out_root, cfg, levels, cfg_hash, commit)


def collect(out_root: Path, cfg: dict, levels: list) -> dict:
    data: dict = {}
    for level, _, _ in levels:
        for method in cfg["methods"]:
            cell = {m: {} for m, _ in METRICS}
            for family, density in cfg["scenes"]:
                for seed in cfg["seeds"]:
                    p = (out_root / level / method / f"scene_{family}_{density}"
                         / str(seed) / "run.json")
                    if not p.exists():
                        continue
                    r = json.loads(p.read_text())
                    if r.get("status") != "ok":
                        continue
                    key = (f"scene_{family}_{density}", seed)
                    for m, _ in METRICS:
                        cell[m][key] = (episode_mean(r["history"], m) if m in PER_ROUND
                                        else float(r["final"].get(m, float("nan"))))
            data[(level, method)] = cell
    return data


def paired(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    xa = np.array([a[k] for k in keys], dtype=float)
    xb = np.array([b[k] for k in keys], dtype=float)
    ok = np.isfinite(xa) & np.isfinite(xb)
    return xa[ok], xb[ok]


def summarize(out_root: Path, cfg: dict, levels: list, cfg_hash: str,
              commit: str) -> None:
    data = collect(out_root, cfg, levels)
    L = [l[0] for l in levels]
    out = ["# E8 frontier 探索基线对照\n",
           f"config_hash `{cfg_hash}` · commit `{commit}`\n",
           "两个轴必须一起读: `mplan_known_ratio` 是 frontier 的目标函数, "
           "`awc_gap_recovery`/`asset_recovery` 是本方法的目标函数。\n"]

    for level in L:
        out.append(f"\n## {level}\n")
        out.append("| 指标 | " + " | ".join(cfg["methods"]) + " |")
        out.append("|" + "---|" * (len(cfg["methods"]) + 1))
        for m, label in METRICS:
            cells = []
            for method in cfg["methods"]:
                v = [x for x in data.get((level, method), {}).get(m, {}).values()
                     if np.isfinite(x)]
                cells.append(f"{np.mean(v):.3f}" if v else "—")
            out.append(f"| {label} `{m}` | " + " | ".join(cells) + " |")
        n = [str(len(data.get((level, m), {}).get("awc_gap_recovery", {})))
             for m in cfg["methods"]]
        out.append("| n | " + " | ".join(n) + " |")

    out.append("\n## 本方法 vs frontier（配对置换检验 + Holm）\n")
    out.append("| 档 | 方法 | 指标 | Δ(本−frontier) | 95%CI | p_holm |")
    out.append("|---|---|---|---|---|---|")
    fam_p, fam_row = [], []
    for level in L:
        fr = data.get((level, "Bfrontier"), {})
        for method in ("B10_full", "B11_disc"):
            cur = data.get((level, method), {})
            for m, _ in METRICS:
                xa, xb = paired(cur.get(m, {}), fr.get(m, {}))
                if len(xa) < 3:
                    continue
                d = xa - xb
                lo, hi = bootstrap_ci(d, seed=0)
                fam_p.append(paired_permutation(xa, xb, seed=0))
                fam_row.append((level, method, m, float(np.mean(d)), lo, hi))
    if fam_p:
        for (lv, mt, m, d, lo, hi), ph in zip(fam_row, holm(fam_p)):
            out.append(f"| {lv} | {mt} | {m} | {d:+.4f} | "
                       f"[{lo:+.4f}, {hi:+.4f}] | {ph:.4f} |")

    # 等站数对照。frontier 按自己的目标函数提前收工(候选池里再没有能揭示未知的
    # 位置), 平均只用 3.9 站 / 101 m, 而本方法用满 5.8 站 / 132 m。不做这一节,
    # "本方法赢"就有靠多花预算赢的嫌疑 —— 这里把本方法的轨迹截到与 frontier
    # 相同的站数再比。
    out.append("\n## 等站数对照（把本方法截到 frontier 实际用掉的站数）\n")
    out.append("| 档 | 方法 | 资产恢复(截断) | frontier | Δ | 95%CI | p | 路径(截断) | frontier 路径 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for level in L:
        fr_runs = _runs(out_root, level, "Bfrontier", cfg)
        for method in ("B10_full", "B11_disc"):
            ours = _runs(out_root, level, method, cfg)
            keys = sorted(set(fr_runs) & set(ours))
            if len(keys) < 3:
                continue
            a, b, pa, pb = [], [], [], []
            for k in keys:
                n = int(fr_runs[k]["final"]["n_stations"])
                h = ours[k]["history"]
                i = min(n, len(h) - 1)
                a.append(h[i]["asset_recovery"]); pa.append(h[i]["path_len_m"])
                b.append(fr_runs[k]["final"]["asset_recovery"])
                pb.append(fr_runs[k]["final"]["path_len_m"])
            a, b = np.array(a), np.array(b)
            lo, hi = bootstrap_ci(a - b, seed=0)
            p = paired_permutation(a, b, seed=0)
            out.append(f"| {level} | {method} | {a.mean():.3f} | {b.mean():.3f} | "
                       f"{(a - b).mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.4f} | "
                       f"{np.mean(pa):.1f} m | {np.mean(pb):.1f} m |")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.md").write_text("\n".join(out) + "\n")


def _runs(out_root: Path, level: str, method: str, cfg: dict) -> dict:
    d = {}
    for family, density in cfg["scenes"]:
        for seed in cfg["seeds"]:
            p = (out_root / level / method / f"scene_{family}_{density}"
                 / str(seed) / "run.json")
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            if r.get("status") == "ok":
                d[(r["scene"], r["scene_seed"])] = r
    return d


if __name__ == "__main__":
    main()
