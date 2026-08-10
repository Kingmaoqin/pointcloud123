"""E7: 规划环境先验强度消融 —— 拿掉"现场完整模型"以后还剩多少。

合作方的意见是: 测试场景给了规划器过强的几何先验。此前 M_ref 与 M_plan 是同一份
几何 —— 可通行空间由完整参考模型的全部构件栅格化得到, 可见性求交也对该完整几何
进行。于是规划器在执行第一站之前, 就已经知道现场每一个障碍物在哪里。真实竣工测量
现场不具备这个条件。

E7 把先验强度作为自变量, 分五档:

    LEGACY  原方案。可通行图 = 完整参考模型栅格化; 遮挡求交 = 完整参考模型。
            与既有 E2 逐比特同构, 作为参照行。
    P100    遮挡先验仍为完整参考模型, 但**移动**改用观测驱动的 M_plan。
            单独隔离"可通行空间不再白给"这一项。
    P75     环境构件按对象随机保留 75% 进入遮挡先验(目标构件与地面始终保留)。
    P50     保留 50%。
    P0      环境构件一个不给。遮挡先验只剩目标表面与地面 —— 即"设计模型只描述
            要检查什么, 不描述现场长什么样"。

LEGACY→P100 与 P100→P0 分开是必要的: 二者是两条不同的信息通路(能不能走 / 看不
看得见), 混在一档里读不出是哪一条在起作用。

方法取两个:

    B10_full  完整价值函数, **不开**在线遮挡发现。先验给多少就是多少, 全程不改。
    B11_disc  B10 + 在线遮挡发现。

这一对是本实验的重点。在 P100 下, 现场几乎一切都已建模, 在线发现只能捡到临时占位
物, E3b 实测其恢复率增益不显著(p=0.50)。先验越弱, 遮挡模型里缺的东西越多, 该机制
要修正的错误也越多 —— 它的价值应当随先验减弱而增大。若果真如此, 则先前"不显著"
的结论不是机制无效, 而是 P100 这个测试条件本身把它的用武之地拿掉了。

真值与初始站在各先验档之间必须完全一致, 否则指标本身会随处理变动:

    gt   由 world.grid/world.oracle 与初始站决定, 三者都不随先验档变化。
         每个(场景,种子)只算一次, 传给全部档位。
    init 初始站是作业输入 —— 现场的初始扫描已经发生过, 是它产生了待补的缺口,
         不是本算法规划的。各档同一组, 不构成规划器的环境知识。
         (该项仍读完整模型, 已记入 06_Oracle_Leakage_Audit.md 的残留项。)

输出: results/exp7/{level}/{method}/{scene}/{seed}/run.json 与 summary.md
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

# (名称, env_prior_frac, observed_env)
LEVELS = [
    ("LEGACY", 1.0, False),
    ("P100",   1.0, True),
    ("P75",    0.75, True),
    ("P50",    0.5, True),
    ("P0",     0.0, True),
]
METHODS = ["B10_full", "B11_disc"]

# 聚合表里出现的全部指标。越界方向记在这里, 免得读表时反过来。
METRICS = [
    ("awc_gap_recovery", "up"), ("asset_recovery", "up"), ("crit_recall", "up"),
    ("dens_ok", "up"), ("dens_ok_vs_full", "up"), ("ig_per_m", "up"),
    ("path_len_m", "down"), ("n_stations", "down"),
    ("vis_mae", "down"), ("vis_over", "down"), ("invalid_view_frac", "down"),
    ("mplan_known_ratio", "up"),
]


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
    """轨迹里逐站记录的量(可见性误差等)取全程平均; 缺该键的轮次跳过。"""
    v = [h[key] for h in hist if key in h]
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp7")
    ap.add_argument("--quick", action="store_true",
                    help="smoke: 1 场景 1 种子 3 档 2 站")
    args = ap.parse_args()

    cfg = {
        "scenes": [("S", "low"), ("S", "mid"), ("M", "mid")],
        "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
        # 竣工态临时占位物。E2/E5 取 0(设计模型与现场完全一致), 那是本实验要
        # 检验的那个过强条件的一部分 —— 现场若与模型完全一致, 在线遮挡发现
        # 本就无事可做。取 E3b 的非零档 6, 使"模型里查不到的东西"确实存在。
        "n_temp": 6,
        "levels": [l[0] for l in LEVELS],
        "methods": METHODS,
        "sensor": {"dtheta_deg": 0.4, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005},
        "sim_dtheta_deg": 0.4,
        # 与 E2 同一预算口径, 使 LEGACY 行可与既有 E2 结果直接对照
        "budget": {"stations_max": 6, "length_max_m": 400.0, "rounds_max": 6},
        "init_stations": 3,
        "rho0": 50.0,
        "lambda_e": 1.0,
        "prior_seed_base": 1000,
    }
    levels = list(LEVELS)
    if args.quick:
        cfg["scenes"] = [("M", "mid")]
        cfg["seeds"] = [0]
        cfg["budget"] = {"stations_max": 2, "length_max_m": 400.0, "rounds_max": 2}
        levels = [LEVELS[0], LEVELS[1], LEVELS[4]]
        cfg["levels"] = [l[0] for l in levels]

    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    commit = git_commit()
    out_root = ROOT / args.out
    print(f"[e7] config_hash={cfg_hash} commit={commit}", flush=True)

    for family, density in cfg["scenes"]:
        for seed in cfg["seeds"]:
            scene_name = f"scene_{family}_{density}"
            scene = generate_scene(seed=seed, family=family, density=density,
                                   n_temp=cfg["n_temp"])
            sensor = SensorModel.from_config(cfg["sensor"])
            # 真值与初始站: 用 LEGACY 世界算一次, 各档共用
            w0 = SimWorld.build(scene, sensor, sim_dtheta_deg=cfg["sim_dtheta_deg"])
            init = default_init_stations(w0, n=cfg["init_stations"])
            probe = ObsState(world=w0)
            for k, o in enumerate(init):
                probe.add_station(o, f"init_{k}", seed=seed * 100 + k)
            gt = build_ground_truth(w0, list(probe.masks))
            print(f"[e7] {scene_name}/seed{seed}: patches={len(w0.patches)} "
                  f"gaps={int(np.sum(gt['y']))} init={len(init)}", flush=True)
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
                    # env 状态在 episode 中被就地更新, 每个 run 必须用新世界
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
                                "n_patches": len(world.patches),
                                "n_ref_tris": int(world.ref_tris.sum())
                                if world.ref_tris is not None else -1,
                                "config_hash": cfg_hash, "git_commit": commit,
                                "runtime_s": round(time.time() - t0, 1)})
                    res.pop("gt", None)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    run_path.write_text(json.dumps(res, indent=1))
                    fin = res.get("final") or {}
                    hist = res.get("history") or []
                    print(f"[e7]   {level:6s} {method:9s} "
                          f"awc={fin.get('awc_gap_recovery', float('nan')):.3f} "
                          f"asset={fin.get('asset_recovery', float('nan')):.3f} "
                          f"mae={episode_mean(hist, 'vis_mae'):.3f} "
                          f"inval={episode_mean(hist, 'invalid_view_frac'):.3f} "
                          f"known={fin.get('mplan_known_ratio', float('nan')):.3f} "
                          f"st={fin.get('n_stations', -1)} "
                          f"stop={fin.get('stop_reason')} ({res['runtime_s']}s)",
                          flush=True)
                    del world
            summarize(out_root, cfg, levels, cfg_hash, commit)


def collect(out_root: Path, cfg: dict, levels: list) -> dict:
    """(level, method) -> {metric: {(scene,seed): value}}"""
    data: dict = {}
    for level, _, _ in levels:
        for method in cfg["methods"]:
            cell: dict = {m: {} for m, _ in METRICS}
            cell["stop"] = {}
            for family, density in cfg["scenes"]:
                for seed in cfg["seeds"]:
                    p = (out_root / level / method / f"scene_{family}_{density}"
                         / str(seed) / "run.json")
                    if not p.exists():
                        continue
                    r = json.loads(p.read_text())
                    if r.get("status") != "ok":
                        continue
                    fin, hist = r["final"], r["history"]
                    key = (f"scene_{family}_{density}", seed)
                    for m, _ in METRICS:
                        if m in ("vis_mae", "vis_over", "invalid_view_frac"):
                            cell[m][key] = episode_mean(hist, m)
                        elif m == "mplan_known_ratio":
                            cell[m][key] = float(fin.get(m, 1.0))
                        else:
                            cell[m][key] = float(fin.get(m, float("nan")))
                    cell["stop"][key] = fin.get("stop_reason", "?")
            data[(level, method)] = cell
    return data


def paired(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(a) & set(b), key=lambda k: (k[0], k[1]))
    xa = np.array([a[k] for k in keys], dtype=float)
    xb = np.array([b[k] for k in keys], dtype=float)
    ok = np.isfinite(xa) & np.isfinite(xb)
    return xa[ok], xb[ok]


def summarize(out_root: Path, cfg: dict, levels: list, cfg_hash: str,
              commit: str) -> None:
    data = collect(out_root, cfg, levels)
    L = [l[0] for l in levels]
    out = [f"# E7 规划环境先验强度消融\n",
           f"config_hash `{cfg_hash}` · commit `{commit}` · "
           f"{len(cfg['scenes'])} 场景 × {len(cfg['seeds'])} 种子 "
           f"× {len(L)} 先验档 × {len(cfg['methods'])} 方法\n"]

    for method in cfg["methods"]:
        out.append(f"\n## {method}\n")
        head = "| 指标 | " + " | ".join(L) + " |"
        out.append(head)
        out.append("|" + "---|" * (len(L) + 1))
        for m, direction in METRICS:
            cells = []
            for level in L:
                v = list(data.get((level, method), {}).get(m, {}).values())
                v = [x for x in v if np.isfinite(x)]
                cells.append(f"{np.mean(v):.3f}" if v else "—")
            arrow = "↑" if direction == "up" else "↓"
            out.append(f"| {m} {arrow} | " + " | ".join(cells) + " |")
        n = [str(len(data.get((level, method), {}).get('awc_gap_recovery', {})))
             for level in L]
        out.append("| n | " + " | ".join(n) + " |")

    # 先验档之间的配对比较: 每个方法内部, 各档对 LEGACY
    out.append("\n## 先验减弱的代价（各档 vs LEGACY，配对置换检验 + Holm）\n")
    out.append("| 方法 | 档 | 指标 | Δ(档−LEGACY) | 95%CI | p_holm |")
    out.append("|---|---|---|---|---|---|")
    for method in cfg["methods"]:
        base = data.get(("LEGACY", method), {})
        fam_p, fam_row = [], []
        for level in L[1:]:
            cur = data.get((level, method), {})
            for m in ("awc_gap_recovery", "asset_recovery", "vis_over",
                      "invalid_view_frac"):
                xb, xa = paired(cur.get(m, {}), base.get(m, {}))
                if len(xb) < 3:
                    continue
                d = xb - xa
                lo, hi = bootstrap_ci(d, seed=0)
                p = paired_permutation(xb, xa, seed=0)
                fam_p.append(p)
                fam_row.append((method, level, m, float(np.mean(d)), lo, hi))
        if not fam_p:
            continue
        for (mt, lv, m, d, lo, hi), ph in zip(fam_row, holm(fam_p)):
            out.append(f"| {mt} | {lv} | {m} | {d:+.4f} | "
                       f"[{lo:+.4f}, {hi:+.4f}] | {ph:.4f} |")

    # 本实验的重点: 在线遮挡发现的价值是否随先验减弱而增大
    out.append("\n## 在线遮挡发现的价值 vs 先验强度（B11_disc − B10_full）\n")
    out.append("| 档 | 指标 | Δ(B11−B10) | 95%CI | p_holm |")
    out.append("|---|---|---|---|---|")
    fam_p, fam_row = [], []
    for level in L:
        a = data.get((level, "B11_disc"), {})
        b = data.get((level, "B10_full"), {})
        for m in ("awc_gap_recovery", "asset_recovery", "vis_over",
                  "invalid_view_frac"):
            xa, xb = paired(a.get(m, {}), b.get(m, {}))
            if len(xa) < 3:
                continue
            d = xa - xb
            lo, hi = bootstrap_ci(d, seed=0)
            fam_p.append(paired_permutation(xa, xb, seed=0))
            fam_row.append((level, m, float(np.mean(d)), lo, hi))
    if fam_p:
        for (lv, m, d, lo, hi), ph in zip(fam_row, holm(fam_p)):
            out.append(f"| {lv} | {m} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {ph:.4f} |")

    # 终止原因分布: 弱先验下"候选池空"会不会成为主要死法
    out.append("\n## 终止原因\n")
    out.append("| 方法 | 档 | " + " | ".join(
        ["rounds", "stations", "length", "no_candidate"]) + " |")
    out.append("|---|---|---|---|---|---|")
    for method in cfg["methods"]:
        for level in L:
            st = list(data.get((level, method), {}).get("stop", {}).values())
            cells = [str(st.count(k)) for k in
                     ("rounds", "stations", "length", "no_candidate")]
            out.append(f"| {method} | {level} | " + " | ".join(cells) + " |")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
