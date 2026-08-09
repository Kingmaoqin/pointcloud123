"""E5: 跨规模/密度的稳定性 —— 换一个变电站还能不能用。

此前 E2/E3 只覆盖 S/low、S/mid、M/mid 三种场景, 且对所有场景一律给 6 站 /
400 m 预算。实测这个固定预算占"完整扫描站数"的比例是:

    S/low 46%   S/mid 60%   S/high 60%
    M/low 24%   M/mid 29%   M/high 35%
    L/low 13%   L/mid 17%   L/high 17%

也就是说大场景被系统性地少给了三到四倍预算, 跨场景比较把"方法好坏"和"预算够
不够"混在了一起, 而 L 家族从未被测过。这不是理论顾虑 —— L/high seed 0 上的直接
反例:

    预算 6 站:  Bdisp awc 0.877 / asset 0.909  >  B10 awc 0.829 / asset 0.884
    预算 14 站: Bdisp awc 0.899 / asset 0.970  <  B10 awc 0.990 / asset 0.995
                                                  (且 B10 路径反而短 10%)

6 站恰好卡在"无信息启发式已经吃饱、B10 还没吃饱"的那个点上, 固定预算因此系统性
偏袒前重型启发式, 在大站上把结论读反。

E5 按**每固定数量的 BIM 构件给一站**分配预算:

    stations_max = max(3, round(n_components / COMPONENTS_PER_STATION))

锚在构件数而不是 n_full_stations: 后者是 8 m 网格里的自由格数, 场景越密自由格
越少, 于是越复杂的场景反而拿到越少的站 —— 实测按 n_full 分配时"每站负责的构件
数"在 5.0(L/low) 到 21.0(S/high) 之间差 4.2 倍, 且与场景难度反着走。按构件数
分配后该值稳定在 8.9–9.9。路径上限取每站 60 m, 宽到基本不绑定 —— 走多远的代价
由时间归一指标承担, 不必再用硬上限去压(注意 awc/1000s 与 B10 的内部成本同构,
见 OPEN_ISSUES #30, 不能作独立确认)。

稳定性看两件事: 各方法在 9 类场景上的**均值**, 以及**跨场景变异系数**与最差
场景。一个只在小场景好用的方法, 均值可能不难看, 变异系数会暴露它。

输出: results/exp5/{method}/{scene}/{seed}/run.json 与 summary.md
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

METHODS = ["Bdisp_maxmin", "B5_occ_rng", "B10_full"]
FAMILIES = ["S", "M", "L"]
DENSITIES = ["low", "mid", "high"]
SEEDS = [0, 1, 2]
# 每站负责多少个 BIM 构件。锚在构件数而不是 n_full_stations: 后者是 8 m 网格里
# 的自由格数, 场景越密自由格越少, 于是**越复杂的场景拿到越少的站** —— 实测按
# n_full 分配时"每站负责的构件数"在 5.0(L/low) 到 21.0(S/high) 之间差 4.2 倍,
# 且与场景难度反着走。按构件数分配则单调随规模与密度增长。
# 取 9.3 使 S/low(56 构件) 得到 6 站, 与既有 E2 协议对齐。
COMPONENTS_PER_STATION = 9.3
LEN_PER_STATION = 60.0    # m/站, 宽到基本不绑定
from patent_gap.planning.objective import (  # noqa: E402
    T_SCAN_DEFAULT as T_SCAN, V_MOVE_DEFAULT as V_MOVE,
)
KEYS = ["awc_gap_recovery", "asset_recovery", "crit_recall", "dens_ok"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/exp5"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out)

    cfg = {"families": FAMILIES, "densities": DENSITIES, "seeds": SEEDS,
           "methods": METHODS, "components_per_station": COMPONENTS_PER_STATION,
           "len_per_station": LEN_PER_STATION,
           "sensor": {"dtheta_deg": 0.4, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005},
           "rho0": 50.0, "init_stations": 3}
    if args.quick:
        cfg["families"], cfg["densities"], cfg["seeds"] = ["S"], ["low"], [0]
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    commit = git_commit()
    print(f"[e5] config_hash={cfg_hash} commit={commit}", flush=True)

    for family in cfg["families"]:
        for density in cfg["densities"]:
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
                n_full = int(gt["n_full_stations"])
                n_comp = int(world.patches["element_guid"].nunique())
                n_st = max(3, int(round(n_comp / cfg["components_per_station"])))
                len_max = cfg["len_per_station"] * n_st
                print(f"[e5] {scene_name}/seed{seed}: patches={len(world.patches)} "
                      f"gaps={int(np.sum(gt['y']))} n_full={n_full} "
                      f"budget={n_st}站/{len_max:.0f}m", flush=True)

                for method in cfg["methods"]:
                    run_dir = out_root / method / scene_name / str(seed)
                    run_path = run_dir / "run.json"
                    if run_path.exists():
                        continue
                    t0 = time.time()
                    try:
                        ep = EpisodeConfig(stations_max=n_st, length_max_m=len_max,
                                           rounds_max=n_st, rho0=cfg["rho0"],
                                           seed=seed, method=method)
                        res = run_episode(world, init, ep, gt=gt)
                        res["status"] = "ok"
                    except Exception as e:
                        res = {"status": "failed", "error": repr(e),
                               "history": [], "final": None}
                    res.update({"method": method, "family": family, "density": density,
                                "scene": scene_name, "scene_seed": seed,
                                "n_full_stations": n_full, "n_components": n_comp,
                                "stations_budget": n_st,
                                "n_patches": len(world.patches),
                                "config_hash": cfg_hash, "git_commit": commit,
                                "runtime_s": round(time.time() - t0, 1)})
                    res.pop("gt", None)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    run_path.write_text(json.dumps(res, indent=1))
                    fin = res.get("final") or {}
                    print(f"[e5]   {method:14s} awc={fin.get('awc_gap_recovery', float('nan')):.3f} "
                          f"asset={fin.get('asset_recovery', float('nan')):.3f} "
                          f"crit={fin.get('crit_recall', float('nan')):.3f} "
                          f"({res['runtime_s']}s)", flush=True)
                    summarize(out_root, cfg, cfg_hash, commit)   # 边跑边出表


def summarize(out_root: Path, cfg: dict, cfg_hash: str, commit: str) -> None:
    rows = []
    for method in cfg["methods"]:
        for family in cfg["families"]:
            for density in cfg["densities"]:
                for seed in cfg["seeds"]:
                    p = out_root / method / f"scene_{family}_{density}" / str(seed) / "run.json"
                    if not p.exists():
                        continue
                    r = json.loads(p.read_text())
                    fin = dict(r.get("final") or {})
                    if fin:
                        T = (float(fin.get("path_len_m", 0.0)) / V_MOVE
                             + float(fin.get("n_stations", 0)) * T_SCAN)
                        fin["awc_per_1000s"] = (fin["awc_gap_recovery"] / T * 1000.0
                                                if T > 0 else float("nan"))
                    rows.append({"method": method, "family": family,
                                 "density": density, "seed": seed,
                                 "scene": f"scene_{family}_{density}",
                                 "stations_budget": r.get("stations_budget"),
                                 "n_components": r.get("n_components"),
                                 "n_full_stations": r.get("n_full_stations"), **fin})
    if not rows:
        return

    lines = ["# E5: 跨规模/密度稳定性", "",
             f"- config_hash `{cfg_hash}`, commit `{commit}`, runs={len(rows)}",
             f"- 预算按每 {cfg['components_per_station']:.1f} 个 BIM 构件给 1 站。"
             f"固定 6 站时 L/high 只拿到完整扫描的 17%、S/low 拿到 46%, 大场景被"
             f"系统性少给三到四倍; 而按 n_full 分配又与场景复杂度反着走",
             "", "## 逐场景 awc 恢复率", "",
             "| 场景 | 构件数 | 完整站数 | 预算站数 | " + " | ".join(cfg["methods"]) + " |",
             "|---" * (4 + len(cfg["methods"])) + "|"]
    for family in cfg["families"]:
        for density in cfg["densities"]:
            sub = [r for r in rows if r["family"] == family and r["density"] == density]
            if not sub:
                continue
            nf = float(np.nanmean([r["n_full_stations"] for r in sub]))
            nc = float(np.nanmean([r["n_components"] for r in sub]))
            nb = float(np.nanmean([r["stations_budget"] for r in sub]))
            cells = []
            for m in cfg["methods"]:
                v = [r.get("awc_gap_recovery", float("nan"))
                     for r in sub if r["method"] == m]
                cells.append(f"{float(np.nanmean(v)):.3f}" if v else "—")
            lines.append(f"| {family}/{density} | {nc:.0f} | {nf:.0f} | {nb:.0f} | "
                         + " | ".join(cells) + " |")

    # 稳定性: 跨场景类型的均值/变异系数/最差场景
    lines += ["", "## 稳定性(按 9 类场景的**场景均值**统计)", "",
              "| 方法 | 指标 | 均值 | 标准差 | 变异系数 | 最差场景 |",
              "|---|---|---|---|---|---|"]
    for m in cfg["methods"]:
        for key in KEYS:
            per_scene = []
            for family in cfg["families"]:
                for density in cfg["densities"]:
                    v = [r.get(key, float("nan")) for r in rows
                         if r["method"] == m and r["family"] == family
                         and r["density"] == density]
                    if v and np.isfinite(np.nanmean(v)):
                        per_scene.append((f"{family}/{density}", float(np.nanmean(v))))
            if not per_scene:
                continue
            vals = np.array([v for _, v in per_scene])
            worst = min(per_scene, key=lambda t: t[1])
            cv = float(vals.std() / vals.mean()) if vals.mean() > 1e-9 else float("nan")
            lines.append(f"| {m} | {key} | {vals.mean():.3f} | {vals.std():.3f} "
                         f"| {cv:.3f} | {worst[0]} {worst[1]:.3f} |")

    # 逐规模的配对检验: 优势在大场景上是否还在
    idx = {(r["method"], r["scene"], r["seed"]): r for r in rows}
    lines += ["", "## B10 相对无信息基线 Bdisp 的优势, 逐规模", "",
              "| 规模 | 指标 | Δ均值 | 95%CI | p |", "|---|---|---|---|---|"]
    tests = []
    for family in cfg["families"]:
        for key in ("awc_gap_recovery", "asset_recovery"):
            a, b = [], []
            for density in cfg["densities"]:
                for seed in cfg["seeds"]:
                    k1 = ("B10_full", f"scene_{family}_{density}", seed)
                    k0 = ("Bdisp_maxmin", f"scene_{family}_{density}", seed)
                    if k0 in idx and k1 in idx:
                        a.append(idx[k1].get(key, float("nan")))
                        b.append(idx[k0].get(key, float("nan")))
            if len(a) < 2:
                continue
            x, y = np.asarray(a, float), np.asarray(b, float)
            tests.append({"family": family, "metric": key,
                          "delta": float(np.nanmean(x - y)),
                          "ci95": bootstrap_ci(x - y, n=5000),
                          "p_raw": paired_permutation(x, y, seed=0)})
    if tests:
        for t, adj in zip(tests, holm([t["p_raw"] for t in tests])):
            t["p_holm"] = float(adj) if np.isfinite(adj) else float("nan")
        for t in tests:
            lines.append("| {} | {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {:.4f} |".format(
                t["family"], t["metric"], t["delta"],
                t["ci95"][0], t["ci95"][1], t["p_holm"]))

    (out_root / "summary.md").write_text("\n".join(lines))
    (out_root / "summary.json").write_text(json.dumps(
        {"config_hash": cfg_hash, "git_commit": commit, "rows": rows, "tests": tests},
        indent=1, default=str))


if __name__ == "__main__":
    main()
