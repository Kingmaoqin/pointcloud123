"""E3b: 只答 H4 —— 在线发现未建模遮挡物能挽回多少。

与 E3 分开跑, 因为 E3 的进程启动于 commit 8d90756, 既不含 B11 也不含
asset_recovery(167 个 run.json 无一有该字段)。在那个混版本目录上续跑, H4 表的
B10 一侧会整列 nan。此处新目录、单一 HEAD、只跑 B10 与 B11 的配对。

主指标是 asset_recovery(构件等权恢复率) —— 临时占位物挡住的多是单体设备, 而
awc 由少数大平面主导, 对此不敏感。检验族 = 各 n_temp 水平一族, 不混入 awc 的
各种变换。

原 E3 的说明保留如下。

E3: BIM 与竣工实景偏离程度对遮挡感知规划的影响。

动机。E2 里规划器与评测器共用同一套几何, 公式(27)(28) 的 Vis 因此不是预测,
而是它事后被对照的那个真值本身 —— "遮挡感知有效"在那种设置下无法被证伪。
generate_scene(n_temp=k) 在道路与设备之间放 k 个 BIM 查不到的临时占位物
(车辆/料堆), 规划器只对 BIM 求交, 仿真与评测对完整竣工几何求交。

要检验的三个可证伪命题:
  H1 遮挡感知方法(B5/B10) 随 n_temp 增大而退化 —— 若不退化, 说明遮挡项本来
     就没在起作用, E2 中它的优势另有来源。
  H2 遮挡盲方法(B1) 对 n_temp 基本不敏感 —— 这是实验本身的对照, 若 B1 也随
     n_temp 大幅波动, 那变化来自场景难度而非预测误差, H1 的结论不成立。
  H3 B10 相对 Bdisp(不用任何几何证据) 的优势随 n_temp 收窄。收窄多少决定了
     "遮挡感知"这项主张能在多大的 BIM 失配下继续成立。
  H4 B11(B10 + 未建模遮挡物在线发现) 的退化显著小于 B10。这是针对 H1 所测退化
     提出的补救: 遮挡物本身会被扫到, 无法关联到任何 BIM 构件的回波就暴露了它
     们的位置。n_temp=0 时 B11 必须与 B10 逐位相同(见 tests/)。

输出: results/exp3/{n_temp}/{method}/{scene}/{seed}/run.json 与 summary.md
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

METHODS = ["B10_full", "B11_disc"]
N_TEMP = [0, 6]
SCENES = [("S", "low"), ("S", "mid"), ("M", "mid")]
SEEDS = [0, 1, 2, 3, 4]
METRICS = ["awc_gap_recovery", "crit_recall", "dens_ok", "path_len_m", "vis_mae"]

from patent_gap.planning.objective import (  # noqa: E402
    T_SCAN_DEFAULT as T_SCAN, V_MOVE_DEFAULT as V_MOVE,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def visibility_mae(world, origins) -> float:
    """规划器(BIM)与实景之间的单站可见性预测误差, 用于标定偏离强度。

    直接测量而非用 n_temp 代指: 同样放 6 个占位物, 挡没挡住视线差别很大。
    """
    if world.plan_oracle is world.oracle:
        return 0.0
    pids = [int(x) for x in world.patches["patch_id"].to_numpy()]
    errs = []
    for o in origins:
        pred = world.plan_oracle.visibility_batch(o, world.sampler, pids, sensor=world.sensor)
        real = world.oracle.visibility_batch(o, world.sampler, pids, sensor=world.sensor)
        errs.append(np.abs(np.array([pred[p] - real[p] for p in pids])).mean())
    return float(np.mean(errs)) if errs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/exp3b"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out)

    cfg = {"scenes": SCENES, "seeds": SEEDS, "methods": METHODS, "n_temp": N_TEMP,
           "sensor": {"dtheta_deg": 0.4, "r": [0.5, 10.0, 60.0], "sigma_r": 0.005},
           "budget": {"stations_max": 6, "length_max_m": 400.0, "rounds_max": 6},
           "rho0": 50.0, "init_stations": 3}
    if args.quick:
        cfg["scenes"], cfg["seeds"], cfg["n_temp"] = [("S", "low")], [0], [0, 6]
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    commit = git_commit()
    print(f"[e3] config_hash={cfg_hash} commit={commit}", flush=True)

    for n_temp in cfg["n_temp"]:
        for family, density in cfg["scenes"]:
            for seed in cfg["seeds"]:
                scene_name = f"scene_{family}_{density}"
                scene = generate_scene(seed=seed, family=family, density=density,
                                       n_temp=n_temp)
                sensor = SensorModel.from_config(cfg["sensor"])
                world = SimWorld.build(scene, sensor,
                                       sim_dtheta_deg=cfg["sensor"]["dtheta_deg"])
                init = default_init_stations(world, n=cfg["init_stations"])
                probe = ObsState(world=world)
                for k, o in enumerate(init):
                    probe.add_station(o, f"init_{k}", seed=seed * 100 + k)
                gt = build_ground_truth(world, list(probe.masks))
                vmae = visibility_mae(world, init)
                print(f"[e3] n_temp={n_temp} {scene_name}/seed{seed}: "
                      f"patches={len(world.patches)} gaps={int(np.sum(gt['y']))} "
                      f"vis_mae={vmae:.4f}", flush=True)

                for method in cfg["methods"]:
                    run_dir = out_root / f"n{n_temp}" / method / scene_name / str(seed)
                    run_path = run_dir / "run.json"
                    if run_path.exists():
                        continue
                    t0 = time.time()
                    try:
                        ep = EpisodeConfig(
                            stations_max=cfg["budget"]["stations_max"],
                            length_max_m=cfg["budget"]["length_max_m"],
                            rounds_max=cfg["budget"]["rounds_max"],
                            rho0=cfg["rho0"], seed=seed, method=method)
                        res = run_episode(world, init, ep, gt=gt)
                        res["status"] = "ok"
                    except Exception as e:
                        res = {"status": "failed", "error": repr(e),
                               "history": [], "final": None}
                    res.update({"method": method, "n_temp": n_temp, "vis_mae": vmae,
                                "scene": scene_name, "scene_seed": seed,
                                "config_hash": cfg_hash, "git_commit": commit,
                                "runtime_s": round(time.time() - t0, 1)})
                    res.pop("gt", None)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    run_path.write_text(json.dumps(res, indent=1))
                    fin = res.get("final") or {}
                    print(f"[e3]   {method:14s} awc={fin.get('awc_gap_recovery', float('nan')):.3f} "
                          f"crit={fin.get('crit_recall', float('nan')):.3f} "
                          f"({res['runtime_s']}s)", flush=True)

    summarize(out_root, cfg, cfg_hash, commit)


def summarize(out_root: Path, cfg: dict, cfg_hash: str, commit: str) -> None:
    rows = []
    for n_temp in cfg["n_temp"]:
        for method in cfg["methods"]:
            for family, density in cfg["scenes"]:
                for seed in cfg["seeds"]:
                    p = (out_root / f"n{n_temp}" / method /
                         f"scene_{family}_{density}" / str(seed) / "run.json")
                    if not p.exists():
                        continue
                    r = json.loads(p.read_text())
                    fin = dict(r.get("final") or {})
                    if fin:
                        T = (float(fin.get("path_len_m", 0.0)) / V_MOVE
                             + float(fin.get("n_stations", 0)) * T_SCAN)
                        fin["T_total_s"] = T
                        fin["awc_per_1000s"] = (fin["awc_gap_recovery"] / T * 1000.0
                                                if T > 0 else float("nan"))
                    rows.append({"method": method, "n_temp": n_temp,
                                 "scene": f"scene_{family}_{density}", "seed": seed,
                                 "vis_mae": r.get("vis_mae", float("nan")), **fin})
    if not rows:
        return

    lines = ["# E3: BIM 与竣工实景偏离对遮挡感知规划的影响", "",
             f"- config_hash `{cfg_hash}`, commit `{commit}`, runs={len(rows)}",
             "- n_temp = 场景中 BIM 查不到的临时占位物数量; vis_mae = 规划器可见性",
             "  预测相对实景的平均绝对误差(直接测量的偏离强度)", "",
             "## 各偏离水平下的表现", "",
             "| n_temp | vis_mae | 方法 | awc恢复率 | 构件等权恢复 | 关键设备召回 | dens_ok | awc/1000s |",
             "|---|---|---|---|---|---|---|---|"]
    for n_temp in cfg["n_temp"]:
        sub_all = [r for r in rows if r["n_temp"] == n_temp]
        vm = float(np.nanmean([r["vis_mae"] for r in sub_all])) if sub_all else float("nan")
        for method in cfg["methods"]:
            sub = [r for r in sub_all if r["method"] == method]
            if not sub:
                continue
            def mu(k):
                return float(np.nanmean([r.get(k, float("nan")) for r in sub]))
            lines.append(f"| {n_temp} | {vm:.4f} | {method} | {mu('awc_gap_recovery'):.3f} "
                         f"| {mu('asset_recovery'):.3f} | {mu('crit_recall'):.3f} "
                         f"| {mu('dens_ok'):.3f} | {mu('awc_per_1000s'):.4f} |")

    # H1/H2: 每种方法从 n_temp=0 到最大偏离的退化幅度(同场景同种子配对)
    idx = {(r["method"], r["n_temp"], r["scene"], r["seed"]): r for r in rows}
    n_hi = max(cfg["n_temp"])
    lines += ["", f"## H1/H2 退化幅度 (n_temp=0 → {n_hi}, 同场景同种子配对)", "",
              "| 方法 | Δawc | 95%CI | p |", "|---|---|---|---|"]
    deg_tests = []
    for method in cfg["methods"]:
        a, b = [], []
        for family, density in cfg["scenes"]:
            for seed in cfg["seeds"]:
                k0 = (method, 0, f"scene_{family}_{density}", seed)
                k1 = (method, n_hi, f"scene_{family}_{density}", seed)
                if k0 in idx and k1 in idx:
                    a.append(idx[k1].get("awc_gap_recovery", float("nan")))
                    b.append(idx[k0].get("awc_gap_recovery", float("nan")))
        if not a:
            continue
        d = np.asarray(a, float) - np.asarray(b, float)
        deg_tests.append({"method": method, "delta": float(np.nanmean(d)),
                          "ci95": bootstrap_ci(d, n=5000),
                          "p_raw": paired_permutation(np.asarray(a, float),
                                                      np.asarray(b, float), seed=0)})
    for t in holm_attach(deg_tests):
        lines.append("| {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {:.4f} |".format(
            t["method"], t["delta"], t["ci95"][0], t["ci95"][1], t["p_holm"]))

    # H3: 每个偏离水平上 B10 相对 Bdisp 的优势
    lines += ["", "## H3 B10 相对无信息基线 Bdisp 的优势随偏离的变化", "",
              "| n_temp | Δawc | 95%CI | p |", "|---|---|---|---|"]
    adv_tests = []
    for n_temp in cfg["n_temp"]:
        a, b = [], []
        for family, density in cfg["scenes"]:
            for seed in cfg["seeds"]:
                k1 = ("B10_full", n_temp, f"scene_{family}_{density}", seed)
                k0 = ("Bdisp_maxmin", n_temp, f"scene_{family}_{density}", seed)
                if k0 in idx and k1 in idx:
                    a.append(idx[k1].get("awc_gap_recovery", float("nan")))
                    b.append(idx[k0].get("awc_gap_recovery", float("nan")))
        if not a:
            continue
        d = np.asarray(a, float) - np.asarray(b, float)
        adv_tests.append({"method": f"n_temp={n_temp}", "delta": float(np.nanmean(d)),
                          "ci95": bootstrap_ci(d, n=5000),
                          "p_raw": paired_permutation(np.asarray(a, float),
                                                      np.asarray(b, float), seed=0)})
    for t in holm_attach(adv_tests):
        lines.append("| {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {:.4f} |".format(
            t["method"].replace("n_temp=", ""), t["delta"],
            t["ci95"][0], t["ci95"][1], t["p_holm"]))

    # H4: 在线发现未建模遮挡物能挽回多少 —— 逐偏离水平比 B11 与 B10
    lines += ["", "## H4 在线发现未建模遮挡物 (B11 − B10, 构件等权恢复)", "",
              "| n_temp | Δasset | 95%CI | p |", "|---|---|---|---|"]
    disc_tests = []
    for n_temp in cfg["n_temp"]:
        a, b = [], []
        for family, density in cfg["scenes"]:
            for seed in cfg["seeds"]:
                k1 = ("B11_disc", n_temp, f"scene_{family}_{density}", seed)
                k0 = ("B10_full", n_temp, f"scene_{family}_{density}", seed)
                if k0 in idx and k1 in idx:
                    a.append(idx[k1].get("asset_recovery", float("nan")))
                    b.append(idx[k0].get("asset_recovery", float("nan")))
        if not a:
            continue
        d = np.asarray(a, float) - np.asarray(b, float)
        disc_tests.append({"method": f"n_temp={n_temp}", "delta": float(np.nanmean(d)),
                           "ci95": bootstrap_ci(d, n=5000),
                           "p_raw": paired_permutation(np.asarray(a, float),
                                                       np.asarray(b, float), seed=0)})
    for t in holm_attach(disc_tests):
        lines.append("| {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {:.4f} |".format(
            t["method"].replace("n_temp=", ""), t["delta"],
            t["ci95"][0], t["ci95"][1], t["p_holm"]))

    lines += ["", "> 配对置换检验在 15 对样本下的最小可能 p 为 2/2^15 = 6.1e-5;",
              "> 各族内 Holm 校正。报告不显著时须一并考虑该分辨率下限。"]

    (out_root / "summary.md").write_text("\n".join(lines))
    (out_root / "summary.json").write_text(json.dumps(
        {"config_hash": cfg_hash, "git_commit": commit, "rows": rows,
         "degradation": deg_tests, "advantage": adv_tests,
         "discovery": disc_tests}, indent=1, default=str))
    print(f"[e3] summary → {out_root/'summary.md'}", flush=True)


def holm_attach(tests: list[dict]) -> list[dict]:
    """在同一族内做 Holm 校正并回填 p_holm。"""
    if not tests:
        return tests
    for t, a in zip(tests, holm([t["p_raw"] for t in tests])):
        t["p_holm"] = float(a) if np.isfinite(a) else float("nan")
    return tests


if __name__ == "__main__":
    main()
