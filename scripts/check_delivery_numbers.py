"""交付包 headline 数字核查: 每个数字回原始 run.json 重算, 并检查其比较对象。

本轮加这个脚本, 是因为出过一次实打实的错误 —— frontier vs B10 的 p 值被写在了
frontier vs B11 的句子里。数字对不对是一回事, 数字**跟谁比**是另一回事, 后者
靠通读容易漏。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from patent_gap.evaluation.stats import bootstrap_ci, holm, paired_permutation  # noqa: E402


def runs(exp, level, method):
    d = {}
    for p in (ROOT / "results" / exp / level / method).rglob("run.json"):
        r = json.loads(p.read_text())
        if r.get("status") == "ok":
            d[(r["scene"], r["scene_seed"])] = r
    return d


def mean_of(d, key, per_round=False):
    v = []
    for r in d.values():
        if per_round:
            xs = [h[key] for h in r["history"] if key in h]
            if xs: v.append(float(np.mean(xs)))
        else:
            v.append(float(r["final"][key]))
    return float(np.mean(v)) if v else float("nan")


def paired_p(a, b, key, per_round=False):
    ks = sorted(set(a) & set(b))
    def g(r):
        if per_round:
            xs = [h[key] for h in r["history"] if key in h]
            return float(np.mean(xs)) if xs else np.nan
        return float(r["final"][key])
    xa = np.array([g(a[k]) for k in ks]); xb = np.array([g(b[k]) for k in ks])
    m = np.isfinite(xa) & np.isfinite(xb)
    return float(np.mean(xa[m] - xb[m])), paired_permutation(xa[m], xb[m], seed=0)


FAIL = []
def chk(label, got, want, tol=5e-4):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'FAIL'}  {label:52s} 文档={want:<9.4f} 重算={got:.4f}")
    if not ok: FAIL.append(label)


print("E8 P0 —— 均值")
fr = runs("exp8", "P0", "Bfrontier"); b10 = runs("exp8", "P0", "B10_full"); b11 = runs("exp8", "P0", "B11_disc")
chk("frontier 地图覆盖", mean_of(fr, "mplan_known_ratio"), 0.952)
chk("B11 地图覆盖", mean_of(b11, "mplan_known_ratio"), 0.949)
chk("frontier 资产恢复", mean_of(fr, "asset_recovery"), 0.523)
chk("B11 资产恢复", mean_of(b11, "asset_recovery"), 0.922)
chk("frontier 路径 m", mean_of(fr, "path_len_m"), 166.0, tol=0.05)
chk("B11 路径 m", mean_of(b11, "path_len_m"), 155.5, tol=0.05)

print("\nE8 —— 地图覆盖的两个 p 值（必须复现 summarize 的同一检验族）")
# summarize 的族: 2 档 × 2 方法 × 10 指标 = 40 项。族大小不同, Holm 结果就不同,
# 故此处必须逐位照抄它的构族顺序, 不能只取 P0 的一半。
METRICS = [("mplan_known_ratio", False), ("awc_gap_recovery", False),
           ("asset_recovery", False), ("crit_recall", False), ("dens_ok", False),
           ("path_len_m", False), ("n_stations", False),
           ("vis_mae", True), ("vis_over", True), ("invalid_view_frac", True)]
fam_p, fam_key = [], []
for lv in ("P50", "P0"):
    f = runs("exp8", lv, "Bfrontier")
    for meth in ("B10_full", "B11_disc"):
        cur = runs("exp8", lv, meth)
        for mname, per in METRICS:
            _, pv = paired_p(cur, f, mname, per)
            fam_p.append(pv); fam_key.append((lv, meth, mname))
adj = holm(fam_p)
look = {k: float(v) for k, v in zip(fam_key, adj)}
chk("B10 vs frontier 地图覆盖 p_holm (P0)", look[("P0","B10_full","mplan_known_ratio")], 0.0252, tol=2e-3)
chk("B11 vs frontier 地图覆盖 p_holm (P0)", look[("P0","B11_disc","mplan_known_ratio")], 0.1300, tol=2e-3)
chk("B11 vs frontier 地图覆盖 p_holm (P50)", look[("P50","B11_disc","mplan_known_ratio")], 0.0649, tol=2e-3)
chk("B11 vs frontier 资产恢复 p_holm (P0)", look[("P0","B11_disc","asset_recovery")], 0.0040, tol=2e-3)

print("\nE7 —— 先验强度")
for lv, w_b10, w_b11 in (("LEGACY", 0.926, 0.949), ("P100", 0.922, 0.934), ("P0", 0.830, 0.922)):
    chk(f"{lv} B10 资产恢复", mean_of(runs("exp7", lv, "B10_full"), "asset_recovery"), w_b10)
    chk(f"{lv} B11 资产恢复", mean_of(runs("exp7", lv, "B11_disc"), "asset_recovery"), w_b11)

print("\n" + ("全部通过" if not FAIL else f"未通过 {len(FAIL)} 项: {FAIL}"))
sys.exit(1 if FAIL else 0)
