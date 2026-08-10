"""E7/E8 曲线图。Phase 5 要求「表格；统计；曲线」，表与统计在 summary.md，曲线在此。

四张图:

  e7_recovery      任务恢复率 vs 先验强度(B10 / B11 两条线, 均值 ± 95% bootstrap CI)
  e7_vis_error     可见性预测误差 vs 先验强度 —— 本实验的重点。若在线遮挡发现
                   清除的误差量随先验减弱而增大, 两条线的间距应当逐档拉开。
  e7_known_ratio   M_plan 已确认栅格比例逐轮变化 —— 观测驱动的地图确实在长
  e8_two_axes      frontier 与本方法在两个不同目标函数上的位置(散点, 每点一档)

用法: python scripts/plot_e7.py [--e7 results/exp7] [--e8 results/exp8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
from matplotlib.font_manager import FontProperties   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from patent_gap.evaluation.stats import bootstrap_ci   # noqa: E402

_CN = None
for cand in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/arphic/uming.ttc"):
    if Path(cand).exists():
        _CN = FontProperties(fname=cand)
        break


def L(cn: str, en: str) -> str:
    return cn if _CN is not None else en


def _fp(**kw):
    return {"fontproperties": _CN, **kw} if _CN is not None else kw


def load(root: Path) -> list[dict]:
    rows = []
    for p in root.rglob("run.json"):
        r = json.loads(p.read_text())
        if r.get("status") != "ok" or not r.get("final"):
            continue
        hist = r.get("history") or []
        rows.append({
            "level": r.get("level"), "method": r.get("method"),
            "scene": r.get("scene"), "seed": r.get("scene_seed"),
            "history": hist, **r["final"],
            "vis_mae": _hmean(hist, "vis_mae"),
            "vis_over": _hmean(hist, "vis_over"),
            "invalid_view_frac": _hmean(hist, "invalid_view_frac"),
        })
    return rows


def _hmean(hist, key):
    v = [h[key] for h in hist if key in h]
    return float(np.mean(v)) if v else float("nan")


def pick(rows, level, method, key):
    v = [r[key] for r in rows
         if r["level"] == level and r["method"] == method and np.isfinite(r.get(key, np.nan))]
    return np.array(v, dtype=float)


def mean_ci(v):
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = bootstrap_ci(v, seed=0)
    return float(np.mean(v)), lo, hi


def line_with_ci(ax, levels, rows, method, key, color, label, marker):
    m, lo, hi = [], [], []
    for lv in levels:
        a, b, c = mean_ci(pick(rows, lv, method, key))
        m.append(a); lo.append(b); hi.append(c)
    x = np.arange(len(levels))
    ax.plot(x, m, marker=marker, color=color, label=label, lw=1.6, ms=5)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    return m


def fig_recovery(rows, levels, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, key, title in (
            (axes[0], "awc_gap_recovery", L("面积加权缺口恢复率", "area-weighted recovery")),
            (axes[1], "asset_recovery", L("资产级恢复率", "asset recovery"))):
        line_with_ci(ax, levels, rows, "B10_full", key, "#1f77b4",
                     L("B10 不开在线发现", "B10 no discovery"), "o")
        line_with_ci(ax, levels, rows, "B11_disc", key, "#d62728",
                     L("B11 开在线发现", "B11 + discovery"), "s")
        ax.set_xticks(np.arange(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_xlabel(L("规划环境先验强度（左强右弱）", "prior strength"), **_fp(fontsize=9))
        ax.set_title(title, **_fp(fontsize=10))
        ax.grid(alpha=0.3, lw=0.5)
        leg = ax.legend(fontsize=8)
        if _CN is not None:
            for t in leg.get_texts():
                t.set_fontproperties(_CN)
    fig.tight_layout()
    fig.savefig(out / "e7_recovery.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_vis_error(rows, levels, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, key, title in (
            (axes[0], "vis_over", L("可见性单向高估量", "one-sided over-prediction")),
            (axes[1], "invalid_view_frac", L("无效视点占比", "invalid viewpoint fraction"))):
        a = line_with_ci(ax, levels, rows, "B10_full", key, "#1f77b4",
                         L("B10 不开在线发现", "B10 no discovery"), "o")
        b = line_with_ci(ax, levels, rows, "B11_disc", key, "#d62728",
                         L("B11 开在线发现", "B11 + discovery"), "s")
        # 两条线的间距 = 在线发现清除掉的误差量
        x = np.arange(len(levels))
        ax.vlines(x, b, a, color="0.45", lw=0.9, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(levels)
        ax.set_xlabel(L("规划环境先验强度（左强右弱）", "prior strength"), **_fp(fontsize=9))
        ax.set_title(title, **_fp(fontsize=10))
        ax.grid(alpha=0.3, lw=0.5)
        leg = ax.legend(fontsize=8)
        if _CN is not None:
            for t in leg.get_texts():
                t.set_fontproperties(_CN)
    fig.tight_layout()
    fig.savefig(out / "e7_vis_error.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_known_ratio(rows, levels, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    cmap = plt.get_cmap("viridis")
    obs_levels = [lv for lv in levels if lv != "LEGACY"]
    for k, lv in enumerate(obs_levels):
        series = []
        for r in rows:
            if r["level"] != lv or r["method"] != "B10_full":
                continue
            v = [h["mplan_known_ratio"] for h in r["history"] if "mplan_known_ratio" in h]
            if v:
                series.append(v)
        if not series:
            continue
        n = min(len(s) for s in series)
        arr = np.array([s[:n] for s in series], dtype=float)
        ax.plot(np.arange(1, n + 1), arr.mean(axis=0), marker="o", ms=4,
                color=cmap(k / max(len(obs_levels) - 1, 1)), label=lv)
    ax.set_xlabel(L("补充扫描轮次", "round"), **_fp(fontsize=9))
    ax.set_ylabel(L("M_plan 已确认栅格比例", "M_plan known ratio"), **_fp(fontsize=9))
    ax.set_title(L("观测驱动的规划环境模型逐轮增长", "M_plan grows per round"),
                 **_fp(fontsize=10))
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "e7_known_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_two_axes(rows8, out: Path) -> None:
    if not rows8:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    style = {"Bfrontier": ("#2ca02c", "^", L("frontier 探索基线", "frontier")),
             "B10_full": ("#1f77b4", "o", "B10"),
             "B11_disc": ("#d62728", "s", "B11")}
    levels = sorted({r["level"] for r in rows8})
    for method, (c, mk, lab) in style.items():
        xs, ys = [], []
        for lv in levels:
            x = pick(rows8, lv, method, "mplan_known_ratio")
            y = pick(rows8, lv, method, "asset_recovery")
            if len(x) and len(y):
                xs.append(np.mean(x)); ys.append(np.mean(y))
        if xs:
            ax.scatter(xs, ys, c=c, marker=mk, s=60, label=lab, zorder=3)
            for x, y, lv in zip(xs, ys, levels):
                ax.annotate(lv, (x, y), textcoords="offset points",
                            xytext=(5, 4), fontsize=7.5)
    ax.set_xlabel(L("地图覆盖 —— frontier 的目标函数",
                    "map coverage (frontier's objective)"), **_fp(fontsize=9))
    ax.set_ylabel(L("资产恢复率 —— 本方法的目标函数",
                    "asset recovery (our objective)"), **_fp(fontsize=9))
    ax.set_title(L("两个不同的目标函数", "two different objectives"), **_fp(fontsize=10))
    ax.grid(alpha=0.3, lw=0.5)
    leg = ax.legend(fontsize=8, loc="best")
    if _CN is not None:
        for t in leg.get_texts():
            t.set_fontproperties(_CN)
    fig.tight_layout()
    fig.savefig(out / "e8_two_axes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e7", default="results/exp7")
    ap.add_argument("--e8", default="results/exp8")
    args = ap.parse_args()

    e7 = ROOT / args.e7
    out = e7 / "figs"
    out.mkdir(parents=True, exist_ok=True)
    rows = load(e7)
    if not rows:
        raise SystemExit(f"{e7} 下没有可用结果")
    order = ["LEGACY", "P100", "P75", "P50", "P0"]
    levels = [lv for lv in order if any(r["level"] == lv for r in rows)]
    print(f"[plot] E7 {len(rows)} 次运行, 档位 {levels}")
    fig_recovery(rows, levels, out)
    fig_vis_error(rows, levels, out)
    fig_known_ratio(rows, levels, out)

    e8 = ROOT / args.e8
    rows8 = load(e8) if e8.exists() else []
    print(f"[plot] E8 {len(rows8)} 次运行")
    fig_two_axes(rows8, out)
    print(f"[plot] 输出目录 {out}")


if __name__ == "__main__":
    main()
