"""按 2026-07-28 代理人批注重画附图 图4 / 图5 / 图8。

批注对应关系：
  批注6  图4：图上"满足两项约束"与说明"三项条件"矛盾；四个三角面用同一阴影，
              看不出哪几个被并入同一表面分块 → 区分填充 + 图上文字明确"共享边邻接
              和两项法向约束" + 画出真实共享边。
  批注7/8 图5：图中未画IFC表面，"刚性对齐"无变换前后对照，竖直虚线无法表达
              "到表面的距离阈值"，"边界内侧"指代不明 → 画出IFC表面实线、两侧
              ±tau_d 平行虚线、带内实心点/带外空心点、对齐前虚影+对齐箭头。
  批注9  图8：三个候选位姿未落在三层弧线上，标题与弧线、标签之间文字重叠，
              "三层距离"孤立 → 每层弧线上各放候选位姿，引线标注，清理碰撞。

风格与既有十张附图一致：黑白线稿、左上角粗体"图N  标题"、Noto Sans CJK。
输出：outputs/patent_figures/formal_bw/ 下 png(350dpi) + svg。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

ROOT = Path("/home/xqin5/patent_gap_nbv")
OUT = ROOT / "outputs" / "patent_figures" / "formal_bw"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    CJK = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    CJK = "DejaVu Sans"
plt.rcParams["font.family"] = CJK
plt.rcParams["font.sans-serif"] = [CJK, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

LW = 2.0
GRAY = "0.82"
DARK = "0.62"


def new_ax(title: str, figsize=(9.6, 5.9)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.045, 0.94, title, ha="left", va="center", fontsize=19, fontweight="bold")
    return fig, ax


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.png", dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {name}")


def arrow(ax, p, q, lw=LW, color="black", style="-|>", ls="-", ms=16):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- 图4
def fig4():
    """共享边和双法向约束形成表面分块。

    T1-T3 依次通过共享边相连且法向一致 → 同一表面分块（深填充）；
    T4 与 T3 共享边但法向偏转过大 → 不并入（空白填充）。
    """
    fig, ax = new_ax("图4  共享边和双法向约束形成表面分块")

    # 真正共享边的三角面带：相邻三角形严格共用一条边
    y0, h, w = 0.44, 0.185, 0.155
    xL = 0.135
    tris = [
        [(xL, y0), (xL + w, y0), (xL + w / 2, y0 + h)],                      # T1 上
        [(xL + w, y0), (xL + w / 2, y0 + h), (xL + 1.5 * w, y0 + h)],        # T2 下
        [(xL + w, y0), (xL + 1.5 * w, y0 + h), (xL + 2 * w, y0)],            # T3 上
        [(xL + 2 * w, y0), (xL + 1.5 * w, y0 + h), (xL + 2.5 * w, y0 + h)],  # T4 下
    ]
    shared = [                       # 相邻两面共用的边
        [(xL + w, y0), (xL + w / 2, y0 + h)],
        [(xL + w, y0), (xL + 1.5 * w, y0 + h)],
        [(xL + 2 * w, y0), (xL + 1.5 * w, y0 + h)],
    ]
    labels = ["T1（种子）", "T2", "T3", "T4"]
    for k, tri in enumerate(tris):
        merged = k < 3
        ax.add_patch(Polygon(tri, closed=True,
                             facecolor=GRAY if merged else "white",
                             edgecolor="black", linewidth=LW,
                             hatch="//" if merged else None, zorder=2))
        cx = float(np.mean([p[0] for p in tri]))
        cy = float(np.mean([p[1] for p in tri]))
        ax.text(cx, cy - 0.028, labels[k], ha="center", va="center",
                fontsize=11.5, zorder=5)
        # 法向箭头：T1-T3 一致向上；T4 明显偏转（zorder 高于填充，确保可见）
        tip = (cx, cy + 0.205) if merged else (cx + 0.165, cy + 0.125)
        arrow(ax, (cx, cy + 0.012), tip, lw=2.0)
        ax.scatter([cx], [cy + 0.012], s=16, c="black", zorder=7)

    for seg in shared:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                color="black", lw=5.0, solid_capstyle="round", zorder=3)

    ax.text(0.285, 0.835, "T1～T3 法向一致", fontsize=12.5, ha="center")
    ax.text(0.855, 0.795, "T4 法向偏转过大\n（不并入）", fontsize=12.5, ha="center")
    mid2 = ((shared[2][0][0] + shared[2][1][0]) / 2,
            (shared[2][0][1] + shared[2][1][1]) / 2)
    ax.annotate("共享边（加粗）", xy=mid2, xytext=(0.545, 0.335),
                fontsize=12, ha="center",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))

    # 结论行：明确"共享边邻接（前提）+ 两项法向约束"
    ax.text(0.5, 0.295,
            "满足共享边邻接、且同时满足两项法向约束的三角面并入同一表面分块",
            ha="center", va="center", fontsize=13)

    bw_, bh_, by = 0.235, 0.082, 0.045
    bxs = [0.075, 0.385, 0.695]
    texts = ["前提：共享边邻接", "约束①：当前法向约束", "约束②：种子法向约束"]
    for bx, tx in zip(bxs, texts):
        ax.add_patch(Rectangle((bx, by), bw_, bh_, facecolor=GRAY,
                               edgecolor="black", linewidth=LW))
        ax.text(bx + bw_ / 2, by + bh_ / 2, tx, ha="center", va="center", fontsize=11.5)
    arrow(ax, (bxs[0] + bw_ + 0.008, by + bh_ / 2), (bxs[1] - 0.008, by + bh_ / 2), lw=1.6)
    arrow(ax, (bxs[1] + bw_ + 0.008, by + bh_ / 2), (bxs[2] - 0.008, by + bh_ / 2), lw=1.6)
    ybr = by + bh_ + 0.022
    ax.plot([bxs[1], bxs[2] + bw_], [ybr] * 2, color="black", lw=1.3)
    ax.plot([bxs[1]] * 2, [ybr - 0.014, ybr], color="black", lw=1.3)
    ax.plot([bxs[2] + bw_] * 2, [ybr - 0.014, ybr], color="black", lw=1.3)
    ax.text((bxs[1] + bxs[2] + bw_) / 2, ybr + 0.032, "两项法向约束",
            ha="center", va="center", fontsize=11.5)
    save(fig, "图4_共享边和双法向约束形成表面分块示意图")


# ---------------------------------------------------------------- 图5
def fig5():
    """点云坐标对齐、最近表面点和距离阈值。

    画出IFC三角表面实线、两侧 ±tau_d 平行虚线构成的阈值带；
    带内点为实心（匹配、计入表面分块），带外为空心（未匹配）；
    左侧以虚影+箭头表示刚性对齐前后。
    """
    fig, ax = new_ax("图5  点云坐标对齐、最近表面点和距离阈值")

    # IFC 表面（略倾斜的实线）
    x0, x1 = 0.30, 0.955
    def surf(x):
        return 0.40 + 0.20 * (x - x0) / (x1 - x0)
    tau = 0.085
    xs = np.linspace(x0, x1, 200)
    ax.plot(xs, surf(xs), color="black", lw=3.0, zorder=3)
    ax.plot(xs, surf(xs) + tau, color="black", lw=1.6, ls="--", zorder=3)
    ax.plot(xs, surf(xs) - tau, color="black", lw=1.6, ls="--", zorder=3)
    ax.fill_between(xs, surf(xs) - tau, surf(xs) + tau, color=GRAY, alpha=0.55, zorder=1)

    ax.annotate("IFC三角表面", xy=(0.60, surf(0.60)), xytext=(0.545, 0.245),
                fontsize=12.5, ha="center",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
    ax.annotate("距离阈值边界\n（表面两侧各 $\\tau_d$）", xy=(0.90, surf(0.90) + tau),
                xytext=(0.855, 0.885), fontsize=12.5, ha="center",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))

    # 点：带内实心，带外空心
    rng = np.random.default_rng(11)
    inx = rng.uniform(x0 + 0.02, x1 - 0.02, 46)
    iny = surf(inx) + rng.uniform(-tau * 0.85, tau * 0.85, inx.size)
    ax.scatter(inx, iny, s=34, c="black", zorder=4)
    outx = rng.uniform(x0 + 0.02, x1 - 0.02, 15)
    sign = rng.choice([-1, 1], outx.size)
    outy = surf(outx) + sign * rng.uniform(tau * 1.45, tau * 2.25, outx.size)
    ax.scatter(outx, outy, s=34, facecolors="white", edgecolors="black",
               linewidths=1.6, zorder=4)

    # 最近表面点示例：从一个带内点向表面作垂线
    px, py = float(inx[3]), float(iny[3])
    m = 0.20 / (x1 - x0)
    t = (px + m * (py - surf(x0) - m * (px - x0) + m * px)) / 1.0
    fx = (px + m * py - m * (0.40 - m * x0)) / (1 + m * m)
    fy = surf(fx)
    ax.plot([px, fx], [py, fy], color="black", lw=1.4, ls=":", zorder=4)
    ax.scatter([fx], [fy], s=52, marker="x", c="black", linewidths=2.0, zorder=5)
    ax.annotate("最近表面点", xy=(fx, fy), xytext=(fx + 0.055, 0.255),
                fontsize=12, ha="center",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))

    # 图例（置于底部空白区，避开点云与标注）
    ax.scatter([0.075], [0.115], s=34, c="black")
    ax.text(0.098, 0.115, "带内的点：与该IFC表面匹配，计入所属表面分块",
            fontsize=11.5, va="center")
    ax.scatter([0.075], [0.048], s=34, facecolors="white", edgecolors="black", linewidths=1.6)
    ax.text(0.098, 0.048, "带外的点：未匹配，计入未匹配点统计",
            fontsize=11.5, va="center")

    # 刚性对齐：对齐前虚影点云 → 对齐后
    gx = rng.uniform(0.045, 0.165, 26)
    gy = rng.uniform(0.42, 0.64, 26)
    ax.scatter(gx, gy, s=26, facecolors="none", edgecolors=DARK, linewidths=1.2, zorder=2)
    ax.text(0.105, 0.375, "对齐前点云", fontsize=11.5, ha="center", color="0.35")
    arrow(ax, (0.190, 0.53), (0.278, 0.53), lw=2.4, ms=20)
    ax.text(0.234, 0.585, "刚性对齐", fontsize=12.5, ha="center")
    save(fig, "图5_点云坐标对齐最近表面点和距离阈值示意图")


# ---------------------------------------------------------------- 图8
def fig8():
    """基于中心、法向、距离、方位角和俯仰角生成候选位姿。

    三条弧线 = 三个候选距离层；每层弧线"上"各放三个候选位姿（空心圆），
    对应不同方位角/俯仰角偏移；候选朝向指回目标表面分块中心。
    """
    FW, FH = 9.6, 5.9
    fig, ax = new_ax("图8  基于中心、法向、距离、方位角和俯仰角生成候选位姿",
                     figsize=(FW, FH))
    ar = FW / FH  # 数据坐标纵横比校正，保证空心圆是正圆

    def circle(xy, d, **kw):
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse(xy, width=d, height=d * ar, **kw))

    cx, cy = 0.50, 0.285  # 目标表面分块中心
    ax.add_patch(Rectangle((cx - 0.075, cy - 0.048), 0.15, 0.075, facecolor=GRAY,
                           edgecolor="black", linewidth=LW, hatch="//", zorder=3))
    ax.text(cx + 0.115, cy - 0.012, "目标表面分块", ha="left", va="center", fontsize=12.5)
    ax.scatter([cx], [cy - 0.010], s=28, c="black", zorder=4)

    radii = [0.200, 0.300, 0.400]
    styles = ["-", "--", ":"]
    angs = np.deg2rad(np.linspace(26, 154, 240))
    for r, st in zip(radii, styles):
        ax.plot(cx + r * np.cos(angs), cy + r * np.sin(angs),
                color="0.35", lw=1.7, ls=st, zorder=1)

    # 每层弧线上各放 3 个候选位姿（严格落在弧线上）
    pick = np.deg2rad([52, 90, 128])
    for r in radii:
        for a in pick:
            px, py = cx + r * np.cos(a), cy + r * np.sin(a)
            if abs(a - np.pi / 2) > 1e-6:      # 90°方向留给法向，避免与法向箭头重叠
                ax.plot([cx, px], [cy, py], color="0.6", lw=1.0, zorder=1)
            arrow(ax, (px, py), (px - 0.055 * np.cos(a), py - 0.055 * np.sin(a)),
                  lw=1.5, ms=13)
            circle((px, py), 0.040, facecolor="white", edgecolor="black",
                   linewidth=1.9, zorder=5)

    # 法向（竖直向上，短于内层弧线）
    arrow(ax, (cx, cy + 0.032), (cx, cy + 0.158), lw=2.6, ms=19)
    ax.text(cx + 0.020, cy + 0.115, "法向", fontsize=12.5, ha="left", va="center")

    # "三层距离"引线，分别指向三条弧线的左端
    aL = np.deg2rad(150)
    for r in radii:
        ax.annotate("", xy=(cx + r * np.cos(aL), cy + r * np.sin(aL)),
                    xytext=(0.108, 0.735),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))
    ax.text(0.108, 0.775, "三层距离\n（三条弧线）", fontsize=12.5, ha="center", va="bottom")

    # 方位角/俯仰角偏移：指向最外层弧线上的一个候选
    ax.annotate("同一距离层上取不同的\n方位角偏移和俯仰角偏移",
                xy=(cx + radii[2] * np.cos(pick[0]) + 0.026,
                    cy + radii[2] * np.sin(pick[0]) + 0.012),
                xytext=(0.845, 0.815), fontsize=12, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))

    # 图例（底部，避开目标表面分块）
    circle((0.075, 0.115), 0.040, facecolor="white", edgecolor="black", linewidth=1.9)
    ax.text(0.105, 0.115, "空心圆：候选补充扫描位姿，位于对应距离层的弧线上",
            fontsize=11.5, va="center")
    ax.text(0.105, 0.048, "箭头：候选位姿的朝向，指向目标表面分块中心",
            fontsize=11.5, va="center")
    save(fig, "图8_候选位姿生成示意图")


if __name__ == "__main__":
    fig4()
    fig5()
    fig8()
