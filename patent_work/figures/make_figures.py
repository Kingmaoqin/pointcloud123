"""生成专利附图 1–6。

图 5 的等代价边界由真实的可通行距离场计算得到（八邻域 Dijkstra，与说明书 8.4
所述一致），不是手绘的欧氏圆；图 3 的距离层以围绕目标表面分块的壳带表示，
不是以分块形心为圆心的同心圆。

输出：patent_work/figures/图1..图6.（svg|png）
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).resolve().parent

# 中文字体：专利附图中的标注须为中文。若系统无中文字体则退回英文标签。
_CN = None
for cand in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/arphic/uming.ttc"):
    if Path(cand).exists():
        _CN = FontProperties(fname=cand)
        break
USE_CN = _CN is not None


def L(cn: str, en: str) -> str:
    return cn if USE_CN else en


def _save(fig, name: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {name}.svg / .png", flush=True)


def _txt(ax, x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("fontsize", 8)
    if USE_CN:
        kw["fontproperties"] = _CN
    return ax.text(x, y, s, **kw)


# ---------------------------------------------------------------- 图 1 总体流程
def fig1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis("off")

    steps = [
        ("S1", L("缺口证据与分数构建", "S1 gap evidence"), 11.4),
        ("S2", L("移动平台可通行空间构建", "S2 traversable space"), 10.2),
        ("S3", L("目标分块筛选与候选站位生成\n（含可通行路径代价）",
                 "S3 candidates + path cost"), 8.9),
        ("S4", L("基于规划用遮挡模型的候选站位评估",
                 "S4 evaluate vs planning occlusion model"), 7.5),
        ("S5", L("预算约束下选定待执行站位", "S5 select stations"), 6.3),
        ("S6", L("按预设执行规则确定本轮执行站位\n并获取新增点云",
                 "S6 execute one, acquire"), 5.0),
        ("S7a", L("在线发现模型外遮挡物", "S7a discover off-model occluders"), 3.5),
        ("S7b", L("新增观测的配准可用性判定（可选）",
                  "S7b registration usability (optional)"), 2.3),
        ("S8", L("更新规划用遮挡模型、缺口证据、剩余预算",
                 "S8 update model / evidence / budget"), 1.0),
    ]
    boxes = {}
    for tag, label, y in steps:
        w = 6.2
        b = mp.FancyBboxPatch((1.9, y - 0.42), w, 0.84,
                              boxstyle="round,pad=0.06", linewidth=1.2,
                              edgecolor="black", facecolor="white")
        ax.add_patch(b)
        _txt(ax, 2.25, y, tag, fontsize=9, ha="left", fontweight="bold")
        _txt(ax, 5.3, y, label, fontsize=8)
        boxes[tag] = y

    # 主数据流（实线箭头）
    chain = ["S1", "S2", "S3", "S4", "S5", "S6"]
    for a, b in zip(chain, chain[1:]):
        ax.annotate("", xy=(5.0, boxes[b] + 0.42), xytext=(5.0, boxes[a] - 0.42),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))
    for tgt in ("S7a", "S7b"):
        ax.annotate("", xy=(5.0, boxes[tgt] + 0.42), xytext=(5.0, boxes["S6"] - 0.42),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black",
                                    connectionstyle="arc3,rad=0"))
    ax.annotate("", xy=(5.0, boxes["S8"] + 0.42), xytext=(5.0, boxes["S7b"] - 0.42),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))

    # 回灌（虚线箭头，走右侧）
    ax.annotate("", xy=(8.35, boxes["S4"]), xytext=(8.35, boxes["S8"]),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black", ls="--"))
    ax.plot([8.1, 8.35], [boxes["S8"]] * 2, color="black", lw=1.2, ls="--")
    ax.plot([8.1, 8.35], [boxes["S4"]] * 2, color="black", lw=1.2, ls="--")
    _txt(ax, 9.1, (boxes["S4"] + boxes["S8"]) / 2,
         L("规划用遮挡模型回灌", "occlusion model"), fontsize=7.5, rotation=90)

    ax.annotate("", xy=(1.15, boxes["S3"]), xytext=(1.15, boxes["S8"]),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black", ls="--"))
    ax.plot([1.15, 1.9], [boxes["S8"]] * 2, color="black", lw=1.2, ls="--")
    ax.plot([1.15, 1.9], [boxes["S3"]] * 2, color="black", lw=1.2, ls="--")
    _txt(ax, 0.55, (boxes["S3"] + boxes["S8"]) / 2,
         L("缺口证据回灌", "gap evidence"), fontsize=7.5, rotation=90)

    _txt(ax, 5.0, 12.15, L("图 1  总体流程", "Fig.1 Overall flow"),
         fontsize=10, fontweight="bold")
    _txt(ax, 5.0, 0.25,
         L("实线箭头：本轮数据流    虚线箭头：跨轮回灌",
           "solid: within-round flow   dashed: cross-round feedback"), fontsize=7.5)
    _save(fig, "图1_总体流程")


# ---------------------------------------------------------------- 图 2 可通行空间
def fig2() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")

    # 构件 A：普通障碍
    ax.add_patch(mp.Rectangle((2, 5.5), 3, 2.4, facecolor="0.35",
                              edgecolor="black", lw=1.2))
    _txt(ax, 3.5, 6.7, L("构件A\n普通障碍", "A obstacle"), color="white", fontsize=7.5)
    ax.add_patch(mp.Rectangle((1.4, 4.9), 4.2, 3.6, facecolor="none",
                              edgecolor="black", lw=1.0, ls=":"))
    # 构件 B：带电，安全膨胀
    ax.add_patch(mp.Rectangle((7.6, 5.8), 2.2, 2.0, facecolor="0.35",
                              edgecolor="black", lw=1.2))
    ax.add_patch(mp.Rectangle((6.5, 4.7), 4.4, 4.2, facecolor="none",
                              edgecolor="black", lw=1.1, hatch="///"))
    _txt(ax, 8.7, 6.8, L("构件B\n带电", "B live"), color="white", fontsize=7.5)
    _txt(ax, 8.7, 9.35, L("带电体安全距离标记", "live safety zone"), fontsize=7.5)
    # 构件 C：顶面低于跨越高度 → 不标记
    ax.add_patch(mp.Rectangle((2.2, 1.4), 3.0, 1.2, facecolor="white",
                              edgecolor="black", lw=1.2, ls="--"))
    _txt(ax, 3.7, 2.0, L("构件C 顶面≤跨越高度\n不标记为障碍",
                         "C top ≤ step height\nnot an obstacle"), fontsize=7.5)
    # 构件 D：净空大于平台高度 → 不标记
    ax.add_patch(mp.Rectangle((11.6, 6.6), 3.2, 0.7, facecolor="0.35",
                              edgecolor="black", lw=1.2))
    ax.annotate("", xy=(13.2, 6.55), xytext=(13.2, 4.4),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color="black"))
    _txt(ax, 14.6, 5.5, L("净空>平台高度\n不标记为障碍",
                          "clearance > robot h\nnot an obstacle"), fontsize=7.5)
    ax.add_patch(mp.Rectangle((11.4, 3.9), 3.6, 0.5, facecolor="white",
                              edgecolor="black", lw=0.8))
    _txt(ax, 13.2, 4.15, L("可从下穿行", "passable underneath"), fontsize=7)

    _txt(ax, 8.0, 0.45, L(
        "实心填充=障碍栅格   斜线阴影=带电体安全距离标记   "
        "点线外框=按平台半径的膨胀   白色=自由空间",
        "solid=obstacle  hatch=live safety  dotted outline=robot-radius dilation  white=free"),
        fontsize=7.2)
    _txt(ax, 8.0, 9.7, L("图 2  可通行空间构建", "Fig.2 Traversable space"),
         fontsize=10, fontweight="bold")
    _save(fig, "图2_可通行空间")


# ---------------------------------------------------------------- 图 3 距离层/壳
def fig3() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.set_xlim(-1, 13); ax.set_ylim(-1, 9); ax.set_aspect("equal"); ax.axis("off")

    # 目标表面分块：一段有限长的面，法向朝右上
    px, py = np.array([3.0, 1.4]), np.array([3.0, 5.2])
    seg = np.vstack([px, py])
    ax.plot(seg[:, 0], seg[:, 1], color="black", lw=3.5, solid_capstyle="butt")
    _txt(ax, 2.15, 3.3, L("目标表面分块", "target patch"), fontsize=8, rotation=90)
    n = np.array([1.0, 0.0])
    mid = seg.mean(axis=0)
    ax.annotate("", xy=(mid[0] + 1.5, mid[1]), xytext=(mid[0], mid[1]),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"))
    _txt(ax, mid[0] + 1.7, mid[1] + 0.35, L("法向 n", "normal n"), fontsize=8)

    # 距离层/壳：沿分块**整段**外扩的壳带（不是以形心为圆心的同心圆）
    # 壳带 = 到线段距离落在 [d-δ, d+δ] 的区域，只取法向侧
    gx, gy = np.meshgrid(np.linspace(-1, 13, 700), np.linspace(-1, 9, 500))
    a, b = seg[0], seg[1]
    ab = b - a
    t = np.clip(((gx - a[0]) * ab[0] + (gy - a[1]) * ab[1]) / (ab @ ab), 0, 1)
    cx, cy = a[0] + t * ab[0], a[1] + t * ab[1]
    dist = np.hypot(gx - cx, gy - cy)
    side = (gx - cx) * n[0] + (gy - cy) * n[1] > 0

    for d, lab in ((2.5, "d1"), (5.0, "d2"), (8.0, "d3")):
        band = (np.abs(dist - d) < 0.30) & side
        ax.contourf(gx, gy, band.astype(float), levels=[0.5, 1.5],
                    colors=["0.78"], alpha=0.95)
        ax.contour(gx, gy, np.where(side, dist, np.nan), levels=[d],
                   colors="black", linewidths=0.9, linestyles="dashed")
        _txt(ax, 3.0 + d, 7.9, L(f"距离层 {lab}", f"shell {lab}"), fontsize=8)

    # 候选：落在壳带上；实心=保留，空心=剔除
    keep = [(5.5, 2.2), (5.5, 4.6), (8.0, 3.1), (8.0, 5.6), (11.0, 3.3)]
    drop = [(5.5, 6.6), (8.0, 1.0)]
    for x, y in keep:
        ax.plot(x, y, "o", ms=8, mfc="black", mec="black")
    for x, y in drop:
        ax.plot(x, y, "o", ms=8, mfc="white", mec="black", mew=1.4)
    _txt(ax, 6.35, 2.2, "ℓ=4.1", fontsize=7)
    _txt(ax, 8.85, 3.1, "ℓ=6.8", fontsize=7)
    _txt(ax, 11.85, 3.3, "ℓ=9.5", fontsize=7)

    _txt(ax, 6.0, 8.6, L("图 3  候选站位与距离层", "Fig.3 Candidates and distance shells"),
         fontsize=10, fontweight="bold")
    _txt(ax, 6.0, -0.7, L(
        "灰带=距离层（沿分块整段外扩的壳，非以形心为圆心的同心圆）   "
        "虚线=层中心距离   实心圆=保留的候选   空心圆=被剔除的候选   ℓ=可通行路径代价",
        "grey band=distance shell along the whole patch, not concentric circles; "
        "filled=kept candidate; hollow=rejected; ℓ=traversable path cost"), fontsize=7.0)
    _save(fig, "图3_候选站位与距离层")


# ---------------------------------------------------------------- 图 4 遮挡发现
def fig4() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4))
    for k, ax in enumerate(axes):
        ax.set_xlim(0, 14); ax.set_ylim(0, 5.4); ax.set_aspect("equal"); ax.axis("off")
        # 候选站位
        ax.plot(1.4, 2.7, "o", ms=9, mfc="black", mec="black")
        _txt(ax, 1.4, 1.9, L("候选站位", "candidate"), fontsize=7.5)
        # 目标分块
        ax.plot([12.4, 12.4], [1.3, 4.1], color="black", lw=3.5)
        _txt(ax, 13.2, 2.7, L("目标\n表面分块", "target\npatch"), fontsize=7.5)
        # 模型外遮挡物（现场存在）
        ax.add_patch(mp.Rectangle((6.4, 1.9), 1.6, 1.7, facecolor="0.55",
                                  edgecolor="black", lw=1.2, hatch="xx"))
        _txt(ax, 7.2, 4.05, L("模型外遮挡物（现场存在）",
                              "off-model occluder (on site)"), fontsize=7.2)
        if k == 0:
            ax.annotate("", xy=(12.3, 2.7), xytext=(1.7, 2.7),
                        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))
            _txt(ax, 4.0, 3.1, L("射线在规划用遮挡模型上无阻断\n→ 预测为可见",
                                 "ray unblocked in planning model\n→ predicted visible"),
                 fontsize=7.2)
            _txt(ax, 7.0, 5.05, L("（上）并入发现体素之前", "(a) before merging"),
                 fontsize=9, fontweight="bold")
        else:
            # 发现体素：不同填充，且明确标注不携带分块归属
            for i in range(4):
                for j in range(4):
                    ax.add_patch(mp.Rectangle((6.4 + i * 0.4, 1.9 + j * 0.425), 0.4,
                                              0.425, facecolor="white",
                                              edgecolor="black", lw=0.8, ls=":"))
            ax.annotate("", xy=(6.3, 2.7), xytext=(1.7, 2.7),
                        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))
            ax.plot([6.35, 6.35], [2.35, 3.05], color="black", lw=2.2)
            _txt(ax, 3.6, 3.1, L("射线被发现体素阻断\n→ 预测为不可见",
                                 "ray blocked by discovered voxels\n→ predicted occluded"),
                 fontsize=7.2)
            _txt(ax, 7.2, 1.35, L("点线方格=发现体素\n（不携带分块归属，只作遮挡体）",
                                  "dotted cells = discovered voxels\n(no patch ownership)"),
                 fontsize=7.0)
            _txt(ax, 7.0, 5.05, L("（下）并入发现体素之后", "(b) after merging"),
                 fontsize=9, fontweight="bold")
    fig.suptitle(L("图 4  在线发现模型外遮挡物", "Fig.4 Online discovery of off-model occluders"),
                 fontsize=10, fontweight="bold",
                 fontproperties=_CN if USE_CN else None)
    _save(fig, "图4_在线遮挡发现")


# ---------------------------------------------------------------- 图 5 等代价边界
def fig5() -> None:
    """预算可达范围由真实的八邻域可通行距离场给出，受障碍与通道影响。"""
    from patent_gap.mapping.traversability import TravGrid

    grid = TravGrid((0.0, 0.0, 16.0, 10.0), res=0.1, r_robot=0.0)
    # 两道带门洞的墙，使等代价边界必须穿过门洞并在其后形成凸出
    walls = [((4.2, 0.0), (4.8, 4.2)), ((4.2, 5.8), (4.8, 10.0)),
             ((9.4, 3.4), (10.0, 10.0)), ((12.8, 0.0), (13.4, 6.0))]
    for lo, hi in walls:
        grid.add_obstacle_box(lo, hi, clearance_z=0.0, h_robot=1.8)

    src = (1.6, 5.0)
    D = grid.distance_field(src)
    free = grid._compute_free()
    Dm = np.where(free, D, np.nan)

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    ax.set_xlim(-0.3, 16.3); ax.set_ylim(-1.5, 12.0)
    ax.set_aspect("equal"); ax.axis("off")
    xs = grid.xmin + (np.arange(grid.nx) + 0.5) * grid.res
    ys = grid.ymin + (np.arange(grid.ny) + 0.5) * grid.res
    ax.add_patch(mp.Rectangle((0, 0), 16, 10, facecolor="none",
                              edgecolor="black", lw=0.8))

    B = 7.2   # 使边界穿过第一道门洞后在其右侧形成可见凸出, 且不越出场地
    ax.contourf(xs, ys, Dm.T, levels=[0, B], colors=["0.87"])
    ax.contour(xs, ys, Dm.T, levels=[B], colors="black", linewidths=2.2)
    # 对照：同代价的欧氏圆，用细虚线画出，明示二者不同
    th = np.linspace(0, 2 * np.pi, 600)
    ex, ey = src[0] + B * np.cos(th), src[1] + B * np.sin(th)
    inside = (ex >= 0) & (ex <= 16) & (ey >= 0) & (ey <= 10)
    ax.plot(np.where(inside, ex, np.nan), np.where(inside, ey, np.nan),
            color="black", lw=0.9, ls="--")

    for lo, hi in walls:
        ax.add_patch(mp.Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1],
                                  facecolor="0.35", edgecolor="black", lw=1.0))

    ax.plot(*src, "s", ms=9, mfc="black", mec="black")
    _txt(ax, src[0], src[1] - 0.85, L("当前位姿", "current pose"), fontsize=7.5)

    # 待执行站位：均在等代价边界之外
    for x, y in [(11.2, 8.6), (14.8, 8.0), (14.6, 2.4)]:
        ax.plot(x, y, "^", ms=10, mfc="white", mec="black", mew=1.5)
    _txt(ax, 11.0, 11.0, L("待执行站位：均在边界之外（剩余预算内不可达）",
                           "stations to execute: outside the boundary"), fontsize=7.4)
    # 回退所选候选：边界之内
    ax.plot(6.4, 6.9, "o", ms=9, mfc="black", mec="black")
    _txt(ax, 7.6, 4.5, L("回退所选候选\n（边界之内，价值最高）",
                         "fallback pick\n(inside, highest value)"), fontsize=7.4)
    ax.annotate("", xy=(6.45, 6.65), xytext=(7.4, 5.1),
                arrowprops=dict(arrowstyle="-", lw=0.8, color="black"))
    # 指出门洞处的绕行
    ax.annotate("", xy=(4.5, 5.0), xytext=(2.6, 1.6),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"))
    _txt(ax, 2.6, 1.15, L("门洞：边界受墙体约束，\n只能经此向右延伸",
                          "doorway: the boundary is walled in\nand extends only through it"),
         fontsize=7.0)

    _txt(ax, 8.0, 11.7, L("图 5  剩余预算的等代价边界",
                          "Fig.5 Iso-cost boundary of remaining budget"),
         fontsize=10, fontweight="bold")
    _txt(ax, 8.0, -0.85, L(
        "粗实线=等代价边界（由可通行路径代价给出，绕开障碍并沿通道延伸）    "
        "细虚线=同代价的欧氏圆（对照，非本方案所用）", ""), fontsize=7.0)
    _txt(ax, 8.0, -1.32, L(
        "灰区=剩余预算内可达    实心填充=障碍    ▲=待执行站位    ●=回退所选候选",
        "grey=reachable within budget; solid=obstacle"), fontsize=7.0)
    _save(fig, "图5_等代价边界")


# ---------------------------------------------------------------- 图 6 构件分组
def fig6() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    rng = np.random.default_rng(3)
    big = [(1.2, 5.4, 3.4, 2.6), (1.2, 1.4, 3.4, 3.2)]
    small = [(6.2 + (i % 3) * 1.5, 1.6 + (i // 3) * 1.6, 0.7, 0.7) for i in range(9)]

    for k, ax in enumerate(axes):
        ax.set_xlim(0, 11); ax.set_ylim(0, 9); ax.set_aspect("equal"); ax.axis("off")
        for x, y, w, h in big:
            ax.add_patch(mp.Rectangle((x, y), w, h, facecolor="0.8",
                                      edgecolor="black", lw=1.1))
        for x, y, w, h in small:
            ax.add_patch(mp.Rectangle((x, y), w, h, facecolor="0.8",
                                      edgecolor="black", lw=1.1))
        _txt(ax, 2.9, 8.35, L("大面积构件", "large components"), fontsize=7.8)
        _txt(ax, 7.7, 8.35, L("小尺寸构件", "small components"), fontsize=7.8)

        if k == 0:
            for x, y, w, h in big:
                for _ in range(6):
                    ax.plot(x + rng.uniform(0.3, w - 0.3), y + rng.uniform(0.3, h - 0.3),
                            "*", ms=9, color="black")
            _txt(ax, 5.5, 0.55, L("（左）按固定条数截断：名额被大面积构件占满，\n"
                                  "小尺寸构件无表面分块入选",
                                  "(a) fixed cap: large components take all slots"),
                 fontsize=7.4)
        else:
            for x, y, w, h in big:
                for _ in range(2):
                    ax.plot(x + rng.uniform(0.3, w - 0.3), y + rng.uniform(0.3, h - 0.3),
                            "*", ms=9, color="black")
            for x, y, w, h in small:
                ax.plot(x + w / 2, y + h / 2, "*", ms=9, color="black")
            _txt(ax, 5.5, 0.55, L("（右）按构件分组、容量随构件数缩放：\n"
                                  "每一构件均有表面分块入选",
                                  "(b) per-component grouping: every component represented"),
                 fontsize=7.4)
    fig.suptitle(L("图 6  目标表面分块的选取", "Fig.6 Selection of target patches"),
                 fontsize=10, fontweight="bold",
                 fontproperties=_CN if USE_CN else None)
    _txt(axes[0], 5.5, -0.35, L("★=入选目标分块集合的表面分块",
                                "star = patch entering the target set"), fontsize=7.2)
    _save(fig, "图6_目标分块选取")


if __name__ == "__main__":
    print(f"中文字体: {'可用' if USE_CN else '不可用（退回英文标签）'}", flush=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print("完成", flush=True)
