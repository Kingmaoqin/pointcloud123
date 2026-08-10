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
    """总体流程。S7b 的判定在 S7a 之前 —— 配准不可用时全局坐标不可信,
    因而不能据以判定'离设计模型超过 ε', S7a 整体不执行。"""
    fig, ax = plt.subplots(figsize=(8.4, 10.4))
    ax.set_xlim(-0.2, 13.0); ax.set_ylim(0, 15.2); ax.axis("off")

    def box(x, y, w, h, tag, label, fs=8):
        ax.add_patch(mp.FancyBboxPatch((x, y - h / 2), w, h,
                                       boxstyle="round,pad=0.05", lw=1.2,
                                       edgecolor="black", facecolor="white"))
        if tag:
            _txt(ax, x + 0.28, y, tag, fontsize=8.5, ha="left", fontweight="bold")
        _txt(ax, x + w / 2 + (0.3 if tag else 0), y, label, fontsize=fs)

    def dia(cx, cy, w, h, label):
        ax.add_patch(mp.Polygon([[cx, cy + h / 2], [cx + w / 2, cy],
                                 [cx, cy - h / 2], [cx - w / 2, cy]],
                                closed=True, lw=1.2, edgecolor="black",
                                facecolor="white"))
        _txt(ax, cx, cy, label, fontsize=7.6)

    def arr(x0, y0, x1, y1, dashed=False, lw=1.3):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color="black",
                                    ls="--" if dashed else "-"))

    Y = {"S1": 14.5, "S2": 13.5, "S3": 12.5, "S4": 11.5, "S5": 10.5, "S6": 9.5}
    for tag in ("S1", "S2", "S3", "S4", "S5", "S6"):
        lbl = {"S1": L("缺口证据与分数构建（含缺口表面分块判定）", "S1 gap evidence"),
               "S2": L("移动平台可通行空间构建", "S2 traversable space"),
               "S3": L("目标分块筛选与候选站位生成（含可通行路径代价）", "S3 candidates"),
               "S4": L("基于规划用遮挡模型的候选站位评估", "S4 evaluate"),
               "S5": L("预算约束下选定待执行站位", "S5 select"),
               "S6": L("按预设执行规则确定本轮执行站位并获取新增点云", "S6 execute")}[tag]
        box(2.6, Y[tag], 7.4, 0.72, tag, lbl)
    for a, b in zip(["S1", "S2", "S3", "S4", "S5"], ["S2", "S3", "S4", "S5", "S6"]):
        arr(6.3, Y[a] - 0.36, 6.3, Y[b] + 0.36)

    dia(6.3, 8.35, 4.6, 1.1, L("是否启用配准可用性判定 S7b？",
                               "registration check enabled?"))
    arr(6.3, Y["S6"] - 0.36, 6.3, 8.9)

    # 左支：不启用 → 缺省满足
    _txt(ax, 3.3, 8.62, L("否", "no"), fontsize=8)
    ax.plot([4.0, 2.3], [8.35, 8.35], color="black", lw=1.3)
    arr(2.3, 8.35, 2.3, 7.35, lw=1.3)
    box(0.55, 7.05, 3.5, 0.66, "", L("缺省：视为满足更新条件", "default: condition holds"), 7.4)
    ax.plot([2.3, 2.3], [6.72, 6.15], color="black", lw=1.3)

    # 右支：启用 → S7b
    _txt(ax, 9.1, 8.62, L("是", "yes"), fontsize=8)
    ax.plot([8.6, 10.2], [8.35, 8.35], color="black", lw=1.3)
    arr(10.2, 8.35, 10.2, 7.35, lw=1.3)
    box(8.3, 7.05, 3.4, 0.66, "S7b", L("配准可用性判定", "usability check"), 7.4)
    ax.plot([10.2, 10.2], [6.72, 6.15], color="black", lw=1.3)

    dia(6.3, 5.55, 5.0, 1.15, L("新增点云满足预设更新条件？",
                                "new cloud satisfies condition?"))
    ax.plot([2.3, 3.8], [6.15, 6.15], color="black", lw=1.3)
    arr(3.8, 6.15, 4.6, 5.85)
    ax.plot([10.2, 8.8], [6.15, 6.15], color="black", lw=1.3)
    arr(8.8, 6.15, 8.0, 5.85)

    # 是 → S7a → S8
    _txt(ax, 5.32, 4.72, L("是", "yes"), fontsize=8)
    arr(5.0, 4.98, 5.0, 4.36)
    box(2.2, 4.0, 5.6, 0.72, "S7a", L("在线发现模型外遮挡物 → 体素 U_k",
                                      "S7a discover off-model occluders"), 7.4)
    arr(5.0, 3.64, 5.0, 3.09)
    box(2.0, 2.66, 5.8, 0.86, "S8", L("并入规划用遮挡模型；更新缺口证据与任务分数；\n重新判定缺口表面分块",
                                      "S8 merge; update evidence and scores;\nre-determine gap patches"), 7.0)

    # 否 → 保持不变
    _txt(ax, 9.3, 5.82, L("否", "no"), fontsize=8)
    ax.plot([8.8, 11.55], [5.55, 5.55], color="black", lw=1.3)
    arr(11.55, 5.55, 11.55, 4.43, lw=1.3)
    ax.plot([11.55, 11.85], [4.43, 4.43], color="black", lw=1.3)
    arr(11.85, 4.43, 11.78, 4.0)
    box(8.45, 4.0, 3.3, 0.86, "", L("遮挡模型与缺口证据均保持不变；\n记录失败站位",
                                      "model & evidence unchanged;\nrecord failed station"), 7.0)

    arr(5.0, 2.30, 5.0, 1.72)
    ax.plot([10.1, 10.1], [3.57, 1.90], color="black", lw=1.3)
    ax.plot([10.1, 5.0], [1.90, 1.90], color="black", lw=1.3)
    box(2.2, 1.36, 7.4, 0.72, "", L("扣减本轮已发生的可通行路径代价",
                                    "deduct path cost incurred"), 7.8)

    # 回灌之一：缺口状态、失败站位、剩余预算 → S3（左侧）
    ax.plot([2.2, 0.62], [1.36, 1.36], color="black", lw=1.2, ls="--")
    arr(0.62, 1.36, 0.62, Y["S3"], dashed=True, lw=1.2)
    ax.plot([0.62, 2.6], [Y["S3"]] * 2, color="black", lw=1.2, ls="--")
    _txt(ax, 0.24, 7.2, L("缺口状态、失败站位、剩余预算 → S3",
                          "gap state / failed station / budget -> S3"),
         fontsize=7.0, rotation=90)
    # 回灌之二：更新后的规划用遮挡模型 → S4（右侧）。这是本方法的核心回路，
    # 必须与上一条分开画：发现遮挡 → 修改模型 → 影响下一轮可见性评估。
    ax.plot([7.8, 12.15], [2.66, 2.66], color="black", lw=1.2, ls="--")
    arr(12.15, 2.66, 12.15, Y["S4"], dashed=True, lw=1.2)
    ax.plot([12.15, 10.0], [Y["S4"]] * 2, color="black", lw=1.2, ls="--")
    _txt(ax, 12.52, 7.2, L("更新后的规划用遮挡模型 → S4",
                           "updated occlusion model -> S4"),
         fontsize=7.0, rotation=90)

    _txt(ax, 6.0, 15.05, L("图 1  总体流程", "Fig.1 Overall flow"),
         fontsize=10.5, fontweight="bold")
    _txt(ax, 6.0, 0.5, L(
        "矩形=处理步骤    菱形=判定    实线箭头=数据流    虚线箭头=跨轮回灌",
        "rect=step  diamond=decision  solid=flow  dashed=cross-round feedback"),
        fontsize=7.4)
    _txt(ax, 6.0, 0.12, L(
        "注：S7b 判定在 S7a 之前——配准不可用时该站点云的全局坐标不可信，"
        "无法据以判定其是否偏离设计模型，故 S7a 整体不执行。", ""), fontsize=7.0)
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
    # 在安全区域基础上再按平台半径膨胀（与算法一致：障碍与禁入区一并膨胀）
    ax.add_patch(mp.Rectangle((5.9, 4.1), 5.6, 5.4, facecolor="none",
                              edgecolor="black", lw=1.0, ls=":"))
    _txt(ax, 8.7, 6.8, L("构件B\n带电", "B live"), color="white", fontsize=7.5)
    _txt(ax, 8.7, 9.35, L("带电体安全距离标记", "live safety zone"), fontsize=7.5)
    # 构件 C：顶面低于跨越高度 → 不标记
    ax.add_patch(mp.Rectangle((2.2, 1.4), 3.0, 1.2, facecolor="white",
                              edgecolor="black", lw=1.0))
    _txt(ax, 3.7, 2.0, L("构件C 顶面≤跨越高度\n不标记为障碍（故无膨胀）",
                         "C top ≤ step height\nnot an obstacle (no dilation)"), fontsize=7.2)
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
        "实心填充=障碍栅格   斜线阴影=带电体安全距离禁入标记   "
        "点线外框=对已标记栅格（障碍与禁入）按平台半径的膨胀   细实线框=未标记为障碍的构件   白色=自由空间",
        "solid=obstacle  hatch=live safety  dotted outline=robot-radius dilation  white=free"),
        fontsize=7.2)
    _txt(ax, 8.0, 9.7, L("图 2  可通行空间构建", "Fig.2 Traversable space"),
         fontsize=10, fontweight="bold")
    _save(fig, "图2_可通行空间")


# ---------------------------------------------------------------- 图 3 距离层
def fig3() -> None:
    """候选位置 = 分块形心 + 距离档 × 方向，方向为法向经方位角/俯仰角偏置旋转。

    故距离层是以形心为球心、以距离档为半径、限制在法向锥内的球面区段，
    在平面投影中表现为围绕形心的圆弧带，而非沿分块整段外扩的壳。
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.set_xlim(-1.2, 14.6); ax.set_ylim(-2.8, 8.2)
    ax.set_aspect("equal"); ax.axis("off")

    # 目标表面分块与其形心
    a, b = np.array([2.4, 1.4]), np.array([2.4, 4.6])
    ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=3.5, solid_capstyle="butt")
    _txt(ax, 1.75, 3.8, L("目标表面分块", "target patch"), fontsize=8, rotation=90)
    c = (a + b) / 2.0
    ax.plot(*c, "o", ms=5, mfc="black", mec="black")
    _txt(ax, 2.15, c[1] - 0.55, L("形心", "centroid"), fontsize=7.4)
    ax.annotate("", xy=(c[0] + 1.5, c[1]), xytext=(c[0], c[1]),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"))
    _txt(ax, c[0] + 1.75, c[1] + 0.38, L("法向 n", "normal n"), fontsize=8)

    # 法向锥：方位角偏置 ±45°
    half = np.deg2rad(38.0)
    for sgn in (-1, 1):
        ang = sgn * half
        ax.plot([c[0], c[0] + 7.6 * np.cos(ang)], [c[1], c[1] + 7.6 * np.sin(ang)],
                color="black", lw=0.8, ls=(0, (2, 3)))
    ang_lab = np.deg2rad(38.0)
    ax.annotate("", xy=(c[0] + 6.9 * np.cos(ang_lab), c[1] + 6.9 * np.sin(ang_lab)),
                xytext=(9.9, c[1] + 4.55),
                arrowprops=dict(arrowstyle="-", lw=0.8, color="black"))
    _txt(ax, 11.2, c[1] + 4.7, L("法向锥的平面投影\n（方位角与俯仰角偏置范围）",
                                "normal cone (azimuth + elevation offsets),\nplan projection"), fontsize=7.4)

    th = np.linspace(-half, half, 240)
    for d, lab in ((2.4, "d1"), (4.4, "d2"), (6.6, "d3")):
        for w in (-0.24, 0.24):
            ax.plot(c[0] + (d + w) * np.cos(th), c[1] + (d + w) * np.sin(th),
                    color="0.55", lw=0.6)
        ax.fill_between(c[0] + (d + 0.24) * np.cos(th), c[1] + (d + 0.24) * np.sin(th),
                        c[1] + (d - 0.24) * np.sin(th), color="0.82", alpha=0.0)
        xs_o = c[0] + (d + 0.24) * np.cos(th); ys_o = c[1] + (d + 0.24) * np.sin(th)
        xs_i = c[0] + (d - 0.24) * np.cos(th); ys_i = c[1] + (d - 0.24) * np.sin(th)
        ax.fill(np.concatenate([xs_o, xs_i[::-1]]),
                np.concatenate([ys_o, ys_i[::-1]]), color="0.82", zorder=0)
        ax.plot(c[0] + d * np.cos(th), c[1] + d * np.sin(th),
                color="black", lw=0.9, ls="dashed")
        aa_lab = np.deg2rad(-30.0)
        _txt(ax, c[0] + (d + 0.62) * np.cos(aa_lab), c[1] + (d + 0.62) * np.sin(aa_lab),
             L(f"距离层 {lab}", f"shell {lab}"), fontsize=7.4,
             rotation=-30, rotation_mode="anchor")

    # 候选：位于距离层上（实心=保留），以及经投影后偏离层的一个（说明可偏离）
    keep = [(2.4, -14), (4.4, -2), (4.4, 24), (6.6, 12), (6.6, -12)]
    for d, adeg in keep:
        aa = np.deg2rad(adeg)
        ax.plot(c[0] + d * np.cos(aa), c[1] + d * np.sin(aa), "o",
                ms=7.5, mfc="black", mec="black")
    drop = [(2.4, 33), (4.4, -35)]
    for d, adeg in drop:
        aa = np.deg2rad(adeg)
        ax.plot(c[0] + d * np.cos(aa), c[1] + d * np.sin(aa), "o",
                ms=7.5, mfc="white", mec="black", mew=1.4)
    # 一个经自由空间投影后偏离层中心线的候选
    aa = np.deg2rad(14); px = c[0] + 4.4 * np.cos(aa) + 0.60
    py = c[1] + 4.4 * np.sin(aa) - 0.42
    ax.plot(px, py, "o", ms=7.5, mfc="black", mec="black")
    ax.annotate("", xy=(px, py), xytext=(px + 1.6, py + 1.1),
                arrowprops=dict(arrowstyle="-", lw=0.8, color="black"))
    _txt(ax, px + 2.7, py + 1.35, L("经自由空间投影后\n可偏离距离层",
                                     "may leave the shell\nafter projection"), fontsize=7.0)

    _txt(ax, 6.2, 7.95, L("图 3  候选补充扫描站位与距离层",
                          "Fig.3 Candidates and distance shells"),
         fontsize=10, fontweight="bold")
    _txt(ax, 6.2, -2.05, L(
        "灰带=距离层：以分块形心为球心、以距离档为半径、限制在法向锥内的球面区段（图为其平面投影）",
        "shell: sphere of radius d about the centroid, restricted to the normal cone"),
        fontsize=7.0)
    _txt(ax, 6.2, -2.48, L(
        "实心圆=保留的候选   空心圆=经自由空间投影、去重或可达性筛选后被剔除的候选   "
        "本图仅示出目标导向候选，S3 另由自由栅格节点生成的候选未示出",
        "filled=kept; hollow=rejected; grid-node candidates not shown"), fontsize=7.0)
    _save(fig, "图3_候选站位与距离层")


# ---------------------------------------------------------------- 图 4 遮挡发现
def fig4() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4))
    for k, ax in enumerate(axes):
        ax.set_xlim(0, 14); ax.set_ylim(0.1, 5.6); ax.set_aspect("equal"); ax.axis("off")
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
            _txt(ax, 4.0, 3.15, L("射线在规划用遮挡模型上无阻断\n→ 预测为可见",
                                 "ray unblocked in planning model\n→ predicted visible"),
                 fontsize=7.2)
            _txt(ax, 7.2, 0.95, L("交叉阴影块仅表示真实现场中客观存在的物体；\n"
                                  "并入发现体素之前，该几何不属于规划用遮挡模型",
                                  "hatched block exists on site only; before merging it is\n"
                                  "not part of the planning occlusion model"), fontsize=7.0)
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
            _txt(ax, 7.2, 0.95, L(
                "满足 d(x,T)>eps 且高度>z0 的回波点 → 体素化 → 并入规划用遮挡模型\n"
                "点线方格=发现体素（不携带分块归属，只作遮挡体，不是新的待扫目标）",
                "returns with d(x,T)>eps and height>z0 -> voxelized -> merged into the model\n"
                "dotted cells = discovered voxels (occluders only, never scan targets)"),
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
    _txt(ax, 7.6, 4.5, L("回退所选候选：全部候选中\n满足 ℓ(v)≤B 且价值最高者",
                         "fallback: highest-value candidate\namong all with l(v)<=B"), fontsize=7.4)
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
        "灰区：ℓ(v)≤B    粗实线：ℓ(v)=B    边界之外：ℓ(v)>B    "
        "实心填充=障碍    ▲=待执行站位    ●=回退所选候选",
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
        _txt(ax, 2.9, 8.35, L("大面积的有缺口构件", "large gapped components"), fontsize=7.6)
        _txt(ax, 7.7, 8.35, L("小尺寸的有缺口构件", "small gapped components"), fontsize=7.6)

        if k == 0:
            for x, y, w, h in big:
                for _ in range(6):
                    ax.plot(x + rng.uniform(0.3, w - 0.3), y + rng.uniform(0.3, h - 0.3),
                            "*", ms=9, color="black")
            _txt(ax, 5.5, 0.55, L("（左）按固定条数截断：名额被大面积构件占满，\n"
                                  "小尺寸的有缺口构件无表面分块入选",
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
                                  "在容量上限未触发时，各有缺口构件均有分块入选",
                                  "(b) per-component grouping: every gapped component\nrepresented when the cap is not reached"),
                 fontsize=7.4)
    fig.suptitle(L("图 6  有缺口构件的目标表面分块选取",
                   "Fig.6 Target patch selection among gapped components"),
                 fontsize=10, fontweight="bold",
                 fontproperties=_CN if USE_CN else None)
    _txt(axes[0], 5.5, -0.35, L(
        "★=入选目标分块集合的表面分块；图中构件均已按 S1(e) 判定存在缺口，"
        "无缺口的构件不参与分组",
        "star = patch entering the target set; only gapped components take part"),
        fontsize=7.0)
    _save(fig, "图6_目标分块选取")


if __name__ == "__main__":
    print(f"中文字体: {'可用' if USE_CN else '不可用（退回英文标签）'}", flush=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print("完成", flush=True)
