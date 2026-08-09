"""升级视点评估: 公式(35) q'_{j,v} 与公式(41) V'(v)。

q'_{j,v} = f_{j,v} · g^rng_{j,v} · Vis_{j,v}        (35)
V'(v)   = Σ_{j∈L_new} G_task_j·A_j·q'_{j,v}
          + λ_reg·Ā_v·R^reg(v) − η·O_v              (41)
硬约束: O^reg(v) < O_min 或 Vis 全零 → 候选剔除。
母专利(24)(25)保留在 ranking.py 供 B1 使用, 本模块不改其数值行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..density.expected import g_rng, rho_required
from ..occlusion.raycast import OcclusionOracle
from ..occlusion.sampling import PatchSampler
from ..registration.predictor import (
    O_MIN_DEFAULT, TAU_COV_DEFAULT, TAU_GAP_DEFAULT,
    degeneracy, r_reg, split_regions,
)
from ..sensors.model import SensorModel


@dataclass
class V2Config:
    tau_gap: float = TAU_GAP_DEFAULT
    tau_cov: float = TAU_COV_DEFAULT
    o_min: float = O_MIN_DEFAULT
    # 剔除候选用的重叠率门槛。缺省跟随 o_min; 软回退时单独放松它, 而 r_reg 仍
    # 按 o_min 计算 —— 否则回退会把 R^reg 悄悄换成 (1−Degen)。
    o_min_gate: float | None = None
    gamma: float = 1.0
    lambda_reg: float = 0.5
    # info 改为步内 max 归一后, 原 η=0.10 的相对权重被放大约 58 倍(旧式 info
    # 中位 5.14, 新式 0.088 < 惩罚上限 0.10) —— S/low 首轮 564 个候选里有 224 个
    # value ≤ 0, 而 B5 以 value>0 作硬门槛, 预算紧时会比改动前更早终止。B5 是
    # 预注册检验的基线, 其行为不该被本方法的归一化改动波及。按同一比例回调:
    # 0.10 / 58.33 ≈ 0.0017, 取 0.002。
    eta: float = 0.002
    frontality_min: float = 0.0   # 正视性下限(f≤0 已自然为0)
    use_occlusion: bool = True    # 消融开关(B5 等)
    use_reg_term: bool = True
    rho0: float = 400.0
    lambda_e: float = 1.0
    # 视点价值通路里 ρ_req 的重要度系数, 与证据通路(34)的 lambda_e 分开取值。
    #
    # 两者取同一个值会精确抵消: (41) 的收益权含 G_task ∝ (1+λ·E_i), (35) 的
    # g^rng = clip(ρ̂/(ρ_0(1+λ_E·E_i))), 在未饱和区(ρ̂<ρ_req)二者相乘得
    #   (1+λ·E)·ρ̂/(ρ_0(1+λ_E·E)) = ρ̂/ρ_0     (λ=λ_E 时)
    # 重要度被整除掉 —— 实测 15 m 外或斜入射处主变与杂物的单位面积权重之比
    # 恰为 1.00, 也就是公式(33) 反而抵消了(41) 想表达的优先级。
    #
    # λ_E 本就是(33) 的自由参数, 取 0 仍在公式族内: 价值通路用统一参考密度
    # ρ_0 度量"这一站能达到多少", 重要度只通过 G_task 起作用; 证据通路(34)
    # 仍按 ρ_req = ρ_0(1+E) 判定"够不够", 重要资产的密度缺口照样更大。
    #
    # 但 A/B 不支持把它设为默认(results/lambda_e_ab, 种子 10-14, 30 次运行):
    # 主指标 crit_recall 差异 +0.001 (p=1.00), awc 反而 -0.022 (p_raw 0.067)。
    # 抵消在数学上确实发生, 只是在这些场景里 crit_recall 已接近上限(0.924),
    # 约束来自可达性而非权重, 改权重无从体现。故默认保持 1.0(即升级前行为),
    # 留作可配参数; 待有"重要设备因权重被忽略"确实成为瓶颈的场景再评估。
    lambda_e_value: float = 1.0


@dataclass
class CandidateScore:
    view_id: str
    position: tuple
    value: float
    info_gain: float
    reg_gain: float
    overlap: float
    degen: float
    r_reg: float
    q_row: np.ndarray = field(repr=False)   # (J,) q'_{j,v} 全 Patch


def score_candidates_v2(
    patch_scores: pd.DataFrame,
    candidates: pd.DataFrame,
    oracle: OcclusionOracle | None,
    sampler: PatchSampler | None,
    sensor: SensorModel,
    coverage: np.ndarray,
    cfg: V2Config | None = None,
) -> list[CandidateScore]:
    """对候选站集合逐一计算 q'(35) 与 V'(41)。

    patch_scores 需含: patch_id, centroid_*, normal_*, area,
                       engineering_importance, G_gap, G_task。
    coverage: (J,) 各 Patch 当前覆盖率 C_j(公式36 的 L^ovl 判据)。
    oracle/sampler 为 None 时 Vis≡1(几何-only 降级, 消融用)。
    """
    cfg = cfg or V2Config()
    J = len(patch_scores)
    cent = patch_scores[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=np.float64)
    nrm = patch_scores[["normal_x", "normal_y", "normal_z"]].to_numpy(dtype=np.float64)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    area = patch_scores["area"].to_numpy(dtype=np.float64)
    g_gap = pd.to_numeric(patch_scores["G_gap"], errors="coerce").fillna(0).to_numpy()
    g_task = pd.to_numeric(patch_scores["G_task"], errors="coerce").fillna(0).to_numpy()
    imp = pd.to_numeric(patch_scores["engineering_importance"], errors="coerce").fillna(0.5).to_numpy()
    rho_req = rho_required(imp, cfg.rho0, cfg.lambda_e_value)
    patch_ids = patch_scores["patch_id"].to_numpy()
    coverage = np.asarray(coverage, dtype=np.float64).reshape(J)


    raw: list[tuple] = []
    for cand in candidates.itertuples(index=False):
        pos = np.asarray(cand.position, dtype=np.float64).reshape(3)
        dvec = pos[None, :] - cent
        d = np.linalg.norm(dvec, axis=1)
        d_safe = np.maximum(d, 1e-9)
        u = dvec / d_safe[:, None]
        # 母专利(23) 正视性 f = max(0, n·unit(o_v−c))
        f = np.clip((nrm * u).sum(axis=1), 0.0, 1.0)
        in_range = (d >= sensor.r_min) & (d <= sensor.r_max)
        in_fov = in_range & (f > cfg.frontality_min)

        # (33) g_rng, cosα = f (正视性即 |n·u| 取正部)
        g = g_rng(d, f, rho_req, sensor)

        # (27)(28) Vis
        if cfg.use_occlusion and oracle is not None and sampler is not None:
            active = np.where(in_fov & (g > 0))[0]
            vis = np.zeros(J)
            if len(active):
                vmap = oracle.visibility_batch(pos, sampler,
                                               [int(patch_ids[k]) for k in active],
                                               sensor=sensor)
                for k in active:
                    vis[k] = vmap.get(int(patch_ids[k]), 0.0)
        else:
            vis = np.ones(J)

        q = f * g * vis * in_fov  # (35)

        # (36) 双区域
        mask_new, mask_ovl = split_regions(g_gap, coverage, in_fov, cfg.tau_gap, cfg.tau_cov)
        # (37) 重叠率 — 实现精化: 用连续覆盖率 C_j 加权代替二值 L^ovl
        # (二值阈值会把面向缺口的候选整体判零重叠, 早期覆盖低时误杀所有有效候选;
        #  连续加权是(37)的平滑化, mask_ovl 仍用于(38)锚定点选取。见 OPEN_ISSUES #16)
        aq = area * q * in_fov
        denom_aq = float(aq.sum())
        ov = float((aq * np.clip(coverage, 0, 1)).sum()) / (denom_aq + 1e-6) if denom_aq > 0 else 0.0
        # (38) 退化度: 锚定点 = L^ovl 采样点
        if cfg.use_reg_term and sampler is not None and mask_ovl.any():
            pts_list, nrm_list = [], []
            for k in np.where(mask_ovl)[0]:
                s = sampler.samples(int(patch_ids[k]))
                if len(s):
                    pts_list.append(s)
                    nrm_list.append(np.tile(nrm[k], (len(s), 1)))
            if pts_list:
                dg = degeneracy(np.vstack(pts_list), np.vstack(nrm_list))
            else:
                dg = 1.0
        else:
            dg = 1.0
        rr = r_reg(ov, dg, cfg.o_min, cfg.gamma) if cfg.use_reg_term else 0.0

        # 硬约束(41 末段)
        if (q > 0).sum() == 0:
            continue
        gate = cfg.o_min if cfg.o_min_gate is None else cfg.o_min_gate
        if cfg.use_reg_term and ov < gate:
            continue

        info_raw = float((g_task[mask_new] * area[mask_new] * q[mask_new]).sum())
        # 冗余惩罚 O_v: 视场内已覆盖面积占比(沿母专利思想)
        denom = float(area[in_fov].sum())
        o_v = float(area[mask_ovl].sum()) / denom if denom > 0 else 0.0
        # 公式(41) 三项量纲配平(实现精化, 见 OPEN_ISSUES #36)。
        # 原式 value = info + λ_reg·Ā_v·R^reg − η·O_v 里, info 是**广延量**(随
        # 视场内缺口 Patch 数与面积增长), 而 Ā_v 是均值、O_v 是 [0,1] 比值。
        # 实测三项中位数之比: S/low 上 reg/info 已只有 0.64%, M/mid 0.07%,
        # L/high **0.02%** —— 公式(41) 在大场景里退化成纯 info 项, B10 数值上
        # 等价于"B5 + 一个重叠率硬门", 配准感知(36)-(40) 被场景规模稀释掉了。
        # 按公式(44) 自己已经采用的做法, 用 F_ub = Σ_j G_task_j·A_j 归一 info,
        # 三项同落 [0,1], λ_reg=0.5 / η=0.10 才在任何规模下表达同一设计意图。
        # 归一后 Ā_v 不再需要(它原本就是为把 reg 项抬到 info 量级而设)。
        raw.append((str(cand.view_id), tuple(pos.tolist()), info_raw,
                    float(rr), float(ov), float(dg), float(o_v), q))

    # 三项在**本轮候选内**配平后再定值。归一基准取本轮 info 的最大值: info 读作
    # "相对本轮最好那一站拿到了几成信息", λ_reg=0.5 于是读作"配准支持最多抵得上
    # 最佳候选信息量的一半"。用全场 F_ub 归一则 info 只有 0.06 量级, λ_reg 仍按
    # 旧量纲取 0.5 会让配准项直接压过信息项(实测 S/low 上 reg/info 达 126%)。
    # 与 _pick() 的单步 min-max 归一同一条纪律。
    if not raw:
        return []
    info_scale = max(r[2] for r in raw) or 1.0
    out: list[CandidateScore] = []
    for view_id, pos_t, info_raw, rr, ov, dg, o_v, q in raw:
        info = info_raw / info_scale
        reg_gain = cfg.lambda_reg * rr if cfg.use_reg_term else 0.0
        out.append(CandidateScore(
            view_id=view_id, position=pos_t,
            value=float(info + reg_gain - cfg.eta * o_v),
            info_gain=float(info), reg_gain=float(reg_gain),
            overlap=float(ov), degen=float(dg), r_reg=float(rr), q_row=q,
        ))
    out.sort(key=lambda s: s.value, reverse=True)
    return out
