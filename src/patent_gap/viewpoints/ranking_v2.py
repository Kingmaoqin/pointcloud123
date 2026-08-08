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
    gamma: float = 1.0
    lambda_reg: float = 0.5
    eta: float = 0.10
    frontality_min: float = 0.0   # 正视性下限(f≤0 已自然为0)
    use_occlusion: bool = True    # 消融开关(B5 等)
    use_reg_term: bool = True
    rho0: float = 400.0
    lambda_e: float = 1.0


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
    rho_req = rho_required(imp, cfg.rho0, cfg.lambda_e)
    patch_ids = patch_scores["patch_id"].to_numpy()
    coverage = np.asarray(coverage, dtype=np.float64).reshape(J)

    out: list[CandidateScore] = []
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
        if cfg.use_reg_term and ov < cfg.o_min:
            continue

        info = float((g_task[mask_new] * area[mask_new] * q[mask_new]).sum())
        a_bar = float(area[mask_new].mean()) if mask_new.any() else 0.0
        # 冗余惩罚 O_v: 视场内已覆盖面积占比(沿母专利思想)
        denom = float(area[in_fov].sum())
        o_v = float(area[mask_ovl].sum()) / denom if denom > 0 else 0.0
        reg_gain = cfg.lambda_reg * a_bar * rr if cfg.use_reg_term else 0.0
        value = info + reg_gain - cfg.eta * o_v

        out.append(CandidateScore(
            view_id=str(cand.view_id), position=tuple(pos.tolist()),
            value=float(value), info_gain=info, reg_gain=float(reg_gain),
            overlap=float(ov), degen=float(dg), r_reg=float(rr), q_row=q,
        ))
    out.sort(key=lambda s: s.value, reverse=True)
    return out
