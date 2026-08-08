"""公式(45) 真实重扫闭环 + E2 方法实现(B0/B1/B5/B10)。

替换母专利实施例9的固定恢复率 0.75: 执行推荐站后, 直接用降级仿真器
(FallbackSimulator, raycast)重扫得到真实新增点云 → 重算关联与全部证据
→ G_gap 更新 → 重新选站定序。恢复率是测量结果, 不是假设参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..gap.scoring import compute_patch_scores
from ..io.station_npz import StationScan
from ..mapping.traversability import TravGrid
from ..occlusion.discovery import UnmodeledOccluders
from ..occlusion.raycast import OcclusionOracle
from ..occlusion.sampling import PatchSampler
from ..planning.objective import ObjectiveParams
from ..planning.route import solve_tsp
from ..planning.set_select import lazy_greedy
from ..registration.predictor import degeneracy, expected_overlap, r_reg, split_regions
from ..registration.realism import register_station
from ..sensors.model import SensorModel
from ..viewpoints.ranking import generate_candidates, score_candidates
from ..viewpoints.ranking_v2 import V2Config, score_candidates_v2
from .helios_bridge import FallbackSimulator
from .scene_gen import SceneModel, build_trav_grid
from .scene_patches import build_scene_patches

TRIPOD_Z = 2.0


# ---------------------------------------------------------------- world

@dataclass
class SimWorld:
    scene: SceneModel
    patches: pd.DataFrame
    tri_to_patch: np.ndarray
    oracle: OcclusionOracle        # 竣工实景几何: 仿真与评测用
    sampler: PatchSampler
    grid: TravGrid
    sim: FallbackSimulator
    sensor: SensorModel        # 仿真实际角步长的 Θ
    plan_oracle: OcclusionOracle | None = None   # 设计 BIM 几何: 规划器用

    @classmethod
    def build(cls, scene: SceneModel, sensor: SensorModel,
              sim_dtheta_deg: float = 0.3) -> "SimWorld":
        patches, tri_to_patch = build_scene_patches(scene)
        oracle = OcclusionOracle(scene.vertices, scene.triangles, tri_to_patch)
        # 规划器只有设计 BIM: 竣工态临时占位物挡得住扫描仪, 却挡不住规划器的
        # 预测。两套几何分开, 公式(27)(28) 的 Vis 才是可能出错的预测量, 而不是
        # 与评测真值同源的恒等式(否则"遮挡感知有效"无法被证伪)。
        # n_temp=0 时两者逐比特相同, 既有基准结果不受影响。
        bim = scene.bim_tri_mask()
        plan_oracle = (oracle if bool(bim.all()) else
                       OcclusionOracle(scene.vertices, scene.triangles[bim],
                                       tri_to_patch[bim]))
        sampler = PatchSampler(scene.vertices, scene.triangles, tri_to_patch)
        grid = build_trav_grid(scene)
        sim = FallbackSimulator(scene.vertices, scene.triangles, tri_to_patch,
                                sensor, sim_dtheta_deg=sim_dtheta_deg)
        return cls(scene=scene, patches=patches, tri_to_patch=tri_to_patch,
                   oracle=oracle, sampler=sampler, grid=grid, sim=sim,
                   sensor=sim.effective_sensor(), plan_oracle=plan_oracle)


def station_sample_masks(world: SimWorld, origin: np.ndarray) -> dict[int, np.ndarray]:
    """单站对全部 Patch 采样点的可见掩码(公式27, 含量程门)。"""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    pids = world.sampler.patch_ids()
    samples, owners, counts = [], [], []
    for pid in pids:
        s = world.sampler.samples(pid)
        counts.append(len(s))
        if len(s):
            samples.append(s)
            owners.append(np.full(len(s), pid, dtype=np.int64))
    samples = np.vstack(samples)
    owner = np.concatenate(owners)
    d = samples - origin[None, :]
    dist = np.linalg.norm(d, axis=1)
    t_hit, hit_patch = world.oracle.first_hits(origin, d / np.maximum(dist, 1e-9)[:, None])
    s = world.sensor
    ok = (hit_patch == owner) & (t_hit >= dist - 0.01) & (dist >= s.r_min) & (dist <= s.r_max)
    out: dict[int, np.ndarray] = {}
    pos = 0
    for pid, c in zip(pids, counts):
        out[pid] = ok[pos:pos + c]
        pos += c
    return out


# ---------------------------------------------------------------- observation state

@dataclass
class ObsState:
    """已执行站集 S_obs 的累积观测。"""
    world: SimWorld
    scans: list[StationScan] = field(default_factory=list)
    masks: list[dict[int, np.ndarray]] = field(default_factory=list)   # 逐站样本可见掩码

    def add_station(self, origin: np.ndarray, station_id: str, seed: int = 0) -> StationScan:
        scan = self.world.sim.scan(origin, station_id, seed=seed)
        self.scans.append(scan)
        self.masks.append(station_sample_masks(self.world, origin))
        return scan

    def drop_last_station(self) -> None:
        """配准失败的一站拼不进全局坐标系, 其数据整体不可用(路程与时间照付)。"""
        if self.scans:
            self.scans.pop()
            self.masks.pop()

    def registration_of_last(self) -> tuple[float, int, float]:
        """新站相对既有站集的**实际**重叠率、重叠点数与重叠区退化度。

        与 score_candidates_v2 里对候选做的预测同构, 区别是这里用真正测到的
        可见掩码, 而不是对 BIM 的预测 —— 被预测的量与预测量必须分开。
        """
        if len(self.masks) < 2:
            return 1.0, 10 ** 6, 0.0
        pids = self.world.sampler.patch_ids()
        new = self.masks[-1]
        prior = self.masks[:-1]
        pts, nrms, n_new, n_ovl = [], [], 0, 0
        p = self.world.patches
        nrm_all = p[["normal_x", "normal_y", "normal_z"]].to_numpy()
        nrm_all = nrm_all / np.maximum(np.linalg.norm(nrm_all, axis=1, keepdims=True), 1e-9)
        idx = {int(v): k for k, v in enumerate(p["patch_id"].to_numpy())}
        for pid in pids:
            m = new[pid]
            if not len(m) or not m.any():
                continue
            seen = np.zeros(len(m), dtype=bool)
            for pm in prior:
                seen |= pm[pid]
            both = m & seen
            n_new += int(m.sum())
            n_ovl += int(both.sum())
            k = idx.get(int(pid))
            if k is not None and both.any():
                sm = self.world.sampler.samples(int(pid))
                # 掩码由 station_sample_masks 按同一 sampler 切片得到, 长度必然一致
                assert len(sm) == len(both), f"patch {pid}: 采样点与掩码长度不一致"
                pts.append(sm[both])
                nrms.append(np.tile(nrm_all[k], (int(both.sum()), 1)))
        if n_new == 0:
            return 0.0, 0, 1.0
        dg = degeneracy(np.vstack(pts), np.vstack(nrms)) if pts else 1.0
        return n_ovl / n_new, n_ovl, dg

    # ---- 逐 Patch 聚合量 ----
    def coverage(self) -> np.ndarray:
        """C_i: 采样点被任一已执行站看到的比例。"""
        pids = self.world.sampler.patch_ids()
        C = np.zeros(len(pids))
        for k, pid in enumerate(pids):
            acc = None
            for m in self.masks:
                acc = m[pid] if acc is None else (acc | m[pid])
            C[k] = float(acc.mean()) if acc is not None and len(acc) else 0.0
        return C

    def vis_per_station(self) -> list[np.ndarray]:
        pids = self.world.sampler.patch_ids()
        return [np.array([m[pid].mean() if len(m[pid]) else 0.0 for pid in pids])
                for m in self.masks]

    def point_counts(self) -> np.ndarray:
        pids = self.world.sampler.patch_ids()
        idx = {pid: k for k, pid in enumerate(pids)}
        n = np.zeros(len(pids))
        for scan in self.scans:
            if scan.hit_patch is None:
                continue
            u, c = np.unique(scan.hit_patch[scan.hit_patch >= 0], return_counts=True)
            for pid, cnt in zip(u, c):
                if int(pid) in idx:
                    n[idx[int(pid)]] += cnt
        return n

    def best_frontality(self) -> np.ndarray:
        cent = self.world.patches[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
        nrm = self.world.patches[["normal_x", "normal_y", "normal_z"]].to_numpy()
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        best = np.zeros(len(cent))
        for scan in self.scans:
            u = scan.origin[None, :] - cent
            u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
            best = np.maximum(best, np.clip((nrm * u).sum(axis=1), 0, 1))
        return best

    def n_valid_views(self, vis_thr: float = 0.2) -> np.ndarray:
        vs = self.vis_per_station()
        if not vs:
            return np.zeros(len(self.world.patches))
        return np.sum(np.stack(vs) > vis_thr, axis=0).astype(float)


def station_rreg(world: SimWorld, origin: np.ndarray, coverage: np.ndarray,
                 g_gap: np.ndarray, cfg: V2Config) -> tuple[float, np.ndarray]:
    """对已执行站计算 R^reg(v) 与其视场掩码(公式39/40 回填用)。"""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    p = world.patches
    cent = p[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    nrm = p[["normal_x", "normal_y", "normal_z"]].to_numpy()
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    area = p["area"].to_numpy()
    d = np.linalg.norm(origin[None, :] - cent, axis=1)
    u = (origin[None, :] - cent) / np.maximum(d, 1e-9)[:, None]
    f = np.clip((nrm * u).sum(axis=1), 0, 1)
    s = world.sensor
    in_fov = (d >= s.r_min) & (d <= s.r_max) & (f > 0)
    mask_new, mask_ovl = split_regions(g_gap, coverage, in_fov, cfg.tau_gap, cfg.tau_cov)
    ov = expected_overlap(area, f, mask_ovl, in_fov)
    if mask_ovl.any():
        pts, ns = [], []
        pids = p["patch_id"].to_numpy()
        for k in np.where(mask_ovl)[0]:
            sm = world.sampler.samples(int(pids[k]))
            if len(sm):
                pts.append(sm)
                ns.append(np.tile(nrm[k], (len(sm), 1)))
        dg = degeneracy(np.vstack(pts), np.vstack(ns)) if pts else 1.0
    else:
        dg = 1.0
    return r_reg(ov, dg, cfg.o_min, cfg.gamma), in_fov


# ---------------------------------------------------------------- evidence → scores

def compute_scores(obs: ObsState, extended: bool, rho0: float = 400.0,
                   lambda_e: float = 1.0, v2cfg: V2Config | None = None) -> pd.DataFrame:
    """由 S_obs 观测构造证据表并调用母专利评分(公式18/19 K_i 机制)。

    extended=False → 母专利六证据(B1 路径, 数值行为不变);
    extended=True  → 追加 D_occ(29)/D_rng(34)/D_regsup(40), 扩展权重表。
    """
    from ..density.expected import range_density_gap, rho_required

    world = obs.world
    p = world.patches
    C = obs.coverage()
    n_pts = obs.point_counts()
    rho_req = rho_required(p["engineering_importance"].to_numpy(), rho0, lambda_e)
    density_ratio = (n_pts / np.maximum(p["area"].to_numpy(), 1e-9)) / np.maximum(rho_req, 1e-9)
    best_f = obs.best_frontality()
    n_views = obs.n_valid_views()

    observations = pd.DataFrame({
        "patch_id": p["patch_id"].to_numpy(),
        "directly_observed": C > 0.05,
        "number_of_valid_views": n_views,
        "projected_resolution_quality": np.clip(density_ratio, 0, 1),
        "registration_confidence": 0.5,          # B1 中性值; 升级路径由(39)回填
        "best_frontality": best_f,
        "angular_diversity": np.clip(n_views / 4.0, 0, 1),
        "coverage_ratio": C,
        "density_ratio": np.clip(density_ratio, 0, 1),
    })

    if extended:
        cfg = v2cfg or V2Config()
        # (29) D_occ
        vs = obs.vis_per_station()
        best_vis = np.max(np.stack(vs), axis=0) if vs else np.zeros(len(p))
        observations["D_occ"] = 1.0 - best_vis
        # (34) D_rng
        observations["D_rng"] = range_density_gap(n_pts, p["area"].to_numpy(), rho_req)
        # (40) D_regsup(需 g_gap 先验: 用未扩展评分的 G_gap 迭代一次)
        base = compute_scores(obs, extended=False, rho0=rho0, lambda_e=lambda_e)
        base = base.sort_values("patch_id")
        g_gap0 = base["G_gap"].fillna(1.0).to_numpy()
        best_r = np.zeros(len(p))
        for scan in obs.scans:
            rr, in_fov = station_rreg(world, scan.origin, C, g_gap0, cfg)
            best_r = np.where(in_fov, np.maximum(best_r, rr), best_r)
        observations["D_regsup"] = 1.0 - best_r
        # (39)→(11): 配准置信度回填
        observations["registration_confidence"] = np.clip(best_r, 0, 1)

    scene_tables = {
        "patches": p,
        "observations": observations,
        "semantics": pd.DataFrame({"patch_id": p["patch_id"].to_numpy()}),   # D_sem 不可算→K_i退出
        "materials": pd.DataFrame({"patch_id": p["patch_id"].to_numpy()}),   # D_mat 同上
    }
    scores = compute_patch_scores(scene_tables)
    # 评分输出仅保留 centroid/normal 元组列; v2 评分与候选生成需要分量列 → 回填
    coord_cols = ["patch_id", "centroid_x", "centroid_y", "centroid_z",
                  "normal_x", "normal_y", "normal_z"]
    return scores.merge(p[coord_cols], on="patch_id", how="left", validate="one_to_one")


# ---------------------------------------------------------------- gt (1.2.6)

def build_ground_truth(world: SimWorld, init_masks: list[dict[int, np.ndarray]],
                       full_spacing: float = 8.0) -> dict:
    """Patch 级缺扫真值: S_full 网格撒站(剔除禁入区) → C_gt; y=I[C_init<0.5·C_gt]。"""
    xmin, ymin, xmax, ymax = world.scene.bounds_xy
    full_origins = []
    for x in np.arange(xmin + 2, xmax - 2, full_spacing):
        for y in np.arange(ymin + 2, ymax - 2, full_spacing):
            if world.grid.is_free((x, y)):
                full_origins.append(np.array([x, y, TRIPOD_Z]))
    pids = world.sampler.patch_ids()
    acc = {pid: None for pid in pids}
    for o in full_origins:
        m = station_sample_masks(world, o)
        for pid in pids:
            acc[pid] = m[pid] if acc[pid] is None else (acc[pid] | m[pid])
    C_gt = np.array([acc[pid].mean() if acc[pid] is not None and len(acc[pid]) else 0.0
                     for pid in pids])
    accI = {pid: None for pid in pids}
    for m in init_masks:
        for pid in pids:
            accI[pid] = m[pid] if accI[pid] is None else (accI[pid] | m[pid])
    C_init = np.array([accI[pid].mean() if accI[pid] is not None and len(accI[pid]) else 0.0
                       for pid in pids])
    y = C_init < 0.5 * C_gt
    return {"C_gt": C_gt, "C_init": C_init, "y": y,
            "n_full_stations": len(full_origins)}


# ---------------------------------------------------------------- metrics

def episode_metrics(world: SimWorld, gt: dict, C_now: np.ndarray,
                    n_pts: np.ndarray, path_len: float, n_stations: int,
                    rho0: float = 400.0, lambda_e: float = 1.0) -> dict:
    from ..density.expected import rho_required

    p = world.patches
    area = p["area"].to_numpy()
    imp = p["engineering_importance"].to_numpy()
    y = gt["y"]
    C_gt, C_init = gt["C_gt"], gt["C_init"]
    denom = np.maximum(C_gt - C_init, 1e-9)
    rec = np.clip((C_now - C_init) / denom, 0, 1)
    w = area * (1 + lambda_e * imp)
    awc = float((rec[y] * w[y]).sum() / max(w[y].sum(), 1e-9)) if y.any() else float("nan")
    rho_req = rho_required(imp, rho0, lambda_e)
    dens_ok = float(((n_pts / np.maximum(area, 1e-9)) >= rho_req).mean())
    crit = y & (imp >= 0.8)
    crit_recall = float((C_now[crit] >= 0.5 * C_gt[crit]).mean()) if crit.any() else float("nan")
    # 按 BIM 构件等权的恢复率。awc 以面积加权, 实测在 S/low 上 4 个主变面 +
    # 5 个建筑面就占了 92% 的权重, 十几个断路器/互感器合计不到 4% —— 一个只
    # 刷大平面的方案能拿到很高的 awc。验收关心的是"每个资产是否都扫到了",
    # 所以构件内按面积加权, 构件之间等权。
    guid = p["element_guid"].to_numpy()
    per_asset = []
    for g in np.unique(guid[y]):
        m = y & (guid == g)
        wa = w[m]
        per_asset.append(float((rec[m] * wa).sum() / max(wa.sum(), 1e-9)))
    asset_recovery = float(np.mean(per_asset)) if per_asset else float("nan")
    return {
        "awc_gap_recovery": awc,
        "asset_recovery": asset_recovery,
        "dens_ok": dens_ok,
        "crit_recall": crit_recall,
        "path_len_m": float(path_len),
        "n_stations": int(n_stations),
        "ig_per_m": awc / max(path_len, 1e-9) if np.isfinite(awc) else float("nan"),
    }


# ---------------------------------------------------------------- closed loop (45)

def _make_candidates(world: SimWorld, scores: pd.DataFrame, v0_xy, rng,
                     distances=(4.0, 8.0, 14.0), grid_spacing: float = 10.0,
                     executed_xy: list | None = None, r_dup: float = 3.0) -> pd.DataFrame:
    """母专利(21)(22)几何候选 → 投影到可通行图(公式43) → 剔除不可达。

    目标 Patch 按 G_task·A 排序并做构件级去重(每构件≤2), 避免候选池
    坍缩到单一高遮挡簇; 另按 2.0 节"候选点集改为可通行图节点约束"补充
    可通行网格节点候选(空间多样性, 供公式(42)集合选择淘汰)。

    目标集容量随构件数缩放, 不再固定 60 行。固定 60 时排序键 G_task·A 里
    G_task 只跨 1.0–1.3 而 Patch 面积跨约 200 倍, 重要度永远抬不过面积, 于是
    截断处的面积下限正好卡在断路器尺寸上 —— 实测即使在最小的 S/low 上,
    隔离开关 0/144、CT/PT 0/72、避雷器 0/72、绝缘子 0/20 **从未有过一个为它们
    生成的候选视点**(L/high 为 0/576、0/288、0/288、0/40), 而隔离开关与 CT/PT
    的 E_i=0.8 是计入 crit_recall 的。取每构件 2 个的容量即可让全部构件进入
    目标集。
    """
    s = scores.copy()
    s["_gain"] = pd.to_numeric(s["G_task"], errors="coerce").fillna(0) * s["area"]
    s = s.sort_values("_gain", ascending=False)
    n_targets = int(np.clip(2 * s["element_guid"].nunique(), 60, 400))
    s = s.groupby("element_guid", sort=False).head(2).head(n_targets)
    cand = generate_candidates(s, {"distances": list(distances),
                                   "gap_threshold": 0.0,
                                   "max_target_patches": len(s)})
    rows, seen = [], set()

    exec_arr = (np.asarray([e[:2] for e in executed_xy], dtype=float)
                if executed_xy else np.zeros((0, 2)))
    # 一次单源 Dijkstra 取代逐候选 A*: 同一套 8 邻域权重与贴角规则, 距离逐位
    # 一致(实测最大差 0.000000), L/high 上 550 个候选省约 175 倍时间。
    dfield = world.grid.distance_field(v0_xy)

    def _add(xy, target_pid, orientation):
        proj = world.grid.project_to_free(xy, max_dist=2.0)
        if proj is None:
            return
        # 已执行站 r_dup 内的重复架站无新信息(确定性仿真), 剔除防原地重扫
        if len(exec_arr) and np.min(np.linalg.norm(exec_arr - np.asarray(proj), axis=1)) < r_dup:
            return
        cell = world.grid.to_ij(proj)
        if cell in seen:
            return
        seen.add(cell)
        d = float(dfield[cell])
        if not np.isfinite(d):
            return
        rows.append({"view_id": f"c{len(rows):04d}",
                     "target_patch_id": target_pid,
                     "position": (proj[0], proj[1], TRIPOD_Z),
                     "orientation": orientation,
                     "astar_from_v0": d})

    for c in cand.itertuples(index=False):
        _add(c.position[:2], c.target_patch_id, c.orientation)
    xmin, ymin, xmax, ymax = world.scene.bounds_xy
    for x in np.arange(xmin + 3, xmax - 3, grid_spacing):
        for y in np.arange(ymin + 3, ymax - 3, grid_spacing):
            _add((float(x), float(y)), -1, (1.0, 0.0, 0.0))
    return pd.DataFrame(rows)


@dataclass
class EpisodeConfig:
    stations_max: int = 6
    length_max_m: float = 400.0
    rounds_max: int = 6
    rho0: float = 400.0
    lambda_e: float = 1.0
    # 见 V2Config.lambda_e_value: 价值通路的 ρ_req 重要度系数。取 1.0 时它与
    # G_task 的 (1+E) 在未饱和区精确抵消; 取 0.0 解除抵消。A/B 不支持改默认,
    # 保持 1.0, 见 results/lambda_e_ab/summary.md。
    lambda_e_value: float = 1.0
    seed: int = 0
    method: str = "B10_full"
    # 是否用已测点云在线发现 BIM 未建模的遮挡物并修正规划遮挡模型(见
    # occlusion/discovery.py)。B11_disc 即 B10 + 该项。
    discover_occluders: bool = False
    # 是否按实际重叠与退化度判定每一站的配准成败(见 registration/realism.py)。
    # 关闭时所有站无条件拼合, 公式(36)-(40) 预测的量在世界里不存在。
    registration_realism: bool = False
    reg_tol_m: float = 0.05
    # 闭环中"本轮执行哪一站"的策略（见 docs/闭环执行策略升级.md）：
    #   "tsp_first"    公式(45)原定：执行 TSP 路线首站（为走完整条路线而排序）
    #   "greedy_first" 执行懒惰贪心的首选（性价比最高者）
    #   "j_step"       执行使单步 J 增量最大者（公式(44) 的单步形式）
    exec_policy: str = "j_step"


def _select_next_station(world: SimWorld, obs: ObsState, cfg: EpisodeConfig,
                         v0_xy, rng, budget_left: float,
                         plan_oracle=None,
                         failed_xy: list | None = None) -> tuple[np.ndarray, float] | None:
    """一轮选站(方法分派), 返回 (下一站原点, A*距离) 或 None。

    plan_oracle 为规划器可用的遮挡模型; 缺省即设计 BIM。B11_disc 会传入被
    已发现的未建模遮挡物增量修正过的版本。

    failed_xy 为配准失败而被丢弃的站位。它们必须与已执行站一样进入去重集:
    失败站已从 obs.scans 弹出, 若不另行记住, 规划器看到的就是"这里没人去过"
    且 A* 距离为 0(机器人就站在那儿), 成本必然最低而被反复重选 —— 实测会在
    同一点空转满全部轮次, awc 归零。"这里配准不上"正是配准感知规划该拿到的
    信号。
    """
    method = cfg.method
    plan_oracle = plan_oracle if plan_oracle is not None else world.plan_oracle
    if method == "B0_random":
        free_cells = np.argwhere(world.grid._compute_free())
        for _ in range(200):
            i, j = free_cells[rng.integers(len(free_cells))]
            xy = world.grid.to_xy((i, j))
            d, _ = world.grid.astar(v0_xy, xy)
            if np.isfinite(d) and d <= budget_left:
                return np.array([xy[0], xy[1], TRIPOD_Z]), d
        return None

    if method == "Bdisp_maxmin":
        # 平凡对照: 完全不使用任何缺口证据/可见性/密度模型, 只在可通行自由格中
        # 取"离已执行站最远"(max-min 离散度)。用于检验公式(26)-(45) 相对一个
        # 无信息启发式是否真有优势 —— 若无, 整套机制的收益主张不成立。
        free_cells = np.argwhere(world.grid._compute_free())
        done = np.array([s.origin[:2] for s in obs.scans] + list(failed_xy or []),
                        dtype=float)
        best, best_d, best_score = None, None, -np.inf
        idx = rng.permutation(len(free_cells))[:400]   # 固定预算, 与 B0 同量级
        for t in idx:
            i, j = free_cells[t]
            xy = np.asarray(world.grid.to_xy((i, j)), dtype=float)
            sep = float(np.min(np.linalg.norm(done - xy[None, :], axis=1))) if len(done) else 1e9
            if sep <= best_score:
                continue
            d, _ = world.grid.astar(v0_xy, xy)
            if not np.isfinite(d) or d > budget_left:
                continue
            best, best_d, best_score = xy, d, sep
        if best is None:
            return None
        return np.array([best[0], best[1], TRIPOD_Z]), best_d

    extended = method not in ("B1_patent",)
    scores = compute_scores(obs, extended=extended, rho0=cfg.rho0, lambda_e=cfg.lambda_e)
    scores = scores.sort_values("patch_id").reset_index(drop=True)
    cand = _make_candidates(world, scores, v0_xy, rng,
                            executed_xy=([s.origin[:2] for s in obs.scans]
                                         + list(failed_xy or [])))
    if cand.empty:
        return None

    if method == "B1_patent":
        # 母专利(24)(25)原样
        ranked = score_candidates(scores, cand, {"max_range_m": world.sensor.r_max})
        if ranked.empty or float(ranked.iloc[0]["value"]) <= 0:
            return None
        for row in ranked.itertuples(index=False):
            d = float(cand.loc[cand["view_id"] == row.view_id, "astar_from_v0"].iloc[0])
            if d <= budget_left:
                return np.asarray(row.position, dtype=float), d
        return None

    # v2 路径(B5/B10)
    v2cfg = V2Config(use_reg_term=(method != "B5_occ_rng"),
                     rho0=cfg.rho0, lambda_e=cfg.lambda_e,
                     lambda_e_value=cfg.lambda_e_value)
    C = obs.coverage()
    cs = score_candidates_v2(scores, cand, plan_oracle, world.sampler,
                             world.sensor, C, v2cfg)
    if not cs and v2cfg.use_reg_term:
        # 硬约束 O^reg<O_min 全灭(早期覆盖过低)→ 软回退: 去掉重叠门重评
        import dataclasses
        cs = score_candidates_v2(scores, cand, plan_oracle, world.sampler,
                                 world.sensor, C,
                                 dataclasses.replace(v2cfg, o_min=1e-9))
    if not cs:
        return None
    if method == "B5_occ_rng":
        for s in cs:
            d = float(cand.loc[cand["view_id"] == s.view_id, "astar_from_v0"].iloc[0])
            if d <= budget_left and s.value > 0:
                return np.asarray(s.position, dtype=float), d
        return None

    # B10_full: 懒惰贪心(42) + TSP(45.2), 执行首站(45.3)
    # 收益与公式(41)的 L^new 一致: 只计缺口 Patch(G_gap≥τ_gap), 否则大面积
    # 中等分数表面会稀释集合函数, 把站集拉向"全场刷密度"
    g_task = pd.to_numeric(scores["G_task"], errors="coerce").fillna(0).to_numpy()
    g_gap_now = pd.to_numeric(scores["G_gap"], errors="coerce").fillna(0).to_numpy()
    area = scores["area"].to_numpy()
    gains = g_task * area * (g_gap_now >= V2Config().tau_gap)
    c_jv = np.stack([s.q_row for s in cs], axis=1)
    # cost_v = A*距离 + t_scan·v_move 当量(时间折算为等效距离, 单一预算维度);
    # 缺少扫描当量会让比值贪心过度偏好近站(原地重扫), 见 3.6 节
    scan_equiv = 180.0 * 0.5
    dist_v0 = np.array([float(cand.loc[cand["view_id"] == s.view_id, "astar_from_v0"].iloc[0])
                        for s in cs])
    cost = dist_v0 + scan_equiv
    n_left = cfg.stations_max - len([s for s in obs.scans if not s.station_id.startswith("init")])
    sel = lazy_greedy(c_jv, gains, cost,
                      budget=budget_left + max(n_left, 1) * scan_equiv,
                      max_stations=max(n_left, 1))
    uncov_now = gains.copy()          # 单步 ΔF 用：当前尚未覆盖的期望收益
    # 集合选择用的是**整条路线**的预算, 本轮只执行一站; 当剩余路径预算很紧时,
    # 选出的整个站集可能没有一站走得到。此时不该终止 episode(实测 S/low 在
    # length_max=25 m 下站数=0、awc=0, 而付得起的候选有两百多个), 而应退而求其
    # 次: 在预算内的候选里取公式(41) 价值最高的一个。
    reachable = [k for k in range(len(cs)) if dist_v0[k] <= budget_left]
    if not any(dist_v0[k] <= budget_left for k in sel):
        if not reachable:
            return None
        sel = [max(reachable, key=lambda k: cs[k].value)]
    if not sel:
        return None
    pts_xy = [v0_xy] + [cs[k].position[:2] for k in sel]
    D = world.grid.distance_matrix(pts_xy)
    # 仅 tsp_first 策略需要完整路线；其余策略下 route 会被丢弃，而 solve_tsp 的
    # 时限搜索每轮要跑满 3 s（实测占单次 episode 用时的 26%），故按需调用。
    route = solve_tsp(D, start=0, time_limit_s=3) if cfg.exec_policy == "tsp_first" else []

    # 本轮实际执行哪一站。公式(45)原定取 TSP 路线首站，但 TSP 是为"走完整条
    # 路线"排序的；闭环每轮只执行一站即重规划，取 TSP 首站等于系统性地挑最近
    # 而非最有价值的站。实测（docs/闭环执行策略升级.md）：在 B10 表现最差的场景中，
    # TSP 首站每轮的边际增益仅为贪心首选的 1/5～1/2。故默认改为单步 J 最大。
    # 只在预算内付得起的站里选。公式(42) 的 cost 是**时间当量**
    # (dist + t_scan·v_move), 而 length_max_m 是**纯路径**预算; 懒惰贪心的预算
    # 又按 budget_left + n_left·scan_equiv 抬高过, 两者只在恰好选满 n_left 站时
    # 等价。选少了等式就松, 于是 sel 里可能全是走不到的站 —— 实测 S/low 在
    # length_max=25 m 时 Σdist=113 m 远超预算, 而付得起的候选其实有 202 个,
    # 原实现却直接 return None 终止整个 episode, awc 归零。
    afford = [i for i in range(len(sel)) if float(D[0, i + 1]) <= budget_left]

    def _pick() -> int | None:
        if not afford:
            return None
        if cfg.exec_policy == "greedy_first":
            return afford[0]
        if cfg.exec_policy == "tsp_first":
            for node in route[1:]:
                if node > 0 and (node - 1) in afford:
                    return node - 1
            return afford[0]
        # j_step：公式(44) 的单步形式。
        # 注意归一化基准：公式(44) 用 F_ub（全部表面分块的可达总收益）归一信息项、
        # 用 L_diag（场景对角线）归一路径项，这在"整条轨迹"粒度上是配平的；但在
        # 单步粒度上，一站只能拿到 F_ub 的百分之几，而一条腿却可达 L_diag 的三成，
        # 距离项会比信息项大一到两个数量级，取极大即退化为"挑最近"（与 tsp_first
        # 同解，实测 8 组配对中 6 组逐位相同）。故单步改用**步内 min-max 归一**：
        # 两项都落在 [0,1]，λ_reg / λ_len 才在这一粒度上表达设计意图。
        op = ObjectiveParams()
        dF = np.array([float((uncov_now * c_jv[:, sel[i]]).sum()) for i in afford])
        dist = np.array([float(D[0, i + 1]) for i in afford])
        f_scale = float(dF.max()) or 1.0
        d_scale = float(dist.max()) or 1.0
        rr = np.array([cs[sel[i]].r_reg for i in afford])
        j = dF / f_scale + op.lambda_reg * rr - op.lambda_len * dist / d_scale
        return afford[int(np.argmax(j))]

    k_local = _pick()
    if k_local is None:
        return None
    nxt = sel[k_local]
    d_first = float(D[0, k_local + 1])
    assert d_first <= budget_left
    return np.asarray(cs[nxt].position, dtype=float), d_first


def run_episode(world: SimWorld, init_origins: list[np.ndarray],
                cfg: EpisodeConfig, gt: dict | None = None) -> dict:
    """完整闭环: S_init → (证据→选站→重扫)×rounds → 指标轨迹。

    gt 可传入复用(真值只依赖 world 与 init_origins, 与方法无关)。
    """
    rng = np.random.default_rng(cfg.seed)
    obs = ObsState(world=world)
    for k, o in enumerate(init_origins):
        obs.add_station(o, f"init_{k}", seed=cfg.seed * 100 + k)
    if gt is None:
        gt = build_ground_truth(world, list(obs.masks))

    # 未建模遮挡物在线发现: 初始站的点云先并入, 首轮选站就能用上
    disc = None
    reg_log: list[dict] = []
    failed_xy: list[np.ndarray] = []
    plan_oracle = world.plan_oracle
    if cfg.discover_occluders or cfg.method == "B11_disc":
        bim = world.scene.bim_tri_mask()
        disc = UnmodeledOccluders(world.scene.vertices, world.scene.triangles[bim],
                                  world.tri_to_patch[bim], seed=cfg.seed)
        for scan in obs.scans:
            disc.update(scan.points)
        plan_oracle = disc.build_oracle()

    v0_xy = tuple(init_origins[-1][:2])
    path_len = 0.0
    history = [episode_metrics(world, gt, obs.coverage(), obs.point_counts(),
                               0.0, 0, cfg.rho0, cfg.lambda_e)]
    executed = 0
    # 提前终止的原因必须记下来。否则候选池空、硬约束全灭、预算走完这三种情况
    # 在 final 里长得和正常收敛一模一样, 聚合表里只表现为"一个安静的低分"。
    stop_reason = "rounds"
    for rnd in range(cfg.rounds_max):
        if executed >= cfg.stations_max:
            stop_reason = "stations"
            break
        if path_len >= cfg.length_max_m:
            stop_reason = "length"
            break
        res = _select_next_station(world, obs, cfg, v0_xy, rng,
                                   budget_left=cfg.length_max_m - path_len,
                                   plan_oracle=plan_oracle, failed_xy=failed_xy)
        if res is None:
            stop_reason = "no_candidate"
            break
        origin, d = res
        scan = obs.add_station(origin, f"{cfg.method}_r{rnd}", seed=cfg.seed * 100 + 50 + rnd)
        if cfg.registration_realism:
            ov, n_ovl, dg = obs.registration_of_last()
            ok_reg, err = register_station(ov, n_ovl, dg, world.sensor.sigma_r, rng,
                                           o_min=V2Config().o_min, tol_m=cfg.reg_tol_m)
            reg_log.append({"round": rnd, "overlap": ov, "n_overlap": int(n_ovl),
                            "degen": dg, "pose_err_m": err, "accepted": bool(ok_reg)})
            if not ok_reg:
                obs.drop_last_station()   # 拼不进全局系, 数据作废; 路程与时间照付
                failed_xy.append(np.asarray(origin[:2], dtype=float))
                scan = None
        if scan is not None and disc is not None and disc.update(scan.points) > 0:
            plan_oracle = disc.build_oracle()   # 只在真有新发现时重建 BVH
        path_len += d
        executed += 1
        v0_xy = tuple(origin[:2])
        history.append(episode_metrics(world, gt, obs.coverage(), obs.point_counts(),
                                       path_len, executed, cfg.rho0, cfg.lambda_e))
    final = dict(history[-1])
    final["stop_reason"] = stop_reason
    if cfg.registration_realism:
        final["n_reg_failed"] = int(sum(not r["accepted"] for r in reg_log))
    return {"method": cfg.method, "seed": cfg.seed, "gt": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                           for k, v in gt.items()},
            "history": history, "final": final, "registration": reg_log, "status": "ok"}


def default_init_stations(world: SimWorld, n: int = 4,
                          spacing: float = 18.0) -> list[np.ndarray]:
    """S_init: 沿主干道路等距撒站, 故意不进入间隔内部(1.2.4)。"""
    xmin, _, xmax, _ = world.scene.bounds_xy
    y = world.scene.road_y
    xs = np.linspace(xmin + 5, xmax - 5, n)
    out = []
    for x in xs:
        proj = world.grid.project_to_free((x, y), max_dist=6.0)
        if proj is not None:
            out.append(np.array([proj[0], proj[1], TRIPOD_Z]))
    return out
