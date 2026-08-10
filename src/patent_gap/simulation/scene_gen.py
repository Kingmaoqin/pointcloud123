"""1.2.3 变电站程序化场景生成器(参数化组件库 + 布局规则)。

尺寸为 110 kV 户外站典型量级(【推断】), 生成"物理一致"的参数化合成变电站,
非任何真实站复刻。每构件独立 comp_id, 携带语义类别与工程重要度 E_i。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import trimesh

# 类别 → (E_i, 是否带电, 对地安全距离 d_safe[m])
# 110kV 安全距离 1.5 m【待核实, GB 26860 类规范】, per-class 可配置
CLASS_TABLE: dict[str, tuple[float, bool, float]] = {
    "transformer": (1.0, True, 1.5),
    "breaker": (0.9, True, 1.5),
    "disconnector": (0.8, True, 1.5),
    "ct_pt": (0.8, True, 1.5),
    "arrester": (0.6, True, 1.5),
    "insulator": (0.7, False, 0.0),
    "busbar": (0.7, True, 0.0),      # 高空带电, 地面净空由 clearance_z 控制
    "gantry": (0.4, False, 0.0),
    "fence": (0.2, False, 0.0),
    "building": (0.3, False, 0.0),
    "clutter": (0.1, False, 0.0),
    "ground": (0.1, False, 0.0),
    # 竣工态特有的临时占位物(车辆/料堆/脚手架): 真实站里存在, 设计 BIM 里没有。
    # 规划器看不到它们, 扫描仪却会被它们挡住 —— 见 SceneModel.bim_tri_mask。
    "temp_obstacle": (0.1, False, 0.0),
}

# 非目标环境构件：它们会遮挡视线、影响通行，但不是本次验收要检查的资产。
# 目标参考模型 M_ref 不必描述它们；规划环境模型 M_plan 对它们的认识应当来自
# 实际观测。E7 通过控制其中多大比例进入 M_plan⁽⁰⁾ 来调节环境几何先验强度。
ENVIRONMENT_CLASSES = frozenset({"building", "fence", "clutter", "ground",
                                 "temp_obstacle"})

SIZE_TABLE = {"S": (44.0, 32.0, 2), "M": (58.0, 42.0, 3), "L": (74.0, 52.0, 4)}
DENSITY_TABLE = {"low": 1.0, "mid": 1.5, "high": 2.0}


@dataclass
class Component:
    comp_id: int
    cls: str
    importance: float
    live: bool
    d_safe: float
    clearance_z: float          # 底面离地净空(管母>车高 → 可从下穿行)
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    tri_start: int
    tri_end: int                # [start, end) 三角面全局索引
    in_bim: bool = True         # False = 只存在于竣工实景, 设计 BIM 中查不到
    is_target: bool = True      # True = 目标构件(M_ref 必含); False = 非目标环境物体


@dataclass
class SceneModel:
    vertices: np.ndarray                 # (V,3) float64
    triangles: np.ndarray                # (T,3) int64
    tri_to_component: np.ndarray         # (T,) int64, -1=地面
    components: list[Component] = field(default_factory=list)
    bounds_xy: tuple[float, float, float, float] = (0, 0, 1, 1)
    road_y: float = 0.0                  # 主干道路中心线
    seed: int = 0
    family: str = "S"
    density: str = "mid"

    def prior_tri_mask(self, env_prior_frac: float = 1.0,
                       seed: int = 0) -> np.ndarray:
        """(T,) bool: 规划环境模型 M_plan⁽⁰⁾ 初始包含哪些三角面。

        目标构件恒含 —— 它们是待扫目标，S4 的首次命中判定必须能命中它们本身；
        非目标环境构件按 env_prior_frac **以对象为单位**随机选取（不是随机删
        三角面，避免一个实体只剩半张皮、产生不物理的穿透）。

        env_prior_frac = 1.0 即现行行为（完整环境几何先验，E7 的 P100 档）；
        = 0.0 则 M_plan⁽⁰⁾ 只含目标表面，环境几何全部有待观测发现。
        """
        mask = np.zeros(len(self.triangles), dtype=bool)
        # 地面恒含: 它是移动平台的支承面, 按定义已知, 不属于"有待观测发现的
        # 环境几何"。若排除它, env_prior_frac=1.0 时 M_plan 也会与完整模型不同,
        # 既有结果即不再逐比特可复现。
        envs = [c for c in self.components
                if (not c.is_target) and c.cls != "ground" and c.in_bim]
        rng = np.random.default_rng(seed * 7919 + 13)
        keep = set()
        if envs and env_prior_frac > 0:
            k = int(round(env_prior_frac * len(envs)))
            if k > 0:
                keep = {envs[i].comp_id
                        for i in rng.choice(len(envs), k, replace=False)}
        for c in self.components:
            if not c.in_bim:
                continue
            if c.cls == "ground" or c.is_target or c.comp_id in keep:
                mask[c.tri_start:c.tri_end] = True
        return mask

    def bim_tri_mask(self) -> np.ndarray:
        """(T,) bool: 该三角面在设计 BIM 中是否存在。

        规划阶段只有 BIM 可用, 临时占位物查不到; 评测/仿真用完整竣工几何。
        两者不一致正是闭环重扫(45)要解决的问题 —— 若规划器直接用竣工几何,
        可见性预测与评测真值同源, 遮挡感知就无法被证伪。
        """
        mask = np.ones(len(self.triangles), dtype=bool)
        for c in self.components:
            if not c.in_bim:
                mask[c.tri_start:c.tri_end] = False
        return mask

    @property
    def l_diag(self) -> float:
        dx = self.bounds_xy[2] - self.bounds_xy[0]
        dy = self.bounds_xy[3] - self.bounds_xy[1]
        return float(np.hypot(dx, dy))


class _Builder:
    def __init__(self):
        self.verts: list[np.ndarray] = []
        self.faces: list[np.ndarray] = []
        self.tri_comp: list[np.ndarray] = []
        self.components: list[Component] = []
        self._nv = 0
        self._nt = 0

    def add(self, mesh: trimesh.Trimesh, cls: str, clearance_z: float = 0.0,
            in_bim: bool = True) -> None:
        e_i, live, d_safe = CLASS_TABLE[cls]
        comp_id = len(self.components)
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64) + self._nv
        cid = comp_id if cls != "ground" else -1
        self.verts.append(v)
        self.faces.append(f)
        self.tri_comp.append(np.full(len(f), cid, dtype=np.int64))
        if cls != "ground":
            self.components.append(Component(
                comp_id=comp_id, cls=cls, importance=e_i, live=live, d_safe=d_safe,
                clearance_z=clearance_z,
                bbox_min=v.min(axis=0), bbox_max=v.max(axis=0),
                tri_start=self._nt, tri_end=self._nt + len(f), in_bim=in_bim,
                is_target=cls not in ENVIRONMENT_CLASSES))
        else:
            self.components.append(Component(
                comp_id=comp_id, cls=cls, importance=e_i, live=False, d_safe=0.0,
                clearance_z=0.0, bbox_min=v.min(axis=0), bbox_max=v.max(axis=0),
                tri_start=self._nt, tri_end=self._nt + len(f), is_target=False))
        self._nv += len(v)
        self._nt += len(f)


def _box(extents, center) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    return m


def _cyl(radius, height, center_xy, base_z=0.0, sections=10) -> trimesh.Trimesh:
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    m.apply_translation([center_xy[0], center_xy[1], base_z + height / 2.0])
    return m


def _transformer(b: _Builder, cx, cy, rng) -> None:
    L, W, H = 8.0, 4.0, 5.0
    b.add(_box([L, W, H], [cx, cy, H / 2]), "transformer")
    # 散热片
    n_fin = int(rng.integers(4, 7))
    for k in range(n_fin):
        fx = cx - L / 2 + (k + 0.5) * L / n_fin
        b.add(_box([L / n_fin * 0.6, 0.6, 3.0], [fx, cy + W / 2 + 0.35, 2.0]), "transformer")
    # 防火墙(强遮挡体)
    b.add(_box([0.4, W + 3.0, H + 1.0], [cx - L / 2 - 1.2, cy, (H + 1) / 2]), "building")


def _three_phase_columns(b: _Builder, cls, cx, cy, h, r, phase_gap, along="x") -> None:
    for k in (-1, 0, 1):
        x = cx + k * phase_gap if along == "x" else cx
        y = cy if along == "x" else cy + k * phase_gap
        b.add(_cyl(r, h, (x, y)), cls)


def _bay(b: _Builder, cx, cy, rng) -> None:
    """一个间隔: 断路器 + 隔离开关×2 + CT/PT, 沿 y 方向排布。"""
    phase_gap = float(rng.uniform(1.5, 2.5))
    h_brk = float(rng.uniform(3.0, 5.0))
    _three_phase_columns(b, "breaker", cx, cy, h_brk, 0.18, phase_gap)
    _three_phase_columns(b, "disconnector", cx, cy + 3.0, float(rng.uniform(2.5, 3.5)), 0.12, phase_gap)
    _three_phase_columns(b, "disconnector", cx, cy - 3.0, float(rng.uniform(2.5, 3.5)), 0.12, phase_gap)
    _three_phase_columns(b, "ct_pt", cx, cy + 6.0, float(rng.uniform(2.0, 4.0)), 0.15, phase_gap)
    _three_phase_columns(b, "arrester", cx, cy - 6.0, float(rng.uniform(1.5, 3.0)), 0.10, phase_gap)


def _gantry_with_busbar(b: _Builder, x0, x1, y, rng) -> None:
    h = float(rng.uniform(8.0, 11.0))
    for x in (x0, x1):
        b.add(_box([0.5, 0.5, h], [x, y, h / 2]), "gantry")
    b.add(_box([x1 - x0, 0.4, 0.4], [(x0 + x1) / 2, y, h - 0.2]), "gantry")
    # 管型母线: 高空, 地面净空 = 离地高 → 车辆可从下穿行
    z_bus = float(rng.uniform(5.0, 7.0))
    m = trimesh.creation.cylinder(radius=float(rng.uniform(0.05, 0.1)),
                                  height=x1 - x0, sections=8)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    m.apply_translation([(x0 + x1) / 2, y, z_bus])
    b.add(m, "busbar", clearance_z=z_bus - 0.1)
    # 悬挂绝缘子串(细长目标)
    for x in (x0 + (x1 - x0) * 0.3, x0 + (x1 - x0) * 0.7):
        b.add(_cyl(0.12, 1.2, (x, y), base_z=z_bus - 1.3, sections=8), "insulator")


def generate_scene(seed: int, family: str = "S", density: str = "mid",
                   n_temp: int = 0) -> SceneModel:
    """n_temp: 竣工态临时占位物数量(BIM 中不存在)。

    默认 0 → 竣工几何与 BIM 完全一致, 与既有 E2 基准逐比特兼容。E3 扫描该参数,
    量化"BIM 与实景偏离"对遮挡感知规划的影响。
    """
    # 注意：不能用内置 hash()——Python 对字符串的哈希每进程随机化(PYTHONHASHSEED)，
    # 会导致同一 (seed, family, density) 在不同进程生成完全不同的场景，实验不可复现。
    # 用 sha1 取稳定摘要（与 occlusion/sampling.py 的 _stable_seed 同一做法）。
    tag = int(hashlib.sha1(f"{family}_{density}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed * 1000 + tag % 997)
    W, D, n_bay_base = SIZE_TABLE[family]
    n_bay = max(1, int(round(n_bay_base * DENSITY_TABLE[density])))
    b = _Builder()

    # 1. 地面
    b.add(_box([W, D, 0.1], [0, 0, -0.05]), "ground")
    road_y = 0.0

    # 2. 间隔沿道路北侧
    # 间隔间距必须与场地宽度相容: 间隔排总长 (n_bay−1)·pitch 加两侧各 6 m 余量
    # 不得超过 W。原先只从 U(9,14) 抽而不做约束, 于是 L/high(n_bay=8, 上限
    # 8.9 m) 有 16/20 个种子、M/high(上限 9.2 m) 有 15/20 个种子把设备摆到围栏
    # 外, 最远 14 m —— 那里没有可通行格, 永远扫不到, 等于给 awc 压一个人为
    # 天花板。抽样次数不变, 故不影响 pitch 上限本就宽裕的场景(S/low、S/mid、
    # M/mid、M/low、L/low)的既有结果。
    # 用 min() 钳位会让上限低于 9 m 的场景退化成常数 —— 实测 L/high 20/20 个
    # 种子的 pitch 全部恒为 8.857 m, 间隔排 x 坐标逐位相同, 跨种子只剩相位与
    # 高度等次要随机量, E5 的跨场景变异系数会低估其真实方差。改为压缩抽样区间
    # (抽样次数不变, 不破坏 RNG 流, 上限宽裕的场景逐比特不受影响)。
    pitch_max = (W - 12.0) / max(n_bay - 1, 1)
    bay_pitch = float(rng.uniform(min(9.0, pitch_max), min(14.0, pitch_max)))
    x_start = -(n_bay - 1) * bay_pitch / 2
    # 间隔排在 y 方向的跨度是 bay_y ± 6 m(CT/PT 在 +6, 避雷器在 −6)。S 族的
    # D/2 恰为 16.0, 而 road_y+10+6 = 16.0 —— 20/20 个种子的 CT/PT 全部嵌在北
    # 围栏里并越界 0.143 m, 且其 E_i=0.8 计入关键设备召回、live=True 的 1.5 m
    # 禁入区把北侧通道封死。此前的 pitch 修复只管 x 方向。
    bay_y = min(road_y + 10.0, D / 2.0 - 8.0)
    for k in range(n_bay):
        _bay(b, x_start + k * bay_pitch, bay_y, rng)

    # 3. 主变 1–2 台, 道路南侧
    n_tr = int(rng.integers(1, 3))
    for k in range(n_tr):
        _transformer(b, -W / 4 + k * (W / 2), road_y - 9.0, rng)

    # 4. 门型构架 + 管母跨越
    _gantry_with_busbar(b, x_start - 3.0, x_start + (n_bay - 1) * bay_pitch + 3.0,
                        bay_y + 3.0, rng)
    if density == "high":
        _gantry_with_busbar(b, x_start - 3.0, x_start + (n_bay - 1) * bay_pitch + 3.0,
                            bay_y - 4.5, rng)

    # 5. 围栏 + 控制楼 + 杂物
    t = 0.15
    hf = 2.2
    b.add(_box([W, t, hf], [0, D / 2 - t, hf / 2]), "fence")
    b.add(_box([W, t, hf], [0, -D / 2 + t, hf / 2]), "fence")
    b.add(_box([t, D, hf], [W / 2 - t, 0, hf / 2]), "fence")
    b.add(_box([t, D, hf], [-W / 2 + t, 0, hf / 2]), "fence")
    b.add(_box([10, 6, 4], [W / 2 - 7.0, -D / 2 + 5.0, 2.0]), "building")
    for _ in range(int(rng.integers(0, 4))):
        cx = float(rng.uniform(-W / 2 + 4, W / 2 - 4))
        cy = float(rng.uniform(road_y - 4.5, road_y - 2.0))
        b.add(_box([float(rng.uniform(0.8, 2.0)), float(rng.uniform(0.8, 2.0)),
                    float(rng.uniform(0.8, 1.8))], [cx, cy, 0.6]), "clutter")

    # 6. 竣工态临时占位物(车辆/料堆/脚手架), BIM 中不存在。
    #    刻意放在道路与设备之间: 站点都架在道路附近, 只有挡在视线上才构成
    #    真正的预测误差, 撒在空地上等于什么都没测。
    for _ in range(n_temp):
        north = bool(rng.integers(0, 2))
        cx = float(rng.uniform(x_start - 2.0,
                               x_start + (n_bay - 1) * bay_pitch + 2.0))
        cy = (road_y + float(rng.uniform(2.5, 7.0)) if north
              else road_y - float(rng.uniform(2.5, 6.5)))
        hz = float(rng.uniform(1.6, 3.0))
        b.add(_box([float(rng.uniform(1.6, 4.0)), float(rng.uniform(1.4, 2.6)), hz],
                   [cx, cy, hz / 2]), "temp_obstacle", in_bim=False)

    vertices = np.vstack(b.verts)
    triangles = np.vstack(b.faces)
    tri_comp = np.concatenate(b.tri_comp)
    return SceneModel(
        vertices=vertices, triangles=triangles, tri_to_component=tri_comp,
        components=b.components,
        bounds_xy=(-W / 2, -D / 2, W / 2, D / 2),
        road_y=road_y, seed=seed, family=family, density=density,
    )


# 可穿行的构件类别: 它们在 IFC 里是实体, 但对作业人员/移动平台而言是通路。
# 户外变电站没有"房间", 所以流水线里从来没有这个概念 —— 而真实建筑里 IfcDoor
# 是门扇(z 0.10–2.33 m), 按障碍处理会把每个房间封死: CRAS 实测自由空间被切成
# 9–10 个互不连通的域, 最大的只占 28%, 从任一初始站 A* 到别处都不可达。
PASSABLE_CLASSES = frozenset({"IfcDoor", "IfcOpeningElement", "IfcWindow"})


def build_trav_grid(scene: SceneModel, res: float = 0.25,
                    r_robot: float = 0.6, h_robot: float = 1.8,
                    z_step: float = 0.20,
                    passable_cls: frozenset[str] = PASSABLE_CLASSES,
                    by_triangle: bool = False,
                    bim_only: bool = False):
    """公式(43): 由场景构件构建 2.5D 可通行图。

    z_step: 顶面低于该高度的构件按"可跨越/可站立的地面"处理, 不计为障碍。
    原先只按类别硬编码跳过 "ground", 这对合成变电站够用, 对真实 IFC 不行 ——
    CRAS 的 IfcSlab 就是楼板(z 顶面 0.10 m), 按障碍处理会把整个建筑底面填满,
    实测自由格为 0/7154, 参考站集一站都建不起来、缺口数恒为 0。用物理高度判据
    替代类别判据, 两种场景都成立(变电站的 ground 顶面为 0.0, 同样被跳过)。

    bim_only: 只用设计模型中存在的构件建图。**规划器用的图必须置 True** ——
    否则临时占位物(in_bim=False)会一并成为障碍, 规划器于是绕开了一批"设计
    模型里根本查不到"的东西, 那是它不可能知道的信息。评测真值与初始站用的图
    仍应置 False: 参考站集要physically可站, 测量员确实站不进一辆车里。
    n_temp=0 时两者逐比特相同。
    """
    from ..mapping.traversability import TravGrid

    grid = TravGrid(scene.bounds_xy, res=res, r_robot=r_robot)
    for c in scene.components:
        if bim_only and not c.in_bim:
            continue
        if (c.cls == "ground" or c.cls in passable_cls
                or float(c.bbox_max[2]) <= z_step):
            continue
        if by_triangle:
            _rasterize_component(grid, scene, c, z_step, h_robot)
        else:
            grid.add_obstacle_box(c.bbox_min[:2], c.bbox_max[:2],
                                  clearance_z=c.clearance_z, h_robot=h_robot)
        if c.live and c.clearance_z < h_robot:
            grid.add_safety_zone(c.bbox_min[:2], c.bbox_max[:2], c.d_safe)
    return grid


def _rasterize_component(grid, scene: SceneModel, c: Component,
                         z_step: float, h_robot: float) -> None:
    """按构件三角面在机器人身高带内的真实投影标记障碍。

    包围盒对变电站的方箱设备够用, 对真实建筑不行: 一个构件可能是 L 形墙、带门洞
    的墙、或跨越整栋楼的楼板边缘, 其包围盒会连同门洞与通道一起封死。逐三角面
    投影只封住确实有实体的格子。
    """
    tri = scene.triangles[c.tri_start:c.tri_end]
    if not len(tri):
        return
    v = scene.vertices[tri]                     # (T,3,3)
    zmin, zmax = v[:, :, 2].min(axis=1), v[:, :, 2].max(axis=1)
    keep = (zmax > z_step) & (zmin < h_robot)   # 只有挡在身高带里的才是障碍
    if not keep.any():
        return
    v = v[keep][:, :, :2]
    lo = np.array([grid.xmin, grid.ymin])
    ij_lo = np.floor((v.min(axis=1) - lo) / grid.res).astype(int)
    ij_hi = np.ceil((v.max(axis=1) - lo) / grid.res).astype(int)
    for (i0, j0), (i1, j1) in zip(ij_lo, ij_hi):
        i0 = max(i0, 0); j0 = max(j0, 0)
        i1 = min(i1 + 1, grid.nx); j1 = min(j1 + 1, grid.ny)
        if i1 > i0 and j1 > j0:
            grid._obstacle[i0:i1, j0:j1] = True
