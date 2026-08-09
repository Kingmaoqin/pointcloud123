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
                tri_start=self._nt, tri_end=self._nt + len(f), in_bim=in_bim))
        else:
            self.components.append(Component(
                comp_id=comp_id, cls=cls, importance=e_i, live=False, d_safe=0.0,
                clearance_z=0.0, bbox_min=v.min(axis=0), bbox_max=v.max(axis=0),
                tri_start=self._nt, tri_end=self._nt + len(f)))
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


def build_trav_grid(scene: SceneModel, res: float = 0.25,
                    r_robot: float = 0.6, h_robot: float = 1.8):
    """公式(43): 由场景构件构建 2.5D 可通行图。"""
    from ..mapping.traversability import TravGrid

    grid = TravGrid(scene.bounds_xy, res=res, r_robot=r_robot)
    for c in scene.components:
        if c.cls == "ground":
            continue
        grid.add_obstacle_box(c.bbox_min[:2], c.bbox_max[:2],
                              clearance_z=c.clearance_z, h_robot=h_robot)
        if c.live and c.clearance_z < h_robot:
            grid.add_safety_zone(c.bbox_min[:2], c.bbox_max[:2], c.d_safe)
    return grid
