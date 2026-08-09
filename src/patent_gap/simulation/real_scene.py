"""把真实 IFC 数据接入升级后的闭环(closed_loop_v2)。

此前升级只在合成变电站场景上验证过（OPEN_ISSUES #11）。本模块用 CRAS 数据集的
真实 IFC 模型构建 SimWorld，使公式(26)–(45) 第一次在真实建筑几何上运行：

- **几何是真的**：604187 顶点 / 1197750 三角面 / 256 个 IFC 构件，真实的墙-柱-
  门窗-家具排布，遮挡结构不是程序化生成的。
- **缺口是真的**：584701977 个实测点按 5 cm 阈值关联到 IFC 构件（12835294 个
  匹配），据此给出各分块的真实初始覆盖，而不是拿仿真站扫出来的。
- **重扫是仿真的**：数据集只提供一份融合后的 ASC，没有逐站位姿（见
  docs/data_audit.md），因此无法回放原始站位。补扫站的重新采集由射线仿真器在
  真实 IFC 网格上完成。这一点必须在任何引用本实验的地方写明。

另需说明：CRAS 是实验楼，不是变电站。它检验的是"这套流水线能不能吃真实 IFC 与
真实点云"，不能替代变电站场景下的效果验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .scene_gen import Component, SceneModel

# IFC 类别 → 工程重要度 E_i。合成场景的 CLASS_TABLE 是变电站一次设备口径，
# 对实验楼不适用。这里按"竣工验收关心程度"给一份建筑口径的映射：承重与围护
# 结构最高，门窗次之，家具最低。数值为【推断】，不是规范值。
IFC_IMPORTANCE: dict[str, float] = {
    "IfcColumn": 1.0,
    "IfcSlab": 0.9,
    "IfcWallStandardCase": 0.9,
    "IfcWall": 0.9,
    "IfcStairFlight": 0.8,
    "IfcBeam": 0.8,
    "IfcDoor": 0.6,
    "IfcWindow": 0.6,
    "IfcCovering": 0.4,
    "IfcOpeningElement": 0.3,
    "IfcBuildingElementProxy": 0.3,
    "IfcFurnishingElement": 0.2,
}
IMPORTANCE_DEFAULT = 0.3


def load_real_scene(processed: Path, importance: dict[str, float] | None = None
                    ) -> tuple[SceneModel, pd.DataFrame]:
    """由 data/processed 下的 IFC 缓存构建 SceneModel。

    返回 (scene, elements)。scene.components 与 IFC 构件一一对应，comp_id 即
    element_index，从而与 patches_real / element_counts 可直接对齐。
    """
    imp_table = importance or IFC_IMPORTANCE
    mesh = np.load(processed / "ifc_mesh.npz")
    verts = np.asarray(mesh["vertices"], dtype=np.float64)
    faces = np.asarray(mesh["faces"], dtype=np.int64)
    guid = np.load(processed / "triangle_element_guid.npy", allow_pickle=True)
    elements = pd.read_parquet(processed / "ifc_elements.parquet")

    guid_to_idx = dict(zip(elements["GlobalId"], elements["element_index"]))
    tri_elem = np.array([guid_to_idx.get(g, -1) for g in guid], dtype=np.int64)

    # 三角面按构件重排，使每个构件占一段连续区间（Component 用 [start, end) 表达）
    order = np.argsort(tri_elem, kind="stable")
    faces = faces[order]
    tri_elem = tri_elem[order]

    comps: list[Component] = []
    for eidx in np.unique(tri_elem):
        if eidx < 0:
            continue
        sel = np.where(tri_elem == eidx)[0]
        v = verts[faces[sel].ravel()]
        row = elements.loc[elements["element_index"] == eidx].iloc[0]
        cls = str(row["ifc_class"])
        comps.append(Component(
            comp_id=int(eidx), cls=cls,
            importance=float(imp_table.get(cls, IMPORTANCE_DEFAULT)),
            live=False, d_safe=0.0,           # 实验楼无带电体
            clearance_z=float(v[:, 2].min()),
            bbox_min=v.min(axis=0), bbox_max=v.max(axis=0),
            tri_start=int(sel[0]), tri_end=int(sel[-1]) + 1, in_bim=True))

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    scene = SceneModel(
        vertices=verts, triangles=faces, tri_to_component=tri_elem,
        components=comps,
        bounds_xy=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
        road_y=float((lo[1] + hi[1]) / 2.0),
        seed=0, family="CRAS", density="real")
    return scene, elements


def real_initial_coverage(processed: Path, patches: pd.DataFrame,
                          tri_to_patch: np.ndarray, scene: SceneModel,
                          rho_floor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """由**实测点云**给出各分块的初始覆盖率与点数。

    数据集只给到构件级的匹配点数（element_counts.csv），没有点到分块的归属，
    因此构件内按面积分摊到其各分块——这是一处近似，会低估构件内部的覆盖不均，
    须在结论中声明。

    返回 (C_init, n_pts)。C_init 由分块实测点密度相对 rho_floor 截断得到。
    """
    counts = pd.read_csv(processed / "cras_full_assoc" / "element_counts.csv")
    per_elem = dict(zip(counts["element_index"], counts["matched_points"]))

    comp_of_patch = {}
    for c in scene.components:
        for pid in np.unique(tri_to_patch[c.tri_start:c.tri_end]):
            if pid >= 0:
                comp_of_patch.setdefault(int(pid), c.comp_id)

    area = patches["area"].to_numpy(dtype=np.float64)
    pid_arr = patches["patch_id"].to_numpy()
    area_by_comp: dict[int, float] = {}
    for k, pid in enumerate(pid_arr):
        cid = comp_of_patch.get(int(pid))
        if cid is not None:
            area_by_comp[cid] = area_by_comp.get(cid, 0.0) + area[k]

    n_pts = np.zeros(len(pid_arr))
    for k, pid in enumerate(pid_arr):
        cid = comp_of_patch.get(int(pid))
        if cid is None:
            continue
        tot = area_by_comp.get(cid, 0.0)
        if tot > 0:
            n_pts[k] = per_elem.get(cid, 0) * area[k] / tot

    rho = n_pts / np.maximum(area, 1e-9)
    return np.clip(rho / max(rho_floor, 1e-9), 0.0, 1.0), n_pts


def real_scene_report(scene: SceneModel, patches: pd.DataFrame) -> dict:
    """给出一份可写进结论的规模与构成摘要。"""
    by_cls: dict[str, int] = {}
    for c in scene.components:
        by_cls[c.cls] = by_cls.get(c.cls, 0) + 1
    lo = np.array([scene.bounds_xy[0], scene.bounds_xy[1]])
    hi = np.array([scene.bounds_xy[2], scene.bounds_xy[3]])
    return {"vertices": int(len(scene.vertices)),
            "triangles": int(len(scene.triangles)),
            "components": len(scene.components),
            "patches": int(len(patches)),
            "extent_m": [round(float(hi[0] - lo[0]), 2), round(float(hi[1] - lo[1]), 2)],
            "total_area_m2": round(float(patches["area"].sum()), 1),
            "by_ifc_class": dict(sorted(by_cls.items(), key=lambda t: -t[1]))}


if __name__ == "__main__":  # 快速自检
    scene, elems = load_real_scene(Path("data/processed"))
    from .scene_patches import build_scene_patches
    patches, t2p = build_scene_patches(scene)
    print(json.dumps(real_scene_report(scene, patches), ensure_ascii=False, indent=1))
    C, n = real_initial_coverage(Path("data/processed"), patches, t2p, scene)
    print(f"实测初始覆盖: 均值 {C.mean():.3f}, 零覆盖分块 {int((C <= 0).sum())}/{len(C)}")
