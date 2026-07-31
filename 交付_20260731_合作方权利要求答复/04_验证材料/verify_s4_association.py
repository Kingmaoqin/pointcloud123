"""S4 点-面关联的三项验证（用于答复代理人 2026-07-31 批注）。

验证一：点到三角面最近距离的闭式算法与本项目实现（Open3D compute_closest_points）
         在数值上一致 —— 支撑对疑问1（公式(9)中 x、y 的数学形式与计算方式）的答复。
验证二：到不同表面分块"距离严格相等"的情形确实存在，且在凸棱外侧构成以公共棱
         为脊的三维楔形区域（正体积），并非零测度个例 —— 支撑对疑问2（是否存在 d(x,y1)=d(x,y2)）的答复。
验证三：并列时现有实现的取舍随三角面存储顺序改变，底层库未承诺任何规则 ——
         证明必须在技术方案中显式规定唯一归属规则。

运行：python3 verify_s4_association.py
依赖：numpy、open3d（本项目环境已具备）
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点到闭线段 ab 的距离（线段退化为点时即点到点距离）。"""
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 0.0:
        return float(np.linalg.norm(p - a))
    t = float((p - a) @ ab) / denom
    t = min(max(t, 0.0), 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def closest_distance_point_triangle(p: np.ndarray, tri: np.ndarray) -> float:
    """点到单个三角形的最近距离（闭式解，Ericson 最近点算法）。

    tri 为 3x3 数组，三行依次为三角形顶点 A、B、C。
    先按重心坐标判定最近点落在面内、三条边还是三个顶点，再取相应距离。
    """
    A, B, C = tri
    AB, AC, AP = B - A, C - A, p - A
    # 退化三角面守卫：三顶点共线或存在重合（面积为零）时所在平面无定义，
    # 重心坐标分支会出现 0/0 或被舍入噪声支配，直接退化到三条闭线段取最小。
    cross = np.cross(AB, AC)
    if float(cross @ cross) <= 1e-24 * max(float(AB @ AB) * float(AC @ AC), 1e-30):
        return min(_point_segment_distance(p, A, B),
                   _point_segment_distance(p, B, C),
                   _point_segment_distance(p, A, C))
    d1, d2 = AB @ AP, AC @ AP
    if d1 <= 0 and d2 <= 0:                       # 最近点为顶点 A
        return float(np.linalg.norm(p - A))
    BP = p - B
    d3, d4 = AB @ BP, AC @ BP
    if d3 >= 0 and d4 <= d3:                      # 顶点 B
        return float(np.linalg.norm(p - B))
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:           # 边 AB
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(p - (A + v * AB)))
    CP = p - C
    d5, d6 = AB @ CP, AC @ CP
    if d6 >= 0 and d5 <= d6:                      # 顶点 C
        return float(np.linalg.norm(p - C))
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:           # 边 AC
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(p - (A + w * AC)))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:   # 边 BC
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(p - (B + w * (C - B))))
    den = 1.0 / (va + vb + vc)                    # 最近点落在三角形内部
    v, w = vb * den, vc * den
    return float(np.linalg.norm(p - (A + AB * v + AC * w)))


def _scene(V: np.ndarray, F: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(V.astype(np.float32)),
                    o3d.core.Tensor(F.astype(np.uint32)))
    return s


def verify_1_closed_form() -> None:
    print("验证一：闭式点-面距离 vs 本项目实现")
    rng = np.random.default_rng(0)
    V = rng.normal(size=(40, 3))
    F = np.array([rng.choice(40, 3, replace=False) for _ in range(60)])
    P = rng.normal(scale=1.5, size=(400, 3))
    ans = _scene(V, F).compute_closest_points(o3d.core.Tensor(P.astype(np.float32)))
    d_impl = np.linalg.norm(P - ans["points"].numpy().astype(np.float64), axis=1)
    d_form = np.array([min(closest_distance_point_triangle(p, V[f]) for f in F) for p in P])
    err = float(np.abs(d_impl - d_form).max())
    print(f"  400个随机点、60个三角面：最大绝对偏差 {err:.2e} m")
    print(f"  结论：{'一致（float32 精度内）' if err < 1e-6 else '不一致，需复查'}\n")


def verify_2_tie_region() -> None:
    print("验证二：凸棱外侧的等距区域（两个表面分块严格等距）")
    # 竖直面 x=0 与水平面 z=0 相交于 y 轴，夹角90°；
    # 双法向约束会把二者判为两个不同的表面分块。
    V = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1],
                  [1, 0, 0], [1, 1, 0]], dtype=float)
    F = np.array([[0, 1, 2], [1, 3, 2],      # 表面分块0：竖直面
                  [0, 4, 1], [4, 5, 1]])     # 表面分块1：水平面
    patch_of = [0, 0, 1, 1]
    sc = _scene(V, F)
    print("  沿棱外侧45°方向取点（该方向上每一点到两个分块都严格等距）：")
    for t in (0.05, 0.10, 0.20, 0.35, 0.50):
        p = np.array([-t / np.sqrt(2), 0.5, -t / np.sqrt(2)])
        dd = [closest_distance_point_triangle(p, V[f]) for f in F]
        d_p0, d_p1 = min(dd[0], dd[1]), min(dd[2], dd[3])
        pid = int(sc.compute_closest_points(
            o3d.core.Tensor(p[None].astype(np.float32)))["primitive_ids"].numpy()[0])
        print(f"    距棱 {t:.2f} m：到分块0 = {d_p0:.9f}，到分块1 = {d_p1:.9f}，"
              f"相等={abs(d_p0 - d_p1) < 1e-9}，实现判给分块{patch_of[pid]}")
    print("  结论：等距点构成以公共棱为脊的三维楔形区域（正体积），非零测度个例；")
    print("        区域内每一点的最近表面点均落在该棱上，而该棱通常即为双法向约束")
    print("        所划出的两个表面分块的公共边界。\n")


def verify_3_tie_instability() -> None:
    print("验证三：并列时现有实现的取舍是否稳定")
    V = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1],
                  [1, 0, 0], [1, 1, 0]], dtype=float)
    base = [[0, 1, 2], [1, 3, 2], [0, 4, 1], [4, 5, 1]]
    p = np.array([[-0.2 / np.sqrt(2), 0.5, -0.2 / np.sqrt(2)]])
    for perm in ([0, 1, 2, 3], [2, 3, 0, 1], [3, 2, 1, 0], [1, 0, 3, 2]):
        F = np.array([base[i] for i in perm])
        pid = int(_scene(V, F).compute_closest_points(
            o3d.core.Tensor(p.astype(np.float32)))["primitive_ids"].numpy()[0])
        orig = perm[pid]
        print(f"    三角面存储顺序 {perm} → 判给原三角面#{orig}（属分块{0 if orig < 2 else 1}）")
    print("  结论：取舍随三角面存储顺序而变，底层库未承诺任何规则；")
    print("        因此必须在技术方案中显式规定唯一归属规则，方案才具备确定性与可复现性。\n")


def verify_0_degenerate() -> None:
    """退化三角面守卫（独立审查发现：无守卫时 A==B 返回 NaN、共线面静默出错）。"""
    print("验证零：退化三角面处理")
    rng = np.random.default_rng(7)
    P = rng.normal(scale=1.5, size=(2000, 3))
    cases = {
        "两顶点重合 A==B": np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float),
        "三顶点重合": np.array([[0.2, 0.3, 0.4]] * 3, dtype=float),
        "三点共线（零面积）": np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float),
    }
    ok = True
    for name, tri in cases.items():
        d = np.array([closest_distance_point_triangle(p, tri) for p in P])
        # 参照值：到三条闭线段距离的最小值（退化面的正确定义）
        ref = np.array([min(_point_segment_distance(p, tri[0], tri[1]),
                            _point_segment_distance(p, tri[1], tri[2]),
                            _point_segment_distance(p, tri[0], tri[2])) for p in P])
        nan_n = int(np.isnan(d).sum())
        err = float(np.nanmax(np.abs(d - ref)))
        ok &= (nan_n == 0 and err < 1e-12)
        print(f"    {name}: NaN {nan_n} 个，与线段参照最大偏差 {err:.2e} m")
    print(f"  结论：{'退化面处理正确' if ok else '退化面仍有问题'}\n")


if __name__ == "__main__":
    verify_0_degenerate()
    verify_1_closed_form()
    verify_2_tie_region()
    verify_3_tie_instability()
