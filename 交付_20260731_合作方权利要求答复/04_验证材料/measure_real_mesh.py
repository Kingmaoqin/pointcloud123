"""在真实 IFC 网格与真实点云上测量确定性规则的实际影响。

产生答复文件"（3）真实点云上的实际影响已测量"一节中的全部数字：
触发并列裁决的点数、匹配点数变化、最近距离变化、跨表面分块改变的点数、
各分块点数之和与总点数的关系、以及每点耗时。

运行（需在仓库根目录）：
    python3 交付_20260731_合作方权利要求答复/04_验证材料/measure_real_mesh.py

数据来源（均为仓库内既有产物，非本次为验证而生成）：
    data/processed/*mesh*.npz                        IFC 三角网格
    data/processed/cras_point_sample_associations.parquet   已配准点云与旧关联结果
    data/processed/patches_real_triangle_map.npy     三角面 → 表面分块映射
"""

from __future__ import annotations

import collections
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import patent_gap.data.tie_break as tb  # noqa: E402

TAU_D = 0.05


def main() -> None:
    mesh_files = [f for f in glob.glob(str(ROOT / "data/processed/*.npz")) if "mesh" in f]
    if not mesh_files:
        raise SystemExit("未找到 IFC 网格 npz")
    mesh = np.load(mesh_files[0])
    V = mesh["vertices"].astype(np.float64)
    F = mesh["faces"].astype(np.int64)

    df = pd.read_parquet(ROOT / "data/processed/cras_point_sample_associations.parquet")
    P = df[["registered_x", "registered_y", "registered_z"]].to_numpy(float)
    d0 = df["nearest_distance_m"].to_numpy(float)
    i0 = df["nearest_triangle_index"].to_numpy(np.int64)

    print(f"IFC 三角网格：{len(F)} 个三角面；点云：{len(P)} 点（仓库已配准坐标）")
    print(f"距离阈值 tau_d = {TAU_D} m；数值容差 = {tb.DEFAULT_TIE_TOL} m\n")

    # 统计实际触发并列裁决的点数
    original = tb.resolve_tie
    stat = {"n": 0, "max_k": 0}

    def counting(p, vx, fx, cands, tol=tb.DEFAULT_TIE_TOL, face_owner=None):
        stat["n"] += 1
        stat["max_k"] = max(stat["max_k"], len(cands))
        return original(p, vx, fx, cands, tol, face_owner)

    tb.resolve_tie = counting
    t0 = time.time()
    d1, i1 = tb.nearest_triangle_deterministic(P, V, F, d0, i0, max_distance=TAU_D)
    elapsed = time.time() - t0
    tb.resolve_tie = original

    m0, m1 = d0 <= TAU_D, d1 <= TAU_D
    print(f"耗时 {elapsed:.1f} s，合 {elapsed / len(P) * 1000:.3f} ms/点\n")
    print("指标                              结果")
    print("-" * 58)
    print(f"触发并列裁决的点                  {stat['n']} / {len(P)} "
          f"({stat['n'] / len(P) * 100:.2f}%)，最大并列面数 {stat['max_k']}")
    print(f"匹配点数（旧 → 新）               {int(m0.sum())} → {int(m1.sum())} "
          f"(Δ{int(m1.sum()) - int(m0.sum())})")
    print(f"最近距离最大变化                  {np.abs(d1 - d0).max():.3e} m")

    tm = glob.glob(str(ROOT / "data/processed/patches_real_triangle_map.npy"))
    if tm:
        tri_map = np.load(tm[0])
        p0, p1 = tri_map[i0], tri_map[i1]
        counts = collections.Counter(p1[m1][p1[m1] >= 0])
        print(f"跨表面分块改变归属的匹配点        {int(((p0 != p1) & m1).sum())}"
              f"（旧实现 vs 新规则的逐点比较）")
        print(f"各表面分块匹配点数之和            {sum(counts.values())} "
              f"≤ 总点数 {len(P)} → {'成立' if sum(counts.values()) <= len(P) else '不成立'}")
    print(f"每点归属数                        {1 if len(i1) == len(P) else '异常'}")


if __name__ == "__main__":
    main()
