"""公式(42) 覆盖概率子模集合函数 + 预算约束懒惰贪心(cost-benefit greedy)。

F(S) = Σ_j G_task_j·A_j·(1 − Π_{v∈S}(1 − c_{j,v}))
F 单调子模 → 懒惰贪心 (1−1/e) 近似(cost-benefit 变体)。
"""

from __future__ import annotations

import heapq

import numpy as np


def coverage_value(c_jv: np.ndarray, gains_j: np.ndarray, selected: list[int]) -> float:
    """F(S)。c_jv: (J,V) 覆盖成功概率 q'_{j,v}; gains_j = G_task_j·A_j。"""
    if not selected:
        return 0.0
    miss = np.prod(1.0 - c_jv[:, selected], axis=1)
    return float((gains_j * (1.0 - miss)).sum())


def lazy_greedy(c_jv: np.ndarray, gains_j: np.ndarray, cost_v: np.ndarray,
                budget: float, feasible: np.ndarray | None = None,
                max_stations: int | None = None) -> list[int]:
    """预算约束懒惰贪心选站。

    c_jv    (J,V): 覆盖成功概率 c_{j,v} = q'_{j,v} ∈ [0,1]
    gains_j (J,) : G_task_j·A_j
    cost_v  (V,) : 每站成本(等效距离/时间, 单一预算维度)
    budget       : Σcost ≤ budget
    feasible (V,) bool: 硬约束可行掩码(公式41 末段已剔除者为 False)
    """
    c_jv = np.asarray(c_jv, dtype=np.float64)
    gains_j = np.asarray(gains_j, dtype=np.float64)
    cost_v = np.asarray(cost_v, dtype=np.float64)
    J, V = c_jv.shape
    if feasible is None:
        feasible = np.ones(V, dtype=bool)
    # uncov_j = G_task_j·A_j·Π_{v∈S}(1−c_jv): 尚未覆盖的期望收益
    uncov = gains_j.copy()

    def marginal(v: int) -> float:
        return float((uncov * c_jv[:, v]).sum())

    heap = []
    for v in range(V):
        if feasible[v] and cost_v[v] > 0:
            heap.append((-marginal(v) / cost_v[v], 0, v))
        elif feasible[v]:
            heap.append((-marginal(v) / 1e-9, 0, v))
    heapq.heapify(heap)

    S: list[int] = []
    spent = 0.0
    while heap:
        if max_stations is not None and len(S) >= max_stations:
            break
        neg, stamp, v = heapq.heappop(heap)
        if v in S or spent + cost_v[v] > budget:
            continue
        if stamp < len(S):  # 懒惰重估
            heapq.heappush(heap, (-marginal(v) / max(cost_v[v], 1e-9), len(S), v))
            continue
        if -neg <= 1e-12:
            break
        S.append(v)
        spent += float(cost_v[v])
        uncov *= (1.0 - c_jv[:, v])
    return S
