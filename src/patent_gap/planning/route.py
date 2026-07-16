"""公式(45) 第2段: 站集定序 — A* 距离矩阵上的开环 TSP(OR-Tools 路由求解器)。"""

from __future__ import annotations

import numpy as np


def solve_tsp(dist_matrix: np.ndarray, start: int = 0, time_limit_s: int = 10) -> list[int]:
    """开环 TSP: 从 start 出发访问所有节点一次, 不回起点。返回节点访问顺序。"""
    dist_matrix = np.asarray(dist_matrix, dtype=np.float64)
    n = len(dist_matrix)
    if n <= 1:
        return list(range(n))
    if n == 2:
        return [start, 1 - start] if start in (0, 1) else [start]
    if not np.isfinite(dist_matrix).all():
        # 不可达对以大数惩罚(上游应已剔除不可达候选)
        finite = dist_matrix[np.isfinite(dist_matrix)]
        big = (finite.max() if len(finite) else 1.0) * 100
        dist_matrix = np.where(np.isfinite(dist_matrix), dist_matrix, big)

    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    # 开环技巧: 增加虚拟终点, 到所有点距离 0
    D = np.zeros((n + 1, n + 1))
    D[:n, :n] = dist_matrix
    mgr = pywrapcp.RoutingIndexManager(n + 1, 1, [start], [n])
    rt = pywrapcp.RoutingModel(mgr)

    def cb(i, j):
        return int(D[mgr.IndexToNode(i)][mgr.IndexToNode(j)] * 100)

    idx = rt.RegisterTransitCallback(cb)
    rt.SetArcCostEvaluatorOfAllVehicles(idx)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = int(time_limit_s)
    sol = rt.SolveWithParameters(p)
    if sol is None:
        return list(range(n))  # 兜底: 原顺序
    route = []
    node = rt.Start(0)
    while not rt.IsEnd(node):
        k = mgr.IndexToNode(node)
        if k < n:
            route.append(k)
        node = sol.Value(rt.NextVar(node))
    return route


def route_length(dist_matrix: np.ndarray, route: list[int]) -> float:
    return float(sum(dist_matrix[route[k - 1], route[k]] for k in range(1, len(route))))
