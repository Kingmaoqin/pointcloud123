"""PR7 验收: 玩具例贪心对拍(≥0.63×穷举最优); TSP; 目标函数(44)。"""

from itertools import combinations

import numpy as np

from patent_gap.planning.objective import ObjectiveParams, trajectory_objective
from patent_gap.planning.route import route_length, solve_tsp
from patent_gap.planning.set_select import coverage_value, lazy_greedy


def test_lazy_greedy_vs_exhaustive_toy():
    """5 站 6 Patch 玩具例: 贪心 F ≥ 0.63 × 穷举最优。"""
    rng = np.random.default_rng(7)
    c_jv = rng.uniform(0, 1, size=(6, 5)) * (rng.random((6, 5)) > 0.3)
    gains = rng.uniform(0.5, 2.0, 6)
    cost = np.ones(5)
    budget = 3.0  # 最多3站
    best = 0.0
    for r in range(1, 4):
        for combo in combinations(range(5), r):
            best = max(best, coverage_value(c_jv, gains, list(combo)))
    sel = lazy_greedy(c_jv, gains, cost, budget)
    f_greedy = coverage_value(c_jv, gains, sel)
    assert f_greedy >= 0.63 * best - 1e-9
    assert sum(cost[v] for v in sel) <= budget + 1e-9


def test_lazy_greedy_budget_and_feasibility():
    c_jv = np.array([[0.9, 0.9, 0.1], [0.1, 0.8, 0.9]])
    gains = np.ones(2)
    cost = np.array([10.0, 1.0, 1.0])
    sel = lazy_greedy(c_jv, gains, cost, budget=2.0)
    assert 0 not in sel                       # 超预算
    feas = np.array([True, False, True])
    sel2 = lazy_greedy(c_jv, gains, cost, budget=2.0, feasible=feas)
    assert 1 not in sel2                      # 硬约束剔除


def test_tsp_open_loop_small():
    # 一条直线上的4点: 最优开环路线是顺序走
    xs = np.array([0.0, 10.0, 2.0, 6.0])
    D = np.abs(xs[:, None] - xs[None, :])
    route = solve_tsp(D, start=0, time_limit_s=2)
    assert route[0] == 0
    assert sorted(route) == [0, 1, 2, 3]
    assert abs(route_length(D, route) - 10.0) < 1e-6  # 0→2→6→10


def test_trajectory_objective_44():
    c_jv = np.array([[0.8, 0.0], [0.0, 0.9]])
    gains = np.array([1.0, 2.0])
    r_reg_v = np.array([0.7, 0.4])
    D = np.array([[0, 5, 8], [5, 0, 4], [8, 4, 0]], dtype=float)  # 0=v0
    route = [0, 1, 2]
    par = ObjectiveParams(m_max=8)
    out = trajectory_objective(c_jv, gains, r_reg_v, D, route, l_diag=50.0,
                               params=par)
    assert abs(out["F"] - (1.0 * 0.8 + 2.0 * 0.9)) < 1e-9
    assert abs(out["path_length_m"] - 9.0) < 1e-9
    assert out["n_stations"] == 2
    # 系数取自 params 而非字面量: 写死会在默认值调整时静默失配
    expect_j = ((2.6 / 3.0) + par.lambda_reg * 0.4 - par.lambda_len * 9.0 / 50.0
                - par.lambda_sta * 2 / 8)
    assert abs(out["J"] - expect_j) < 1e-9
    assert abs(out["time_s"] - (9.0 / 0.5 + 2 * 180.0)) < 1e-9
