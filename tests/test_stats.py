"""PR9 验收: 统计模块行为。"""

import numpy as np

from patent_gap.evaluation.stats import bootstrap_ci, cliffs_delta, holm, paired_permutation


def test_paired_permutation_detects_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.5, 40)
    y = x - 0.8
    assert paired_permutation(x, y, n=2000, seed=1) < 0.01
    z = x + rng.normal(0, 0.01, 40)
    assert paired_permutation(x, z, n=2000, seed=1) > 0.05


def test_holm_monotone_and_bounded():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = holm(p)
    assert (adj >= p - 1e-12).all() and (adj <= 1.0).all()
    assert abs(adj[0] - 0.04) < 1e-12  # 4×0.01
    assert np.isnan(holm([0.01, np.nan])[1])


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(5.0, 1.0, 200)
    lo, hi = bootstrap_ci(x, n=2000, seed=3)
    assert lo < x.mean() < hi
    assert hi - lo < 1.0


def test_cliffs_delta_signs():
    assert cliffs_delta([2, 3, 4], [0, 1]) == 1.0
    assert cliffs_delta([0, 1], [2, 3, 4]) == -1.0
    assert abs(cliffs_delta([1, 2, 3], [1, 2, 3])) < 1e-12
