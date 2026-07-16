"""3.8 统计推断: 配对置换检验 / Holm 校正 / bootstrap CI / Cliff's delta。"""

from __future__ import annotations

import numpy as np


def paired_permutation(x: np.ndarray, y: np.ndarray, n: int = 10000, seed: int = 0) -> float:
    """双侧配对置换检验 p 值。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    if len(d) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    t0 = d.mean()
    signs = rng.choice([1.0, -1.0], size=(n, len(d)))
    ts = (signs * d[None, :]).mean(axis=1)
    return float((np.abs(ts) >= abs(t0)).mean())


def holm(pvals) -> np.ndarray:
    """Holm 逐步校正。NaN 保留为 NaN。"""
    p = np.asarray(pvals, dtype=float)
    adj = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    m = len(valid)
    order = valid[np.argsort(p[valid])]
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def bootstrap_ci(x, stat=np.mean, n: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = stat(x[idx], axis=1) if stat in (np.mean, np.median) else \
        np.array([stat(x[i]) for i in idx])
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def cliffs_delta(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))
