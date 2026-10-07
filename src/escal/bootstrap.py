"""Paired day-block bootstrap. All models in one call are resampled with the same days."""
from __future__ import annotations

import numpy as np
import pandas as pd


def day_index(timestamps) -> np.ndarray:
    """Local standard-time day (08 UTC to 08 UTC) as an integer block id."""
    d = (pd.DatetimeIndex(timestamps) - pd.Timedelta(hours=8)).normalize()
    return pd.factorize(d)[0]


def day_sums(y: np.ndarray, p: np.ndarray, sel: np.ndarray, day: np.ndarray, n_days: int):
    """Per-day, per-horizon sum of squared errors and cell counts over the selected cells."""
    sse = np.zeros((n_days, y.shape[1]))
    cnt = np.zeros((n_days, y.shape[1]))
    for j in range(y.shape[1]):
        s = sel[:, j]
        np.add.at(sse[:, j], day[s], (y[s, j] - p[s, j]) ** 2)
        np.add.at(cnt[:, j], day[s], 1.0)
    return sse, cnt


def weights(n_days: int, cfg: dict) -> np.ndarray:
    """(B, n_days) multiplicities of each day in each resample."""
    b = cfg["bootstrap"]
    rng = np.random.default_rng(b["seed"])
    return rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=b["n_resamples"]).astype(float)


def avg_rmse(sse: np.ndarray, cnt: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """RMSE per horizon averaged over horizons; for each resample if w is given."""
    if w is None:
        return np.sqrt(sse.sum(0) / cnt.sum(0)).mean()
    return np.sqrt((w @ sse) / (w @ cnt)).mean(axis=1)


def interval(point: float, samples: np.ndarray, cfg: dict) -> dict:
    a = (1 - cfg["bootstrap"]["ci"]) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "includes_zero": bool(lo <= 0 <= hi)}


def paired_diff(y, pa, pb, sel, timestamps, cfg) -> dict:
    """Interval for RMSE_a - RMSE_b (averaged over horizons), paired by day. Only rows with at
    least one selected cell form day blocks, so empty days never enter a resample."""
    keep = sel.any(axis=1)
    y, pa, pb, sel = y[keep], pa[keep], pb[keep], sel[keep]
    day = day_index(pd.DatetimeIndex(timestamps)[keep])
    n = day.max() + 1
    sa, ca = day_sums(y, pa, sel, day, n)
    sb, cb = day_sums(y, pb, sel, day, n)
    w = weights(n, cfg)
    return {**interval(avg_rmse(sa, ca) - avg_rmse(sb, cb), avg_rmse(sa, ca, w) - avg_rmse(sb, cb, w), cfg),
            "n_days": int(n)}
