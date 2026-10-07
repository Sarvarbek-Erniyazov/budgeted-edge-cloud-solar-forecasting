"""Escalation gates. The unit of decision is one issue time; an escalation returns the
escalation target's forecast for all horizons. Budget = share of issue times escalated."""
from __future__ import annotations

import numpy as np


def gate_input_columns(columns: list[str], forbidden: list[str]) -> list[str]:
    """On-device inputs only. Raises if any column looks cloud-side."""
    bad = [c for c in columns if any(f in c.lower() for f in forbidden)]
    if bad:
        raise ValueError(f"cloud-side columns in gate inputs: {bad}")
    return list(columns)


def fit_thresholds(scores: np.ndarray, budgets: list[float]) -> dict[float, float]:
    """Threshold per budget from the fitting population's scores only: the (1-b) quantile.
    b=0 escalates nothing, b=1 everything."""
    out = {}
    for b in budgets:
        if b <= 0:
            out[b] = float("inf")
        elif b >= 1:
            out[b] = float("-inf")
        else:
            out[b] = float(np.quantile(scores, 1 - b))
    return out


def decide(scores: np.ndarray, tau: float) -> np.ndarray:
    if tau == float("inf"):
        return np.zeros(len(scores), bool)
    return scores >= tau


def fixed_interval(day: np.ndarray, b: float) -> np.ndarray:
    """Escalate when floor((i+1)b) > floor(ib), i counting issue times within each day (in time order)."""
    esc = np.zeros(len(day), bool)
    for d in np.unique(day):
        idx = np.nonzero(day == d)[0]
        i = np.arange(len(idx))
        esc[idx] = np.floor((i + 1) * b + 1e-9) > np.floor(i * b + 1e-9)
    return esc


def oracle(benefit: np.ndarray, b: float) -> np.ndarray:
    k = int(round(b * len(benefit)))
    esc = np.zeros(len(benefit), bool)
    if k > 0:
        esc[np.argsort(-benefit, kind="stable")[:k]] = True
    return esc


def random_draw(n: int, b: float, rng: np.random.Generator) -> np.ndarray:
    esc = np.zeros(n, bool)
    esc[rng.choice(n, int(round(b * n)), replace=False)] = True
    return esc


def blend(edge: np.ndarray, target: np.ndarray, esc: np.ndarray) -> np.ndarray:
    return np.where(esc[:, None], target, edge)


def benefit(y: np.ndarray, edge: np.ndarray, target: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Per issue time: sum over selected horizons of (e_edge^2 - e_target^2)."""
    d = (y - edge) ** 2 - (y - target) ** 2
    return np.where(sel, d, 0.0).sum(axis=1)


def avg_metrics(y, p, sp, sel) -> dict:
    """RMSE, MAE per horizon and skill over smart persistence, averaged over horizons."""
    rm, ma, sk = [], [], []
    for j in range(y.shape[1]):
        s = sel[:, j]
        r = np.sqrt(np.mean((y[s, j] - p[s, j]) ** 2))
        rm.append(r)
        ma.append(np.mean(np.abs(y[s, j] - p[s, j])))
        sk.append(1 - r / np.sqrt(np.mean((y[s, j] - sp[s, j]) ** 2)))
    return {"RMSE": float(np.mean(rm)), "MAE": float(np.mean(ma)), "skill": float(np.mean(sk)),
            "rmse_per_horizon": [float(x) for x in rm]}


def share_retained(rmse_edge: float, rmse_gate: float, rmse_cloud: float) -> float:
    return (rmse_edge - rmse_gate) / (rmse_edge - rmse_cloud)
