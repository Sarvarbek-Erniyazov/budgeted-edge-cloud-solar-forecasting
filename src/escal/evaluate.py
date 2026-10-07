"""Scoring in W/m2 on fixed row sets; every model in a call is scored on identical rows."""
from __future__ import annotations

import numpy as np

from escal.benchmark import metrics


def common_rows(preds: dict, rows: np.ndarray, day: np.ndarray, strict: bool = True) -> np.ndarray:
    """(N, H) mask: rows & daylight & every model finite. With strict=True a model that is
    missing where the others exist raises, so no model is silently scored on fewer rows."""
    base = rows[:, None] & day
    fin = np.ones_like(base)
    for name, p in preds.items():
        f = np.isfinite(p)
        if strict and (base & ~f).any():
            raise ValueError(f"{name} has no prediction on {int((base & ~f).sum())} required cells")
        fin &= f
    return base & fin


def score(preds: dict, sel: np.ndarray, ghi: np.ndarray, sp: np.ndarray, hz: list[str]) -> list[dict]:
    rows = []
    for name, p in preds.items():
        per = []
        for j, h in enumerate(hz):
            s = sel[:, j]
            r = metrics(ghi[s, j], p[s, j], sp[s, j])
            per.append(r)
            rows.append({"model": name, "horizon": h, **r})
        rows.append({"model": name, "horizon": "mean", "n": int(sum(r["n"] for r in per)),
                     **{k: float(np.mean([r[k] for r in per])) for k in ("RMSE", "MAE", "MBE", "skill")}})
    return rows
