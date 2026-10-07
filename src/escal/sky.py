"""Daily sky-condition classes from the clear-sky index level and its variability."""
from __future__ import annotations

import numpy as np
import pandas as pd

from escal.splits import select_split

CLASSES = ["clear", "partly_cloudy", "overcast"]
DEV_SPLITS = ["models_train", "gate_fit", "validation"]


def daily_stats(mk: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """mk: minute frame from clearsky.minute_kt. Days are UTC dates of the local daytime
    (Folsom daylight lies between about 13 and 03 UTC, so days are shifted by 8 h first)."""
    s = cfg["sky"]
    x = mk.set_index("timestamp")
    step = pd.Timedelta(s["resample"])
    k = x["kt"].resample(step).mean()
    el = x["elevation"].resample(step).mean()
    k = k.where(el >= s["min_elevation"])
    expected = (el >= s["min_elevation"]).groupby((el.index - pd.Timedelta(hours=8)).date).sum()
    day = (k.index - pd.Timedelta(hours=8)).date
    g = k.groupby(day)
    d = pd.DataFrame({
        "mean_kt": g.mean(),
        "variability": g.apply(lambda v: np.sqrt(np.nanmean(np.diff(v.values) ** 2))
                               if v.notna().sum() > 2 else np.nan),
        "n": g.count(),
    })
    d["coverage"] = d["n"] / expected.reindex(d.index).replace(0, np.nan)
    d.index = pd.to_datetime(d.index)
    d.index.name = "date"
    return d


def classify(d: pd.DataFrame, cfg: dict) -> pd.Series:
    s = cfg["sky"]
    low = d["variability"] < s["low_variability"]
    cls = np.where(low & (d["mean_kt"] >= s["clear_min_mean_kt"]), "clear",
                   np.where(low & (d["mean_kt"] < s["overcast_max_mean_kt"]), "overcast", "partly_cloudy"))
    cls = pd.Series(cls, index=d.index)
    return cls.where(d["coverage"] >= s["min_coverage"], "insufficient")


def monthly_mix(cls: pd.Series, cfg: dict) -> pd.DataFrame:
    f = cls.rename("sky").reset_index().rename(columns={"date": "timestamp"})
    rows = []
    for split in DEV_SPLITS:
        part = select_split(f, cfg, split)
        part["month"] = part["timestamp"].dt.strftime("%Y-%m")
        for month, grp in part.groupby("month"):
            c = grp["sky"].value_counts()
            n_cls = int(sum(c.get(k, 0) for k in CLASSES))
            row = {"split": split, "month": month, "days": len(grp), "classified_days": n_cls}
            for k in CLASSES + ["insufficient"]:
                row[k] = int(c.get(k, 0))
            for k in CLASSES:
                row[f"{k}_share"] = row[k] / n_cls if n_cls else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
