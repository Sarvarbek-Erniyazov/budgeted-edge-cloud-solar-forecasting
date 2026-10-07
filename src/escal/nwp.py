"""NAM forecasts as they would have been available at an issue time."""
from __future__ import annotations

import numpy as np
import pandas as pd

H = pd.Timedelta(hours=1)


def nam_at(nam: pd.DataFrame, issue, valid, lag_h: float, fields: list[str]) -> pd.DataFrame:
    """For each (issue time t, valid time v): the newest run with reftime + lag <= t whose
    hourly valid times bracket v, with `fields` interpolated linearly to v.
    Rows with no such run are NaN. Output is aligned with the inputs."""
    q = pd.DataFrame({"t": pd.DatetimeIndex(issue), "v": pd.DatetimeIndex(valid)})
    q["i"] = np.arange(len(q))
    q["lo"], q["hi"] = q["v"].dt.floor("h"), q["v"].dt.ceil("h")
    a = nam[["reftime", "timestamp"] + fields]
    m = q.merge(a.rename(columns={"timestamp": "lo"}), on="lo")
    m = m[m["reftime"] + pd.Timedelta(hours=lag_h) <= m["t"]]
    m = m.merge(a.rename(columns={"timestamp": "hi"}), on=["reftime", "hi"], suffixes=("_lo", "_hi"))
    m = m.sort_values("reftime").groupby("i").tail(1).set_index("i")
    w = ((m["v"] - m["lo"]) / H).values
    out = pd.DataFrame(index=pd.RangeIndex(len(q)))
    for f in fields:
        out[f] = pd.Series(m[f"{f}_lo"].values * (1 - w) + m[f"{f}_hi"].values * w, index=m.index)
    out["reftime"] = m["reftime"]
    out["lead_h"] = (q["v"] - out["reftime"]) / H
    return out
