"""NAM forecasts as they would have been available at an issue time."""
from __future__ import annotations

import numpy as np
import pandas as pd

H = pd.Timedelta(hours=1)


def nam_at(nam: pd.DataFrame, issue, valid, lag_h: float, fields: list[str],
           max_gap_h: float = 1.0) -> pd.DataFrame:
    """For each (issue time t, valid time v): the newest run with reftime + lag <= t whose
    valid times span v. `fields` are interpolated linearly in time between the two valid
    times of that same run that bracket v, if they are at most `max_gap_h` apart; runs are
    never mixed. `interpolated` is 1 when the bracketing valid times are more than 1 h apart.
    Rows with no such run are NaN. Output is aligned with the inputs."""
    t = pd.DatetimeIndex(issue).values
    v = pd.DatetimeIndex(valid).values
    span = nam.groupby("reftime")["timestamp"].agg(["min", "max"])
    ref = span.index.values
    ok = ((ref[None, :] + np.timedelta64(int(lag_h * 3600), "s") <= t[:, None])
          & (span["min"].values[None, :] <= v[:, None]) & (span["max"].values[None, :] >= v[:, None]))
    pick = np.where(ok, np.arange(len(ref))[None, :], -1).max(axis=1)  # runs sorted: newest = largest

    out = pd.DataFrame(index=pd.RangeIndex(len(v)), columns=fields + ["reftime", "lead_h", "interpolated"],
                       dtype=float)
    out["reftime"] = pd.NaT
    has = pick >= 0
    if not has.any():
        return out
    q = pd.DataFrame({"i": np.nonzero(has)[0], "v": v[has], "reftime": ref[pick[has]]}).sort_values("v")
    a = nam[["reftime", "timestamp"] + fields].sort_values("timestamp")
    lo = pd.merge_asof(q, a, left_on="v", right_on="timestamp", by="reftime", direction="backward")
    hi = pd.merge_asof(q, a, left_on="v", right_on="timestamp", by="reftime", direction="forward")
    lo, hi = lo.set_index("i").sort_index(), hi.set_index("i").sort_index()
    gap = (hi["timestamp"] - lo["timestamp"]) / H
    w = np.where(gap > 0, ((lo["v"] - lo["timestamp"]) / H) / gap.where(gap > 0, 1.0), 0.0)
    good = (gap <= max_gap_h).values
    idx = lo.index[good]
    for f in fields:
        out.loc[idx, f] = (lo[f].values * (1 - w) + hi[f].values * w)[good]
    out.loc[idx, "reftime"] = lo["reftime"].values[good]
    out["reftime"] = pd.to_datetime(out["reftime"])
    out.loc[idx, "interpolated"] = (gap.values[good] > 1.0).astype(float)
    out["lead_h"] = (pd.Series(v) - out["reftime"]) / H
    return out
