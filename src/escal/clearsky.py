"""Clear-sky irradiance and clear-sky index at the Folsom site (pvlib)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def location(cfg: dict) -> pvlib.location.Location:
    s = cfg["site"]
    return pvlib.location.Location(s["latitude"], s["longitude"], tz="UTC", altitude=s["altitude"])


def _utc(times) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(times)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def solar_elevation(times, cfg: dict) -> pd.Series:
    idx = _utc(times)
    sp = location(cfg).get_solarposition(idx)
    return pd.Series(sp["elevation"].values, index=pd.DatetimeIndex(times))


def clearsky_ghi(times, cfg: dict) -> pd.Series:
    idx = _utc(times)
    cs = location(cfg).get_clearsky(idx, model=cfg["clearsky"]["model"])
    return pd.Series(cs["ghi"].values, index=pd.DatetimeIndex(times))


def clearsky_index(ghi, ghi_clear, elevation, min_elevation: float, max_kt: float | None = None):
    """kt = ghi / ghi_clear in daylight; NaN at night (elevation below the cut) and
    wherever the clear-sky value is not positive. Never divides by zero."""
    ghi = np.asarray(ghi, dtype=float)
    cs = np.asarray(ghi_clear, dtype=float)
    el = np.asarray(elevation, dtype=float)
    ok = (el >= min_elevation) & (cs > 0) & np.isfinite(ghi)
    kt = np.full(ghi.shape, np.nan)
    kt[ok] = ghi[ok] / cs[ok]
    if max_kt is not None:
        kt = np.where(ok, np.clip(kt, 0.0, max_kt), np.nan)
    return kt


def minute_kt(irr: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Clear-sky GHI, elevation and kt for a frame with `timestamp` and `ghi`."""
    out = irr[["timestamp", "ghi"]].copy()
    out["elevation"] = solar_elevation(out["timestamp"], cfg).values
    out["ghi_clear"] = clearsky_ghi(out["timestamp"], cfg).values
    c = cfg["clearsky"]
    out["kt"] = clearsky_index(out["ghi"], out["ghi_clear"], out["elevation"],
                               c["daylight_min_elevation"], c["max_kt"])
    return out
