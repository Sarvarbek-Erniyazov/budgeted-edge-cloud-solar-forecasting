"""Inputs for the edge and cloud tiers, built only from values available at the issue time."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from escal.data import drop_targets_reaching_test, horizon_minutes, horizons, nam_files, read_dev
from escal.nwp import nam_at
from escal.splits import _bounds, select_split

NAM_FIELDS = ["dwsw", "cloud_cover", "precipitation", "pressure", "wind-u", "wind-v", "temperature", "rel_humidity"]
MIN = pd.Timedelta(minutes=1)


@dataclass
class Base:
    """Everything except the satellite tiles and NAM, one row per issue time."""
    df: pd.DataFrame                 # timestamp, targets, benchmark features, part
    hz: list[str]
    ground_cols: list[str]           # edge inputs (before standardisation)
    endo_cols: list[str]             # benchmark intra-day ground features (anchors)
    sif_cols: list[str]              # benchmark satellite features (anchors)
    sat_t: np.ndarray = field(repr=False)
    sat_px: np.ndarray = field(repr=False)
    nams: dict = field(repr=False)


def load_base(cfg: dict) -> Base:
    tar = read_dev("Target_intra-day.csv", cfg)
    hz = horizons(tar)
    tar = drop_targets_reaching_test(tar, cfg, max(horizon_minutes(h) for h in hz))
    endo = read_dev("Irradiance_features_intra-day.csv", cfg)
    ih = read_dev("Irradiance_features_intra-hour.csv", cfg)
    ih.columns = ["timestamp"] + [f"ih_{c}" for c in ih.columns[1:]]
    sif = read_dev("Sat_image_features_intra-day.csv", cfg)
    wx = read_dev("Folsom_weather.csv", cfg).set_index("timestamp")
    wx = wx.resample("30min", closed="right", label="right").mean()     # mean over (t-30min, t]
    wx.columns = [f"wx_{c}" for c in wx.columns]
    df = (tar.merge(endo, on="timestamp").merge(ih, on="timestamp", how="left")
          .merge(sif, on="timestamp").merge(wx, left_on="timestamp", right_index=True, how="left"))

    t = df["timestamp"]
    doy = t.dt.dayofyear / 365.25 * 2 * np.pi
    hod = (t.dt.hour + t.dt.minute / 60) / 24 * 2 * np.pi
    det = pd.DataFrame({"doy_sin": np.sin(doy), "doy_cos": np.cos(doy), "hod_sin": np.sin(hod), "hod_cos": np.cos(hod)})
    for h in hz:
        det[f"cs_{h}"] = df[f"ghi_clear_{h}"] / 1000.0
    df = pd.concat([df, det], axis=1)
    wx_cols = list(wx.columns)
    df["wx_missing"] = df[wx_cols].isna().any(axis=1).astype(float)
    endo_cols = [c for c in endo.columns if c != "timestamp"]
    ground = (endo_cols + [c for c in ih.columns if c != "timestamp"] + wx_cols + ["wx_missing"]
              + [f"elevation_{h}" for h in hz] + list(det.columns))
    df["part"] = assign_parts(df, cfg, max(horizon_minutes(h) for h in hz))

    sat = read_dev("Folsom_satellite.csv", cfg)
    nams = {p.name: read_dev(p.name, cfg) for p in nam_files(cfg)}
    return Base(df=df, hz=hz, ground_cols=ground, endo_cols=endo_cols,
                sif_cols=[c for c in sif.columns if c != "timestamp"],
                sat_t=sat["timestamp"].values, sat_px=sat.iloc[:, 1:].values.astype(np.float32).reshape(-1, 10, 10),
                nams=nams)


def assign_parts(df: pd.DataFrame, cfg: dict, max_m: int) -> pd.Series:
    """train / es (last days of models_train, early stopping only) / val; targets never cross."""
    part = pd.Series(None, index=df.index, dtype=object)
    t, reach = df["timestamp"], df["timestamp"] + pd.Timedelta(minutes=max_m)
    s, e = _bounds(cfg, "models_train")
    es_start = e - pd.Timedelta(days=cfg["tiers"]["early_stop_days"])
    part[(t >= s) & (reach < es_start)] = "train"
    part[(t >= es_start) & (reach < e)] = "es"
    vs, ve = _bounds(cfg, "validation")
    part[(t >= vs) & (reach < ve)] = "val"
    return part


def satellite(base: Base, lag_min: int, n_frames: int, lookback_min: int, diffs: bool):
    """Newest `n_frames` tiles stamped in [t-lag-lookback, t-lag], newest first.
    Returns tiles (N, C, 10, 10), meta (N, 2*n_frames) [age/60, present], frame stamps (N, n_frames)."""
    t = base.df["timestamp"].values
    end = t - np.timedelta64(lag_min, "m")
    start = end - np.timedelta64(lookback_min, "m")
    hi = np.searchsorted(base.sat_t, end, side="right") - 1
    n = len(t)
    tiles = np.zeros((n, n_frames, 10, 10), np.float32)
    present = np.zeros((n, n_frames), np.float32)
    age = np.zeros((n, n_frames), np.float32)
    stamps = np.full((n, n_frames), np.datetime64("NaT"), dtype="datetime64[ns]")
    for k in range(n_frames):
        j = hi - k
        ok = (j >= 0) & (base.sat_t[np.clip(j, 0, None)] >= start)
        jj = j[ok]
        tiles[ok, k] = base.sat_px[jj] / 255.0
        present[ok, k] = 1.0
        age[ok, k] = (t[ok] - base.sat_t[jj]) / np.timedelta64(1, "m") / 60.0
        stamps[ok, k] = base.sat_t[jj]
    chans = [tiles]
    if diffs and n_frames > 1:
        both = (present[:, :-1] * present[:, 1:])[:, :, None, None]
        chans.append((tiles[:, :-1] - tiles[:, 1:]) * both)
    return np.concatenate(chans, axis=1), np.concatenate([age, present], axis=1), stamps


def sat_available(base: Base, lag_min: int, max_age_min: int) -> np.ndarray:
    _, meta, _ = satellite(base, lag_min, 1, max_age_min, False)
    return meta[:, 1] > 0


def nwp(base: Base, cfg: dict, extra: bool) -> tuple[pd.DataFrame, dict]:
    """NAM features per horizon at valid time t+h-15min (4 nodes, all fields, dwsw/clear,
    interpolation and missing flags). `extra` adds the forecast valid at the issue time,
    its error against the measured last-30-min kt, and dwsw/cloud cover at v-1h and v+1h."""
    tc = cfg["tiers"]
    lag = cfg["nam"]["availability_lag_hours"]
    df = base.df
    t = df["timestamp"]
    cols, info = {}, {}
    lo, hi = tc["nam_ratio_clip"]
    for n_i, (name, nam) in enumerate(sorted(base.nams.items())):
        for h in base.hz:
            v = t + pd.Timedelta(minutes=horizon_minutes(h) + tc["nam_valid_offset_minutes"])
            got = nam_at(nam, t, v, lag, NAM_FIELDS, tc["nam_interp_max_gap_h"])
            for f in NAM_FIELDS:
                cols[f"nam{n_i}_{f}_{h}"] = got[f].values
            clear = df[f"ghi_clear_{h}"].where(df[f"ghi_clear_{h}"] >= tc["nam_min_clear"])
            cols[f"nam{n_i}_ratio_{h}"] = (got["dwsw"].values / clear.values).clip(lo, hi)
            if n_i == 0:
                cols[f"nam_interp_{h}"] = got["interpolated"].fillna(0).values
                cols[f"nam_missing_{h}"] = got["dwsw"].isna().astype(float).values
                info[h] = {"reftime_plus_lag_le_t": bool(((got["reftime"] + pd.Timedelta(hours=lag) <= t)
                                                          | got["reftime"].isna()).all()),
                           "lead_h_min": float(got["lead_h"].min()), "lead_h_max": float(got["lead_h"].max()),
                           "share_missing": float(got["dwsw"].isna().mean()),
                           "share_interpolated": float(got["interpolated"].fillna(0).mean())}
            if extra:
                for dt in (-60, 60):
                    g2 = nam_at(nam, t, v + pd.Timedelta(minutes=dt), lag, ["dwsw", "cloud_cover"],
                                tc["nam_interp_max_gap_h"])
                    cols[f"nam{n_i}_dwsw_{h}_{dt:+d}m"] = g2["dwsw"].values
                    cols[f"nam{n_i}_cc_{h}_{dt:+d}m"] = g2["cloud_cover"].values
        if extra:
            v0 = t + pd.Timedelta(minutes=tc["nam_valid_offset_minutes"])
            g0 = nam_at(nam, t, v0, lag, ["dwsw", "cloud_cover"], tc["nam_interp_max_gap_h"])
            # benchmark clear-sky for the 30-min window ending at t (row t-30min, horizon 30min)
            c0 = (df.set_index("timestamp")["ghi_clear_30min"]
                  .reindex(t - pd.Timedelta(minutes=30)).values)
            c0 = np.where(c0 >= tc["nam_min_clear"], c0, np.nan)
            r0 = np.clip(g0["dwsw"].values / c0, lo, hi)
            cols[f"nam{n_i}_dwsw_t"] = g0["dwsw"].values
            cols[f"nam{n_i}_cc_t"] = g0["cloud_cover"].values
            cols[f"nam{n_i}_err_t"] = r0 - df["B(ghi_kt|30min)"].values
    return pd.DataFrame(cols, index=df.index), info


def standardise(x: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    mu = x[train_mask].mean()
    sd = x[train_mask].std().replace(0, 1).fillna(1)
    return ((x - mu) / sd).fillna(0.0).values.astype(np.float32)


def targets(base: Base, cfg: dict) -> dict:
    df, hz = base.df, base.hz
    el = np.stack([df[f"elevation_{h}"].values for h in hz], 1)
    kt = np.stack([df[f"ghi_kt_{h}"].values for h in hz], 1)
    clear = np.stack([df[f"ghi_clear_{h}"].values for h in hz], 1)
    ghi = np.stack([df[f"ghi_{h}"].values for h in hz], 1)
    day = (el >= cfg["tiers"]["daylight_min_elevation"]) & np.isfinite(kt)
    sp = df["B(ghi_kt|30min)"].values[:, None] * clear
    sp = np.where(day, sp, np.nan)
    return {"kt": np.nan_to_num(kt).astype(np.float32), "mask": day.astype(np.float32),
            "clear": clear, "ghi": ghi, "sp": sp, "day": day}
