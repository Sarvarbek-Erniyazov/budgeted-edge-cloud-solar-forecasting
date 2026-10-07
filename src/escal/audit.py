"""Data-audit checks. Each function returns plain dicts so the results go to JSON."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from escal.data import horizon_minutes, horizons
from escal.nwp import nam_at

MIN = pd.Timedelta(minutes=1)


def _ts(x) -> str | None:
    return None if pd.isna(x) else str(pd.Timestamp(x))


def schema(df: pd.DataFrame, top: int = 5) -> dict:
    t = df["timestamp"]
    steps = t.diff().dropna()
    modal = steps.mode().iloc[0] if len(steps) else pd.NaT
    gaps = steps[steps > modal] if len(steps) else steps
    big = gaps.sort_values(ascending=False).head(top)
    return {
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(d) for c, d in df.dtypes.items()},
        "rows": int(len(df)),
        "time_start": _ts(t.min()),
        "time_end": _ts(t.max()),
        "modal_step": str(modal),
        "step_counts_top": {str(k): int(v) for k, v in steps.value_counts().head(top).items()},
        "duplicate_timestamps": int(t.duplicated().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "steps_longer_than_modal": int(len(gaps)),
        "largest_gaps": [{"after": _ts(t.loc[i - 1]), "length": str(v)} for i, v in big.items()],
        "missing_values_total": int(df.drop(columns="timestamp").isna().sum().sum()),
        "rows_with_any_missing": int(df.drop(columns="timestamp").isna().any(axis=1).sum()),
        "timestamps_naive": t.dt.tz is None,
    }


def minute_coverage(df: pd.DataFrame) -> dict:
    t = df["timestamp"]
    full = pd.date_range(t.min(), t.max(), freq="min")
    missing = full.difference(pd.DatetimeIndex(t))
    days = pd.Series(missing.date).value_counts()
    return {"expected_minutes": int(len(full)), "missing_minutes": int(len(missing)),
            "days_with_missing_minutes": int(len(days)),
            "days_with_over_60_missing": int((days > 60).sum())}


def tz_evidence(irr: pd.DataFrame, cfg: dict, n_days: int = 40) -> dict:
    """Night GHI by UTC hour, and the GHI centroid on the smoothest days against
    pvlib solar transit at the configured site."""
    g = irr.set_index("timestamp")["ghi"]
    by_hour = g.groupby(g.index.hour).median()
    days = []
    for d, x in g.groupby((g.index - pd.Timedelta(hours=8)).normalize()):
        if len(x) < 1200 or x.max() < 300:
            continue
        days.append((np.abs(np.diff(x.values, 2)).mean(), d, x))
    days.sort(key=lambda r: r[0])
    s = cfg["site"]
    offs = []
    for _, d, x in days[:n_days]:
        x = x.clip(lower=0)
        m = (x.index - d).total_seconds().values / 60.0
        cen = float((m * x.values).sum() / x.values.sum())
        tr = pvlib.solarposition.sun_rise_set_transit_spa(
            pd.DatetimeIndex([d + pd.Timedelta(hours=12)]).tz_localize("UTC"),
            s["latitude"], s["longitude"])["transit"].iloc[0]
        offs.append(cen - (tr.tz_convert("UTC").tz_localize(None) - d).total_seconds() / 60.0)
    return {
        "dataset_statement": "Zenodo record 2826939: 'All time stamps are in UTC'",
        "median_ghi_by_utc_hour": {int(h): round(float(v), 2) for h, v in by_hour.items()},
        "clear_days_used": len(offs),
        "ghi_centroid_minus_solar_transit_min_mean": round(float(np.mean(offs)), 2),
        "ghi_centroid_minus_solar_transit_min_sd": round(float(np.std(offs)), 2),
    }


def target_window_check(irr: pd.DataFrame, tar: pd.DataFrame, n: int = 1500, seed: int = 0) -> dict:
    """Which minute window does target column var_h in row t average?"""
    g = irr.set_index("timestamp")["ghi"]
    cs = g.cumsum()
    cnt = pd.Series(1.0, index=g.index).cumsum()
    rng = np.random.default_rng(seed)
    samp = tar.iloc[np.sort(rng.choice(len(tar), min(n, len(tar)), replace=False))]
    out = {}
    cands = {"(t+h-30, t+h]": (-30, 0, "right"), "[t+h-30, t+h)": (-30, 0, "left"),
             "(t+h, t+h+30]": (0, 30, "right"), "[t+h-15, t+h+15)": (-15, 15, "left")}

    def wmean(ends_lo, ends_hi):
        a = cs.reindex(ends_lo, method="ffill").values
        b = cs.reindex(ends_hi, method="ffill").values
        na = cnt.reindex(ends_lo, method="ffill").values
        nb = cnt.reindex(ends_hi, method="ffill").values
        return (b - a) / (nb - na)

    for h in horizons(tar):
        hm = horizon_minutes(h)
        res = {}
        for name, (a, b, closed) in cands.items():
            lo = samp["timestamp"] + pd.Timedelta(minutes=hm + a)
            hi = samp["timestamp"] + pd.Timedelta(minutes=hm + b)
            if closed == "left":
                lo, hi = lo - MIN, hi - MIN
            d = np.abs(wmean(pd.DatetimeIndex(lo), pd.DatetimeIndex(hi)) - samp[f"ghi_{h}"].values)
            res[name] = {"median_abs_diff": float(np.nanmedian(d)), "max_abs_diff": float(np.nanmax(d))}
        out[h] = res
    return out


def feature_definition_check(tar: pd.DataFrame, fea: pd.DataFrame) -> dict:
    """B, L features against 30-min block kt taken from the targets."""
    blk = tar.set_index("timestamp")["ghi_kt_30min"]
    blk.index = blk.index + pd.Timedelta(minutes=30)          # block ending at index time
    f = fea.set_index("timestamp")
    lag = blk.copy()
    lag.index = lag.index + pd.Timedelta(minutes=30)
    j = f.join(blk.rename("b0")).join(lag.rename("b1")).dropna(subset=["b0", "b1"])
    def share(a, b):
        return float((np.abs(a - b) < 1e-5).mean())
    pair = j[["b0", "b1"]].values
    return {
        "rows_compared": int(len(j)),
        "B(ghi_kt|30min) == kt of the 30-min block ending at t": share(j["B(ghi_kt|30min)"], j["b0"]),
        "L(ghi_kt|60min) == kt of the block ending at t-30min": share(j["L(ghi_kt|60min)"], j["b1"]),
        "B(ghi_kt|60min) == mean of the two blocks ending at t and t-30min":
            share(j["B(ghi_kt|60min)"], pair.mean(1)),
        "V(ghi_kt|60min) == std of those two blocks (ddof 0)": share(j["V(ghi_kt|60min)"], pair.std(1)),
        "V(ghi_kt|60min) == std of those two blocks (ddof 1)": share(j["V(ghi_kt|60min)"], pair.std(1, ddof=1)),
        "V(ghi_kt|30min) nonzero share (one block only)": float((f["V(ghi_kt|30min)"] > 0).mean()),
    }


def elevation_convention_check(tar: pd.DataFrame, cfg: dict, n: int = 400, seed: int = 0) -> dict:
    """elevation_h in row t against pvlib elevation at the configured site."""
    s = cfg["site"]
    rng = np.random.default_rng(seed)
    samp = tar.iloc[np.sort(rng.choice(len(tar), min(n, len(tar)), replace=False))]
    out = {}
    for h in horizons(tar):
        hm = horizon_minutes(h)
        res = {}
        mins = np.arange(-29, 1)
        allt = pd.DatetimeIndex((samp["timestamp"].values[:, None]
                                 + pd.to_timedelta(hm + mins, "min").values[None, :]).ravel())
        e = pvlib.solarposition.get_solarposition(allt.tz_localize("UTC"), s["latitude"], s["longitude"])
        e = e["elevation"].values.reshape(len(samp), len(mins))
        for name, val in {"mean over minutes t+h-29..t+h": e.mean(1),
                          "instant t+h": e[:, -1], "instant t+h-15min": e[:, 15]}.items():
            d = val - samp[f"elevation_{h}"].values
            res[name] = {"rms_deg": float(np.sqrt(np.mean(d ** 2))), "max_abs_deg": float(np.abs(d).max())}
        out[h] = res
    return out


def kt_definition_check(tar: pd.DataFrame) -> dict:
    out = {}
    for h in horizons(tar):
        r = (tar[f"ghi_{h}"] / tar[f"ghi_clear_{h}"]).clip(upper=1.2)
        k = tar[f"ghi_kt_{h}"]
        ok = k.notna() & r.notna()
        out[h] = {"share_kt_equals_ratio_of_means": float((np.abs(r[ok] - k[ok]) < 1e-4).mean()),
                  "kt_min": float(k.min()), "kt_max": float(k.max())}
    return out


def satellite_feature_check(sat: pd.DataFrame, sif: pd.DataFrame, window_min: int) -> dict:
    """Is the benchmark satellite feature at t the mean of raw frames stamped in (t-w, t]?"""
    s = sat.set_index("timestamp")
    f = sif.set_index("timestamp").dropna()
    px = s.values.astype(float)
    t_sat = s.index.values
    match = n_frames = 0
    nf = []
    for t, row in f.iterrows():
        lo = np.searchsorted(t_sat, np.datetime64(t - pd.Timedelta(minutes=window_min)), side="right")
        hi = np.searchsorted(t_sat, np.datetime64(t), side="right")
        nf.append(hi - lo)
        if hi > lo and np.allclose(px[lo:hi].mean(0), row.values, atol=1e-3):
            match += 1
    nf = pd.Series(nf)
    return {"rows_with_satellite": int(len(f)),
            "share_equal_to_mean_of_frames_in_(t-30min,t]": match / len(f),
            "frames_in_window_counts": {int(k): int(v) for k, v in nf.value_counts().sort_index().items()},
            "rows_all_missing": int(sif.drop(columns="timestamp").isna().all(axis=1).sum())}


def satellite_availability(sat: pd.DataFrame, issue: pd.Series, lag_min: int) -> dict:
    """Age of the newest frame usable at each issue time, with and without the latency."""
    t_sat = sat["timestamp"].values
    out = {}
    for lag in sorted({0, lag_min}):
        usable = issue.values - np.timedelta64(lag, "m")
        i = np.searchsorted(t_sat, usable, side="right") - 1
        age = np.where(i >= 0, (issue.values - t_sat[np.clip(i, 0, None)]) / np.timedelta64(1, "m"), np.nan)
        age = pd.Series(age)
        out[f"lag_{lag}min"] = {"median_age_min": float(age.median()),
                                "p90_age_min": float(age.quantile(0.9)),
                                "share_age_le_60min": float((age <= 60).mean())}
    return out


def nam_summary(nam: pd.DataFrame) -> dict:
    ref = nam["reftime"]
    days = pd.date_range(ref.min().normalize(), ref.max().normalize(), freq="D")
    fields = [c for c in nam.columns if c not in ("timestamp", "reftime", "lead_h")]
    return {
        "reftime_hours_utc": sorted(int(h) for h in ref.dt.hour.unique()),
        "runs": int(ref.nunique()),
        "days_without_run": int(len(days.difference(pd.DatetimeIndex(ref.dt.normalize().unique())))),
        "lead_hours": sorted(float(x) for x in nam["lead_h"].unique()),
        "rows_per_run": {int(k): int(v) for k, v in ref.value_counts().value_counts().items()},
        "duplicate_reftime_valtime": int(nam.duplicated(["reftime", "timestamp"]).sum()),
        "missing_by_field": {c: int(nam[c].isna().sum()) for c in fields},
        "field_ranges": {c: [float(nam[c].min()), float(nam[c].max())] for c in fields},
    }


def nam_alignment(nam: pd.DataFrame, tar: pd.DataFrame, lag_h: float, min_elev: float) -> dict:
    """For each daylight target (issue t, valid v=t+h): the newest run with
    reftime + lag <= t whose hourly valid times bracket v. Reports coverage and margins."""
    reft = np.sort(nam["reftime"].unique())
    out = {}
    for h in horizons(tar):
        hm = horizon_minutes(h)
        t = tar.loc[tar[f"elevation_{h}"] >= min_elev, "timestamp"].reset_index(drop=True)
        got = nam_at(nam, t, t + pd.Timedelta(minutes=hm), lag_h, [])
        ok = got["reftime"].notna()
        newest = reft[np.searchsorted(reft, (t - pd.Timedelta(hours=lag_h)).values, side="right") - 1]
        margin = (t[ok] - got.loc[ok, "reftime"]) / pd.Timedelta(hours=1)
        out[h] = {"daylight_targets": int(len(t)), "covered": int(ok.sum()),
                  "share_covered": float(ok.mean()) if len(t) else None,
                  "lead_h_min": float(got["lead_h"].min()) if ok.any() else None,
                  "lead_h_max": float(got["lead_h"].max()) if ok.any() else None,
                  "issue_minus_reftime_h_min": float(margin.min()) if ok.any() else None,
                  "share_where_newest_available_run_lacked_valid_time":
                      float((got.loc[ok, "reftime"].values != newest[ok.values]).mean()) if ok.any() else None}
    return out
