"""Intra-day benchmark data and the persistence / linear anchors, following the logic of
Forecast_intra-day.py and Postprocess.py, restricted to development data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from escal.data import drop_targets_reaching_test, horizon_minutes, horizons, read_dev
from escal.splits import _bounds, select_split


def load_intra_day(cfg: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Targets joined with ground features and satellite features, development rows only.
    Returns the frame, the endogenous feature names and the satellite feature names."""
    tar = read_dev("Target_intra-day.csv", cfg)
    endo = read_dev("Irradiance_features_intra-day.csv", cfg)
    exo = read_dev("Sat_image_features_intra-day.csv", cfg)
    hz = horizons(tar)
    tar = drop_targets_reaching_test(tar, cfg, max(horizon_minutes(h) for h in hz))
    df = tar.merge(endo, on="timestamp", how="inner").merge(exo, on="timestamp", how="inner")
    return df, [c for c in endo.columns if c != "timestamp"], [c for c in exo.columns if c != "timestamp"]


def split_rows(df: pd.DataFrame, cfg: dict, name: str, max_minutes: int) -> pd.DataFrame:
    """Rows of one split whose furthest target time is still inside that split."""
    part = select_split(df, cfg, name)
    _, end = _bounds(cfg, name)
    return part.loc[part["timestamp"] + pd.Timedelta(minutes=max_minutes) < end]


def metrics(y, f, ref) -> dict:
    """Postprocess.py: errors over non-NaN forecasts; skill = 1 - RMSE / RMSE(reference)."""
    y, f, ref = (np.asarray(a, float) for a in (y, f, ref))
    e = y - f
    rmse = float(np.sqrt(np.nanmean(e ** 2)))
    rmse_p = float(np.sqrt(np.nanmean((y - ref) ** 2)))
    return {"n": int(np.isfinite(e).sum()), "RMSE": rmse, "MAE": float(np.nanmean(np.abs(e))),
            "MBE": float(np.nanmean(e)), "skill": 1.0 - rmse / rmse_p}


def run_anchors(df, endo_cols, exo_cols, cfg) -> pd.DataFrame:
    a = cfg["anchors"]
    hz = horizons(df)
    max_m = max(horizon_minutes(h) for h in hz)
    rows = []
    for target in a["targets"]:
        feat_endo = [c for c in endo_cols if target in c]          # inpEndo.filter(regex=target)
        feat_all = feat_endo + exo_cols
        for h in hz:
            cols = [f"{target}_{h}", f"{target}_kt_{h}", f"{target}_clear_{h}", f"elevation_{h}"]
            tr = split_rows(df, cfg, a["fit_split"], max_m)[cols + feat_all].dropna(how="any")
            te = split_rows(df, cfg, a["eval_split"], max_m)[cols + feat_all].dropna(how="any")
            night = te[f"elevation_{h}"].values < a["night_elevation"]
            y, clear = te[f"{target}_{h}"].values, te[f"{target}_clear_{h}"].values
            sp = te[f"B({target}_kt|30min)"].values * clear
            sp[night] = np.nan
            preds = {"sp": sp}
            for fs, feats in (("endo", feat_endo), ("exo", feat_all)):
                sc = StandardScaler().fit(tr[feats].values)
                xtr, xte = sc.transform(tr[feats].values), sc.transform(te[feats].values)
                models = [("ols", linear_model.LinearRegression()),
                          ("ridge", linear_model.RidgeCV(cv=a["cv_folds"])),
                          ("lasso", linear_model.LassoCV(cv=a["cv_folds"], n_jobs=-1, max_iter=a["lasso_max_iter"]))]
                for name, m in models:
                    m.fit(xtr, tr[f"{target}_kt_{h}"].values)
                    p = np.clip(m.predict(xte), *a["kt_clip"]) * clear
                    p[night] = np.nan
                    preds[f"{name}_{fs}"] = p
            for name, p in preds.items():
                rows.append({"target": target, "horizon": h, "model": name, "fit_split": a["fit_split"],
                             "eval_split": a["eval_split"], "n_fit": len(tr),
                             **metrics(y, p, preds["sp"])})
    return pd.DataFrame(rows)


def anchor_predict(df: pd.DataFrame, fit_rows: np.ndarray, feats: list[str], drop_cols: list[str],
                   target: str, h: str, cfg: dict) -> np.ndarray:
    """LassoCV anchor as in run_anchors (fit rows dropna over `drop_cols`, kt target, clip,
    times ghi_clear_h, night removed), predicted for every row of `df` whose features exist."""
    a = cfg["anchors"]
    cols = [f"{target}_{h}", f"{target}_kt_{h}", f"{target}_clear_{h}", f"elevation_{h}"]
    tr = df.loc[fit_rows, list(dict.fromkeys(cols + drop_cols))].dropna(how="any")
    sc = StandardScaler().fit(tr[feats].values)
    m = linear_model.LassoCV(cv=a["cv_folds"], n_jobs=-1, max_iter=a["lasso_max_iter"])
    m.fit(sc.transform(tr[feats].values), tr[f"{target}_kt_{h}"].values)
    out = np.full(len(df), np.nan)
    ok = df[feats].notna().all(axis=1).values
    out[ok] = np.clip(m.predict(sc.transform(df.loc[ok, feats].values)), *a["kt_clip"]) * df.loc[ok, f"{target}_clear_{h}"].values
    out[df[f"elevation_{h}"].values < a["night_elevation"]] = np.nan
    return out
