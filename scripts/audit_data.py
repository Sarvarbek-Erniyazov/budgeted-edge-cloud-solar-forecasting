"""Stage: data audit on the development period (2014-2015). Writes results/audit/ and figures/audit/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from escal import audit
from escal.clearsky import location, minute_kt
from escal.data import FILES, drop_targets_reaching_test, horizon_minutes, horizons, load_config, nam_files, read_dev
from escal.sky import CLASSES, DEV_SPLITS, classify, daily_stats, monthly_mix
from escal.splits import _bounds

OUT = Path("results/audit")
FIG = Path("figures/audit")
# Okabe-Ito, colour-blind-safe; hatching as a second encoding
SKY_STYLE = {"clear": ("#E69F00", ""), "partly_cloudy": ("#56B4E9", "///"), "overcast": ("#0072B2", "..")}


def dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # a. schema of every CSV, development rows only
    frames, sch = {}, {}
    names = list(FILES) + [p.name for p in nam_files(cfg)]
    for name in names:
        df = read_dev(name, cfg)
        sch[name] = audit.schema(df)
        if name in ("Folsom_irradiance.csv", "Folsom_weather.csv"):
            sch[name]["minute_coverage"] = audit.minute_coverage(df)
        if name in ("Folsom_irradiance.csv", "Target_intra-day.csv", "Irradiance_features_intra-day.csv",
                    "Sat_image_features_intra-day.csv", "Folsom_satellite.csv") or name.startswith("Folsom_NAM"):
            frames[name] = df
        print(f"read {name}: {len(df)} rows")
    irr = frames["Folsom_irradiance.csv"]
    sch["Folsom_irradiance.csv"]["time_zone_evidence"] = audit.tz_evidence(irr, cfg)
    sch["_note"] = ("Development rows only (before the test start in configs/base.yaml). "
                    "time_end is the last development time stamp, not the end of the file.")
    dump(sch, OUT / "schema.json")

    # b. task definition
    tar = frames["Target_intra-day.csv"]
    fea = frames["Irradiance_features_intra-day.csv"]
    sif = frames["Sat_image_features_intra-day.csv"]
    hz = horizons(tar)
    tar_c = drop_targets_reaching_test(tar, cfg, max(horizon_minutes(h) for h in hz))
    checks = {
        "horizons": hz,
        "target_rows": len(tar), "target_rows_after_dropping_targets_in_test_year": len(tar_c),
        "issue_time_minutes_past_hour": sorted(int(m) for m in tar["timestamp"].dt.minute.unique()),
        "issue_rows_by_utc_hour": {int(k): int(v) for k, v in tar["timestamp"].dt.hour.value_counts().sort_index().items()},
        "same_timestamps_target_features_satellite": bool(
            tar["timestamp"].equals(fea["timestamp"]) and tar["timestamp"].equals(sif["timestamp"])),
        "target_window": audit.target_window_check(irr, tar),
        "features": audit.feature_definition_check(tar, fea),
        "elevation": audit.elevation_convention_check(tar, cfg),
        "kt": audit.kt_definition_check(tar),
        "rows_with_missing_target_ghi": {h: int(tar[f"ghi_{h}"].isna().sum()) for h in hz},
    }
    # c. clear-sky: our pvlib model against the benchmark's ghi_clear columns
    mk = minute_kt(irr, cfg)
    cs_cmp = {}
    samp = tar.loc[tar["elevation_30min"] >= cfg["clearsky"]["daylight_min_elevation"]].iloc[::5]
    mins = np.arange(-29, 1)
    for h in hz:
        hm = horizon_minutes(h)
        allt = pd.DatetimeIndex((samp["timestamp"].values[:, None]
                                 + pd.to_timedelta(hm + mins, "min").values[None, :]).ravel())
        cs_cmp[h] = {}
        for model in ("ineichen", "haurwitz", "simplified_solis"):
            ours = location(cfg).get_clearsky(allt.tz_localize("UTC"), model=model)["ghi"].values
            d = ours.reshape(len(samp), len(mins)).mean(1) - samp[f"ghi_clear_{h}"].values
            cs_cmp[h][model] = {"rows": int(len(samp)), "mean_diff_Wm2": float(d.mean()),
                                "rms_diff_Wm2": float(np.sqrt(np.mean(d ** 2)))}
    checks["clearsky_ours_minus_benchmark"] = cs_cmp
    day = mk[mk["elevation"] >= cfg["clearsky"]["daylight_min_elevation"]]
    checks["minute_kt_daylight"] = {
        "minutes": int(len(day)), "night_minutes_with_kt": int(mk.loc[mk["elevation"] < cfg["clearsky"]["daylight_min_elevation"], "kt"].notna().sum()),
        "kt_median": float(day["kt"].median()), "kt_p99": float(day["kt"].quantile(0.99)),
        "share_kt_at_cap": float((day["kt"] >= cfg["clearsky"]["max_kt"]).mean()),
        "nonfinite_kt": int((~np.isfinite(day["kt"].dropna())).sum())}
    dump(checks, OUT / "task_checks.json")
    write_task_definition(checks, cfg)

    # d. sky mix
    d = daily_stats(mk, cfg)
    d["sky"] = classify(d, cfg)
    d.to_csv(OUT / "daily_sky.csv")
    mix = monthly_mix(d["sky"], cfg)
    mix.to_csv(OUT / "sky_mix.csv", index=False)
    per_split = (mix.groupby("split", sort=False)[["days", "classified_days"] + CLASSES + ["insufficient"]].sum())
    for k in CLASSES:
        per_split[f"{k}_share"] = per_split[k] / per_split["classified_days"]
    per_split.to_csv(OUT / "sky_mix_by_split.csv")
    plot_sky_mix(mix, cfg)
    # measured / clear-sky GHI on clear days, by elevation band, morning vs afternoon
    m = mk.assign(day=pd.to_datetime((mk["timestamp"] - pd.Timedelta(hours=8)).dt.date))
    m = m[m["day"].isin(d.index[d["sky"] == "clear"]) & m["kt"].notna()]
    m = m.assign(afternoon=(m["timestamp"] - pd.Timedelta(hours=8)).dt.hour >= 12,
                 band=pd.cut(m["elevation"], [5, 10, 15, 20, 30, 45, 90]),
                 ratio=m["ghi"] / m["ghi_clear"])
    (m.groupby(["band", "afternoon"], observed=True)["ratio"].agg(["median", "count"])
      .reset_index().to_csv(OUT / "clearsky_ratio_clear_days.csv", index=False, float_format="%.4f"))

    # e. satellite and NAM
    sat = frames["Folsom_satellite.csv"]
    day_issue = tar.loc[tar["elevation_30min"] >= cfg["clearsky"]["daylight_min_elevation"], "timestamp"]
    irr_t = pd.DatetimeIndex(irr["timestamp"])
    sat_info = {
        "frames": int(len(sat)),
        "pixel_columns": int(sat.shape[1] - 1),
        "pixel_value_range": [float(sat.iloc[:, 1:].min().min()), float(sat.iloc[:, 1:].max().max())],
        "frame_minutes_past_hour_top": {int(k): int(v) for k, v in sat["timestamp"].dt.minute.value_counts().head(8).items()},
        "share_frames_on_irradiance_minute": float(pd.DatetimeIndex(sat["timestamp"]).isin(irr_t).mean()),
        "benchmark_feature": audit.satellite_feature_check(sat, sif, cfg["satellite"]["benchmark_window_minutes"]),
        "availability_at_daylight_issue_times": audit.satellite_availability(
            sat, day_issue.reset_index(drop=True), cfg["satellite"]["availability_lag_minutes"]),
    }
    nam_info = {}
    for p in nam_files(cfg):
        nam = frames[p.name]
        nam_info[p.name] = {"summary": audit.nam_summary(nam),
                            "alignment": audit.nam_alignment(nam, tar_c, cfg["nam"]["availability_lag_hours"],
                                                             cfg["clearsky"]["daylight_min_elevation"])}
    dump({"satellite": sat_info, "nam": nam_info,
          "nam_availability_lag_hours_assumed": cfg["nam"]["availability_lag_hours"],
          "satellite_availability_lag_minutes_assumed": cfg["satellite"]["availability_lag_minutes"]},
         OUT / "satellite_nam.json")
    print("audit written to", OUT)


def write_task_definition(c: dict, cfg: dict) -> None:
    tw = c["target_window"]
    best = {h: min(v, key=lambda k: v[k]["median_abs_diff"]) for h, v in tw.items()}
    el = c["elevation"]
    el_best = {h: min(v, key=lambda k: v[k]["rms_deg"]) for h, v in el.items()}
    f = c["features"]
    lines = [
        "# Intra-day task definition (development data, 2014-2015)",
        "",
        "Generated by `scripts/audit_data.py`; every number below is in `results/audit/task_checks.json`.",
        "",
        "## Horizons and issuing times",
        f"- Horizons, read from the column names of `Target_intra-day.csv`: {', '.join(c['horizons'])}.",
        f"- Issue times are the row time stamps (UTC), on minutes {c['issue_time_minutes_past_hour']} past the hour, "
        f"i.e. every 30 min. Rows exist only for UTC hours {sorted(c['issue_rows_by_utc_hour'])} (daytime in Folsom).",
        f"- Development rows: {c['target_rows']}; after dropping issue times whose furthest target reaches the test "
        f"year: {c['target_rows_after_dropping_targets_in_test_year']}.",
        f"- `Target_intra-day.csv`, `Irradiance_features_intra-day.csv` and `Sat_image_features_intra-day.csv` have "
        f"identical time stamps: {c['same_timestamps_target_features_satellite']}.",
        "",
        "## What a target is",
        "Column `ghi_h` in the row stamped t is the mean of the 1-min GHI of `Folsom_irradiance.csv` over the window "
        "that ends at t+h. Window that matches best per horizon (median |difference| in W/m2, 1500 random rows):",
        "",
        "| horizon | best window | median abs diff | max abs diff | next best window | its median abs diff |",
        "|---|---|---|---|---|---|",
    ]
    for h in c["horizons"]:
        v = tw[h]
        b = best[h]
        nb = sorted(v, key=lambda k: v[k]["median_abs_diff"])[1]
        lines.append(f"| {h} | {b} | {v[b]['median_abs_diff']:.2e} | {v[b]['max_abs_diff']:.2e} | {nb} | "
                     f"{v[nb]['median_abs_diff']:.2f} |")
    lines += [
        "",
        "`elevation_h` is the solar elevation averaged over the 30 minute stamps t+h-29 ... t+h of the same window, "
        f"at the configured site ({cfg['site']['latitude']}, {cfg['site']['longitude']}). RMS difference in degrees:",
        "",
        "| horizon | mean over window minutes | instant t+h-15min | instant t+h |",
        "|---|---|---|---|",
    ]
    for h in c["horizons"]:
        v = el[h]
        lines.append(f"| {h} | {v['mean over minutes t+h-29..t+h']['rms_deg']:.2e} | "
                     f"{v['instant t+h-15min']['rms_deg']:.3f} | {v['instant t+h']['rms_deg']:.3f} |")
    lines += [
        "",
        "`ghi_clear_h` comes from a clear-sky model that is not identified: it does not equal pvlib Ineichen, "
        "Haurwitz or Simplified Solis at the site (see `clearsky_ours_minus_benchmark`). `ghi_kt_h` is capped at 1.2; "
        "it equals min(ghi_h / ghi_clear_h, 1.2) in only part of the rows "
        f"({c['kt']['30min']['share_kt_equals_ratio_of_means']:.2f} at 30min), so its exact construction is unclear "
        "(possibly a mean of 1-min ratios; unverified).",
        "",
        "## How features align with targets",
        f"Checked on {f['rows_compared']} rows. Share of rows where the identity holds to 1e-5:",
        "",
    ]
    for k, v in f.items():
        if k != "rows_compared":
            lines.append(f"- {k}: {v:.3f}")
    lines += [
        "",
        "So `B(ghi_kt|w)` is the mean of the 30-min block clear-sky indices over the w minutes ending at the issue "
        "time t, and `L(ghi_kt|w)` is the block ending at t-(w-30min). Both use only data up to t. `V(ghi_kt|w)` is "
        "a variability measure over the same window, but it is not the standard deviation of the 30-min blocks and "
        "is non-zero for a single block, so it is built from finer data; its exact definition is unclear.",
        "",
        "The benchmark's satellite feature at t is the mean of the raw GOES-15 frames stamped in (t-30min, t] "
        "(see `results/audit/satellite_nam.json`). It contains no frame stamped after t, but it does include a "
        "frame stamped exactly t, whose real availability delay is unclear.",
        "",
        "Smart persistence in `Forecast_intra-day.py`: `B(ghi_kt|30min)` (kt of the last 30-min block) times "
        "`ghi_clear_h`; predictions at `elevation_h` < 5 degrees are set to NaN and excluded from the metrics.",
    ]
    (OUT / "task_definition.md").write_text("\n".join(lines) + "\n")


def plot_sky_mix(mix: pd.DataFrame, cfg: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(mix))
    bottom = np.zeros(len(mix))
    labels = {"clear": "Clear", "partly_cloudy": "Partly cloudy", "overcast": "Overcast"}
    for k in CLASSES:
        color, hatch = SKY_STYLE[k]
        ax.bar(x, mix[k], bottom=bottom, color=color, hatch=hatch, edgecolor="white", linewidth=1.0,
               label=labels[k], width=0.8)
        bottom += mix[k].values
    for split in DEV_SPLITS:
        idx = np.where(mix["split"].values == split)[0]
        if len(idx):
            ax.axvline(idx[0] - 0.5, color="#555555", linewidth=0.8, linestyle="--")
            ax.text(idx.mean(), 32.5, split.replace("_", " "), ha="center", va="bottom", fontsize=9, color="#333333")
    ax.set_xticks(x, mix["month"], rotation=90, fontsize=8)
    ax.set_ylabel("Days")
    ax.set_ylim(0, 35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=9)
    ax.set_title("Daily sky condition by month, development period (UTC-8 days)", fontsize=10, loc="left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"sky_mix.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
