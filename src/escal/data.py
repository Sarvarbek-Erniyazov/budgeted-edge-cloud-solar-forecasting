"""Loaders for the Folsom files. Every loader passes its frame through `dev_only`
immediately after the time stamps are parsed, before any other operation."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from escal.splits import _bounds, dev_only

# file -> how to read it. `time` is the column the split guard filters on.
FILES = {
    "Folsom_irradiance.csv": dict(time="timeStamp"),
    "Folsom_weather.csv": dict(time="timeStamp"),
    "Folsom_sky_image_features.csv": dict(time="timestamp"),
    "Folsom_satellite.csv": dict(time=0, header=None),
    "Irradiance_features_intra-hour.csv": dict(time="timestamp"),
    "Irradiance_features_intra-day.csv": dict(time="timestamp"),
    "Irradiance_features_day-ahead.csv": dict(time="timestamp"),
    "Sky_image_features_intra-hour.csv": dict(time="timestamp"),
    "Sat_image_features_intra-day.csv": dict(time="timestamp"),
    "NAM_nearest_node_day-ahead.csv": dict(time="timestamp"),
    "Target_intra-hour.csv": dict(time="timestamp"),
    "Target_intra-day.csv": dict(time="timestamp"),
    "Target_day-ahead.csv": dict(time="timestamp"),
}
NAM_GLOB = "Folsom_NAM_lat*_lon*.csv"


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def raw_dir(cfg: dict) -> Path:
    return Path(cfg["data"]["raw_dir"])


def nam_files(cfg: dict) -> list[Path]:
    return sorted(raw_dir(cfg).glob(NAM_GLOB))


def _finish(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = dev_only(df, cfg)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df.sort_values("timestamp").reset_index(drop=True)


def read_dev(name: str, cfg: dict) -> pd.DataFrame:
    """Read one benchmark CSV; the time column is renamed to `timestamp`."""
    if re.fullmatch(r"Folsom_NAM_lat.*\.csv", name):
        return read_nam(raw_dir(cfg) / name, cfg)
    spec = FILES[name]
    df = pd.read_csv(raw_dir(cfg) / name, header=spec.get("header", "infer"))
    df = df.rename(columns={spec["time"]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = _finish(df, cfg)
    if name == "Folsom_satellite.csv":
        df.columns = ["timestamp"] + [f"px{i:02d}" for i in range(df.shape[1] - 1)]
    if name == "Sat_image_features_intra-day.csv":
        df.columns = ["timestamp"] + [f"sat{i:02d}" for i in range(df.shape[1] - 1)]
    return df


def read_nam(path: Path, cfg: dict) -> pd.DataFrame:
    """NAM node file: one title line, then reftime, valtime, fields. Filtered on valtime
    (the later of the two), so no forecast valid in the test year is kept."""
    df = pd.read_csv(path, skiprows=1)
    df["reftime"] = pd.to_datetime(df["reftime"])
    df["timestamp"] = pd.to_datetime(df["valtime"])
    df = _finish(df.drop(columns="valtime"), cfg)
    df["lead_h"] = (df["timestamp"] - df["reftime"]).dt.total_seconds() / 3600
    return df


def horizons(target: pd.DataFrame, var: str = "ghi") -> list[str]:
    """Horizon labels as they appear in the target columns, e.g. ['30min', ...]."""
    pat = re.compile(rf"^{var}_(\d+(?:min|h))$")
    return [m.group(1) for c in target.columns if (m := pat.match(c))]


def horizon_minutes(label: str) -> int:
    n = int(re.match(r"\d+", label).group())
    return n * 60 if label.endswith("h") else n


def drop_targets_reaching_test(df: pd.DataFrame, cfg: dict, max_minutes: int) -> pd.DataFrame:
    """Drop issue times whose furthest target falls in the test period."""
    test_start, _ = _bounds(cfg, "test")
    return df.loc[df["timestamp"] + pd.Timedelta(minutes=max_minutes) < test_start].copy()
