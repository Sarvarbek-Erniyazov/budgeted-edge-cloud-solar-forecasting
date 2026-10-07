"""Chronological split and the test-year guard.

Every loader must pass its frame through `dev_only` (development) or
`select_split`. The test period can be selected only when the config says the
protocol is frozen AND the environment variable ESCAL_UNLOCK_TEST=1 is set.
"""
from __future__ import annotations

import os

import pandas as pd


class LockedTestYear(RuntimeError):
    pass


def _bounds(cfg: dict, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = cfg["split"][name]
    # end date is inclusive: up to the last instant of that day
    return pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)


def test_unlocked(cfg: dict) -> bool:
    return bool(cfg.get("protocol", {}).get("frozen")) and os.environ.get("ESCAL_UNLOCK_TEST") == "1"


def dev_only(df: pd.DataFrame, cfg: dict, time_col: str = "timestamp") -> pd.DataFrame:
    """Drop every row at or after the start of the test period."""
    test_start, _ = _bounds(cfg, "test")
    return df.loc[df[time_col] < test_start].copy()


def select_split(df: pd.DataFrame, cfg: dict, name: str, time_col: str = "timestamp") -> pd.DataFrame:
    if name == "test" and not test_unlocked(cfg):
        raise LockedTestYear(
            "The test year is locked. Set protocol.frozen: true in the config and "
            "ESCAL_UNLOCK_TEST=1 only after the protocol freeze."
        )
    start, end = _bounds(cfg, name)
    return df.loc[(df[time_col] >= start) & (df[time_col] < end)].copy()


def check_no_overlap(cfg: dict) -> None:
    names = ["models_train", "gate_fit", "validation", "test"]
    b = [_bounds(cfg, n) for n in names]
    for (s0, e0), (s1, _), n0 in zip(b, b[1:], names):
        if not (s0 < e0 <= s1):
            raise ValueError(f"split '{n0}' overlaps or is out of order")
