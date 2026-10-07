"""The test-period loader path. Uses synthetic files only; no real 2016 data is read."""
import copy

import pandas as pd
import pytest
import yaml

from escal.data import read, read_dev
from escal.features import load_base
from escal.splits import LockedTestYear


@pytest.fixture
def cfg(tmp_path):
    with open("configs/base.yaml") as f:
        c = yaml.safe_load(f)
    c = copy.deepcopy(c)
    c["data"]["raw_dir"] = str(tmp_path)
    ts = pd.date_range("2015-12-30 16:00", "2016-01-02 02:00", freq="30min")
    pd.DataFrame({"timestamp": ts, "ghi_30min": range(len(ts))}).to_csv(tmp_path / "Target_intra-day.csv", index=False)
    nam = tmp_path / "Folsom_NAM_lat1_lon2.csv"
    nam.write_text("12Z NAM forecast for Folsom\nreftime,valtime,dwsw\n"
                   "2015-12-30 12:00:00,2015-12-31 14:00:00,1\n2015-12-31 12:00:00,2016-01-01 14:00:00,2\n")
    return c


def test_test_path_locked_while_not_frozen(cfg, monkeypatch):
    monkeypatch.delenv("ESCAL_UNLOCK_TEST", raising=False)
    assert cfg["protocol"]["frozen"] is False
    with pytest.raises(LockedTestYear):
        read("Target_intra-day.csv", cfg, include_test=True)
    with pytest.raises(LockedTestYear):
        read("Folsom_NAM_lat1_lon2.csv", cfg, include_test=True)
    with pytest.raises(LockedTestYear):
        load_base(cfg, include_test=True)


def test_env_alone_does_not_unlock_test_path(cfg, monkeypatch):
    monkeypatch.setenv("ESCAL_UNLOCK_TEST", "1")
    with pytest.raises(LockedTestYear):
        read("Target_intra-day.csv", cfg, include_test=True)


def test_default_path_never_returns_test_rows(cfg, monkeypatch):
    monkeypatch.setenv("ESCAL_UNLOCK_TEST", "1")
    cfg["protocol"]["frozen"] = True                     # even when unlocked, the default path stays dev-only
    df = read_dev("Target_intra-day.csv", cfg)
    assert df["timestamp"].max() < pd.Timestamp("2016-01-01")
    assert read_dev("Folsom_NAM_lat1_lon2.csv", cfg)["timestamp"].max() < pd.Timestamp("2016-01-01")


def test_unlocked_test_path_adds_test_rows(cfg, monkeypatch):
    monkeypatch.setenv("ESCAL_UNLOCK_TEST", "1")
    cfg["protocol"]["frozen"] = True
    df = read("Target_intra-day.csv", cfg, include_test=True)
    assert df["timestamp"].min() < pd.Timestamp("2016-01-01") <= df["timestamp"].max()
    assert df["timestamp"].is_monotonic_increasing and not df["timestamp"].duplicated().any()


def test_test_run_refuses_without_unlock(monkeypatch):
    """scripts/test_run.py in test mode raises before reading any data."""
    import sys
    monkeypatch.delenv("ESCAL_UNLOCK_TEST", raising=False)
    sys.path.insert(0, "scripts")
    import test_run
    monkeypatch.setattr(sys, "argv", ["test_run.py", "predictions"])
    with pytest.raises(LockedTestYear):
        test_run.main()
