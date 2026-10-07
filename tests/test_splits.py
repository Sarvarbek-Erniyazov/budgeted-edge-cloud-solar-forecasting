import pandas as pd
import pytest
import yaml

from escal.splits import LockedTestYear, check_no_overlap, dev_only, select_split


@pytest.fixture
def cfg():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def df():
    ts = pd.date_range("2014-01-01", "2016-12-31 23:00", freq="h")
    return pd.DataFrame({"timestamp": ts, "x": range(len(ts))})


def test_splits_ordered_and_disjoint(cfg, df):
    check_no_overlap(cfg)
    parts = [select_split(df, cfg, n) for n in ["models_train", "gate_fit", "validation"]]
    assert sum(len(p) for p in parts) == len(dev_only(df, cfg))
    assert parts[0]["timestamp"].max() < parts[1]["timestamp"].min() < parts[2]["timestamp"].min()


def test_dev_only_has_no_test_rows(cfg, df):
    assert dev_only(df, cfg)["timestamp"].max() < pd.Timestamp("2016-01-01")


def test_test_year_locked_by_default(cfg, df, monkeypatch):
    monkeypatch.delenv("ESCAL_UNLOCK_TEST", raising=False)
    with pytest.raises(LockedTestYear):
        select_split(df, cfg, "test")


def test_env_alone_does_not_unlock(cfg, df, monkeypatch):
    monkeypatch.setenv("ESCAL_UNLOCK_TEST", "1")
    assert cfg["protocol"]["frozen"] is False
    with pytest.raises(LockedTestYear):
        select_split(df, cfg, "test")


def test_unlock_needs_both(cfg, df, monkeypatch):
    monkeypatch.setenv("ESCAL_UNLOCK_TEST", "1")
    cfg["protocol"]["frozen"] = True
    out = select_split(df, cfg, "test")
    assert out["timestamp"].min() == pd.Timestamp("2016-01-01")
