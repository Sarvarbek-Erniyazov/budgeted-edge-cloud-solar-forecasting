import numpy as np
import pandas as pd
import pvlib
import pytest
import yaml

from escal.clearsky import clearsky_ghi, clearsky_index, solar_elevation


@pytest.fixture
def cfg():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def day():
    # one summer day at 1-min resolution, UTC
    return pd.date_range("2014-06-21 00:00", "2014-06-21 23:59", freq="min")


def _kt(cfg, times, ghi):
    el = solar_elevation(times, cfg).values
    cs = clearsky_ghi(times, cfg).values
    c = cfg["clearsky"]
    return clearsky_index(ghi, cs, el, c["daylight_min_elevation"], c["max_kt"]), el


def test_night_excluded(cfg, day):
    kt, el = _kt(cfg, day, np.full(len(day), 50.0))
    night = el < cfg["clearsky"]["daylight_min_elevation"]
    assert night.any() and (~night).any()
    assert np.isnan(kt[night]).all()
    assert np.isfinite(kt[~night]).all()


def test_index_near_one_on_clear_day(cfg, day):
    # an independent clear-sky model (Haurwitz) stands in for a measured clear day
    s = cfg["site"]
    zen = pvlib.solarposition.get_solarposition(day.tz_localize("UTC"), s["latitude"], s["longitude"])["apparent_zenith"]
    ghi = pvlib.clearsky.haurwitz(zen)["ghi"].values
    kt, el = _kt(cfg, day, ghi)
    high = el > 20
    assert abs(np.nanmedian(kt[high]) - 1.0) < 0.1


def test_no_division_by_zero():
    kt = clearsky_index([0.0, 100.0, 100.0, 5.0], [0.0, 0.0, -1.0, 50.0], [30.0, 30.0, 30.0, 30.0], 5.0, 1.2)
    assert np.isnan(kt[:3]).all()
    assert kt[3] == pytest.approx(0.1)
    assert not np.isinf(kt).any()


def test_cap_applied():
    kt = clearsky_index([500.0], [100.0], [40.0], 5.0, 1.2)
    assert kt[0] == pytest.approx(1.2)
