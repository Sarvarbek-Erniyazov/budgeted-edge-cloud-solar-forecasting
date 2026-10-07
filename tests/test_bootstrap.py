import numpy as np
import pandas as pd

from escal.bootstrap import day_index, paired_diff

CFG = {"bootstrap": {"n_resamples": 300, "seed": 0, "ci": 0.95}}


def _setup():
    ts = pd.date_range("2015-07-01 16:00", periods=40 * 20, freq="30min")
    rng = np.random.default_rng(1)
    y = rng.normal(500, 100, size=(len(ts), 2))
    return ts, y, np.ones_like(y, dtype=bool)


def test_identical_models_give_zero_interval():
    ts, y, sel = _setup()
    p = y + 10
    r = paired_diff(y, p, p.copy(), sel, ts, CFG)
    assert r["point"] == 0 and r["lo"] == 0 and r["hi"] == 0 and r["includes_zero"]


def test_clearly_better_model_excludes_zero():
    ts, y, sel = _setup()
    rng = np.random.default_rng(2)
    good, bad = y + rng.normal(0, 10, y.shape), y + rng.normal(0, 50, y.shape)
    r = paired_diff(y, bad, good, sel, ts, CFG)
    assert r["lo"] > 0 and not r["includes_zero"]


def test_day_blocks_are_local_days():
    d = day_index(pd.to_datetime(["2015-07-01 16:00", "2015-07-02 02:00", "2015-07-02 09:00"]))
    assert d[0] == d[1] != d[2]
