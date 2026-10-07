import numpy as np
import pandas as pd

from escal.nwp import nam_at


def _runs():
    rows = []
    for k, ref in enumerate(pd.to_datetime(["2015-03-01 12:00", "2015-03-02 12:00"])):
        for lead in range(26, 41):
            rows.append({"reftime": ref, "timestamp": ref + pd.Timedelta(hours=lead), "x": 100 * k + lead})
    return pd.DataFrame(rows)


def test_run_not_used_before_available():
    nam = _runs()
    v = pd.Timestamp("2015-03-03 15:00")                  # covered by both runs? only by 03-02 12Z (lead 27h)
    early = nam_at(nam, [pd.Timestamp("2015-03-02 13:00")], [v], lag_h=6, fields=["x"])
    late = nam_at(nam, [pd.Timestamp("2015-03-02 18:00")], [v], lag_h=6, fields=["x"])
    assert np.isnan(early["x"].iloc[0])                   # 03-02 12Z not yet available at 13:00
    assert late["x"].iloc[0] == 127
    assert late["reftime"].iloc[0] == pd.Timestamp("2015-03-02 12:00")


def test_newest_covering_run_and_interpolation():
    nam = _runs()
    t = pd.Timestamp("2015-03-02 20:00")
    v = pd.Timestamp("2015-03-02 21:30")                 # only the 03-01 12Z run covers it (leads 33h, 34h)
    out = nam_at(nam, [t], [v], lag_h=6, fields=["x"])
    assert out["reftime"].iloc[0] == pd.Timestamp("2015-03-01 12:00")
    assert out["x"].iloc[0] == 33.5
    assert out["lead_h"].iloc[0] == 33.5
    assert out["reftime"].iloc[0] + pd.Timedelta(hours=6) <= t


def test_interpolates_within_one_run_and_flags():
    # one run with a 3-hourly tail: valid times at leads 36 h and 39 h only
    ref = pd.Timestamp("2015-03-01 12:00")
    nam = pd.DataFrame({"reftime": ref, "timestamp": [ref + pd.Timedelta(hours=36), ref + pd.Timedelta(hours=39)],
                        "x": [0.0, 30.0]})
    t = pd.Timestamp("2015-03-02 23:00")
    v = ref + pd.Timedelta(hours=37)
    strict = nam_at(nam, [t], [v], lag_h=6, fields=["x"])                  # hourly bracketing only
    loose = nam_at(nam, [t], [v], lag_h=6, fields=["x"], max_gap_h=3)
    assert np.isnan(strict["x"].iloc[0])
    assert loose["x"].iloc[0] == 10.0
    assert loose["interpolated"].iloc[0] == 1.0


def test_never_mixes_runs():
    # run A covers 14:00, run B (newer) covers 16:00 only: v=15:00 must not blend A and B
    a, b = pd.Timestamp("2015-03-01 12:00"), pd.Timestamp("2015-03-02 12:00")
    nam = pd.DataFrame({"reftime": [a, b], "timestamp": [pd.Timestamp("2015-03-03 14:00"),
                                                         pd.Timestamp("2015-03-03 16:00")], "x": [1.0, 2.0]})
    out = nam_at(nam, [pd.Timestamp("2015-03-03 12:00")], [pd.Timestamp("2015-03-03 15:00")], 6, ["x"], max_gap_h=3)
    assert np.isnan(out["x"].iloc[0])
