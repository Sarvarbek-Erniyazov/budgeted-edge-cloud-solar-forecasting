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
