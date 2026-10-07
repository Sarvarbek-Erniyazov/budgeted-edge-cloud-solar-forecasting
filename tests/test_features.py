import numpy as np
import pandas as pd

from escal.features import Base, sat_available, satellite


def _base(issue, frames):
    sat_t = pd.DatetimeIndex(frames).values
    px = np.stack([np.full((10, 10), float(i + 1)) for i in range(len(frames))]).astype(np.float32)
    return Base(df=pd.DataFrame({"timestamp": pd.DatetimeIndex(issue)}), hz=[], ground_cols=[], endo_cols=[],
                sif_cols=[], sat_t=sat_t, sat_px=px, nams={})


def test_no_frame_newer_than_lag():
    b = _base(["2015-07-01 18:00"], ["2015-07-01 17:30", "2015-07-01 17:45", "2015-07-01 18:00"])
    tiles, meta, stamps = satellite(b, lag_min=15, n_frames=2, lookback_min=60, diffs=False)
    assert stamps[0, 0] == np.datetime64("2015-07-01T17:45")       # 18:00 frame excluded
    assert stamps[0, 1] == np.datetime64("2015-07-01T17:30")
    assert (stamps[~np.isnat(stamps)] <= np.datetime64("2015-07-01T17:45")).all()
    assert tiles[0, 0, 0, 0] == 2 / 255.0
    assert meta[0, 0] == 0.25                                       # age 15 min, in hours


def test_max_age_and_missing():
    b = _base(["2015-07-01 18:00", "2015-07-01 18:30"], ["2015-07-01 16:30"])
    assert not sat_available(b, 15, 60).any()                       # 90 min old: unavailable
    tiles, meta, _ = satellite(b, 15, 1, 60, False)
    assert (tiles == 0).all() and (meta[:, 1] == 0).all()


def test_differences():
    b = _base(["2015-07-01 18:00"], ["2015-07-01 17:15", "2015-07-01 17:30", "2015-07-01 17:45"])
    tiles, _, _ = satellite(b, 15, 3, 120, diffs=True)
    assert tiles.shape[1] == 5
    assert np.allclose(tiles[0, 3], (3 - 2) / 255.0)
