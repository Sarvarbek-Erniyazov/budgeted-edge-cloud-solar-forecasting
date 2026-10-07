import numpy as np
import torch

from escal.models import CloudNet, EdgeNet
from escal.train import fit, predict, set_seed

CFG = {"tiers": {"device": "cpu", "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 32, "max_epochs": 5,
                 "patience": 2}}


def _data(tiles=False):
    rng = np.random.default_rng(0)
    d = {"x": rng.normal(size=(200, 8)).astype(np.float32), "kt": rng.uniform(size=(200, 6)).astype(np.float32),
         "mask": np.ones((200, 6), np.float32), "tiles": None}
    if tiles:
        d["tiles"] = rng.uniform(size=(200, 2, 10, 10)).astype(np.float32)
    return d


def _run(make, d, tmp_path, k):
    tr = np.arange(200) < 150
    set_seed(3)
    m = make()
    fit(m, d, tr, ~tr, CFG, seed=3, log_path=tmp_path / f"log{k}.json")
    return predict(m, d)


def test_same_seed_same_result(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = _data()
    a = _run(lambda: EdgeNet(8, 6, [16]), d, tmp_path, 0)
    b = _run(lambda: EdgeNet(8, 6, [16]), d, tmp_path, 1)
    assert np.array_equal(a, b)
    dc = _data(tiles=True)
    c1 = _run(lambda: CloudNet(8, 6, 2, [16], [4], 8), dc, tmp_path, 2)
    c2 = _run(lambda: CloudNet(8, 6, 2, [16], [4], 8), dc, tmp_path, 3)
    assert np.array_equal(c1, c2)
