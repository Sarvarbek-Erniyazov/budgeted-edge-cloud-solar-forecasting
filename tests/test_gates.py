import numpy as np
import pytest
import yaml

from escal.gates import decide, fit_thresholds, fixed_interval, gate_input_columns, oracle, random_draw

BUDGETS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]


@pytest.fixture
def cfg():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)


def test_budget_accounting_threshold_gate():
    rng = np.random.default_rng(0)
    fit, val = rng.normal(size=5000), rng.normal(size=5000)       # same distribution
    tau = fit_thresholds(fit, BUDGETS)
    for b in BUDGETS:
        assert abs(decide(fit, tau[b]).mean() - b) < 0.002          # exact on the fitting data
        assert abs(decide(val, tau[b]).mean() - b) < 0.03           # close on new data


def test_budget_accounting_exact_gates():
    day = np.repeat(np.arange(50), 22)
    rng = np.random.default_rng(1)
    for b in BUDGETS:
        assert abs(fixed_interval(day, b).mean() - b) < 1 / 22 + 1e-9
        assert oracle(rng.normal(size=1100), b).sum() == round(b * 1100)
        assert random_draw(1100, b, rng).sum() == round(b * 1100)


def test_thresholds_come_from_fit_scores_only():
    rng = np.random.default_rng(2)
    fit = rng.normal(size=1000)
    t1 = fit_thresholds(fit, BUDGETS)
    t2 = fit_thresholds(fit.copy(), BUDGETS)                          # validation scores play no part
    assert t1 == t2
    assert t1[0.25] == pytest.approx(np.quantile(fit, 0.75))


def test_gate_inputs_contain_no_cloud_side_column(cfg):
    forb = cfg["gates"]["forbidden_input_substrings"]
    ok = ["B(ghi_kt|30min)", "ih_V(ghi_kt|30min)", "wx_air_temp", "cs_30min", "edge_kt_30min"]
    assert gate_input_columns(ok, forb) == ok
    for bad in ["sat00", "nam0_dwsw_30min", "cloud_kt_30min", "sif"]:
        with pytest.raises(ValueError):
            gate_input_columns(ok + [bad], forb)
