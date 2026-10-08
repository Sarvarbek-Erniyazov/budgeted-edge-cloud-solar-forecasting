"""Smoke check of the 2015 gate-refit path (development data only, no 2016).

Calls the unchanged run_gates() with the test run's fitting period (gate_fit + validation) on the dry-run
prediction files. Because no test period exists here, evaluation is in-sample on validation; only
completion, the thresholds file and the realised rates on the fitting population are checked.
The outcome is not used to change anything. Writes results/freeze/refit_smoke.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from gates import kt_array, run_gates  # noqa: E402

from escal import gates as G  # noqa: E402
from escal.data import load_config  # noqa: E402
from escal.models import EdgeNet  # noqa: E402
from escal.train import predict  # noqa: E402

PRED = Path("results/dryrun/predictions")
OUT = Path("results/freeze")
THR = OUT / "refit_smoke_thresholds.yaml"
CK = Path("checkpoints/gates_refit_smoke")
FIT = ("gate_fit", "val")


def main() -> None:
    cfg = load_config()
    gc, seeds, budgets = cfg["gates"], cfg["seeds"], cfg["budget"]["targets"]
    run_gates(cfg, pred=PRED, out=OUT / "refit_smoke_gates", fit_parts=FIT, eval_part="val",
              label="insample_val", thresholds_path=THR, ckpt=CK, logs=Path("logs/freeze/refit_smoke"))
    tau = yaml.safe_load(THR.read_text())["gate_thresholds"]

    truth = pd.read_parquet(PRED / "truth.parquet")
    inp = pd.read_parquet(PRED / "on_device_inputs.parquet")
    edge = pd.read_parquet(PRED / "edge.parquet")
    hz = [c[4:] for c in truth.columns if c.startswith("ghi_")]
    day = truth[[f"day_{h}" for h in hz]].values.astype(bool)
    pop = truth["sat_available"].values & day.any(1)
    fit = truth["part"].isin(FIT).values & pop
    dev = cfg["tiers"]["device"] if torch.cuda.is_available() else "cpu"
    rates = {}
    for s in seeds:
        norm = np.load(CK / f"seed{s}" / "norm.npz", allow_pickle=True)
        X = np.concatenate([inp.iloc[:, 1:].values, kt_array(edge, s, hz)], axis=1)
        Xs = np.nan_to_num((X - norm["mu"]) / norm["sd"]).astype(np.float32)
        scores = {"variability": inp[gc["variability_feature"]].fillna(0).values}
        for g in ("uncertainty", "learned"):
            m = EdgeNet(Xs.shape[1], 1, gc["net_hidden"])
            m.load_state_dict(torch.load(CK / f"seed{s}" / f"{g}.pt", map_location=dev))
            m.to(dev)
            scores[g] = predict(m, {"x": Xs, "tiles": None})[:, 0]
        rates[s] = {g: {str(b): float(G.decide(sc[fit], tau[int(s)][g][float(b)]).mean()) for b in budgets}
                    for g, sc in scores.items()}
    dev_max = {g: max(abs(rates[s][g][str(b)] - b) for s in seeds for b in budgets) for g in ("variability", "uncertainty", "learned")}
    meta = json.loads((OUT / "refit_smoke_gates" / "meta.json").read_text())
    out = {"purpose": "smoke check of the 2015 refit path; development data only; outcome not used to change anything",
           "fit_parts": list(FIT), "completed": True, "thresholds_file_written": THR.exists(),
           "thresholds_file": str(THR).replace("\\", "/"),
           "fit_population_issue_times": int(fit.sum()),
           "gate_net_train_rows": meta["gate_net_train_rows"], "gate_net_es_rows": meta["gate_net_es_rows"],
           "es_slice": "last 30 days of the fitting period (December 2015)",
           "realised_rate_on_fitting_population": rates,
           "max_abs_realised_minus_target": dev_max,
           "note": "thresholds are (1-b) quantiles of these same scores, so rates match targets up to ties"}
    (OUT / "refit_smoke.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "realised_rate_on_fitting_population"}, indent=1))


if __name__ == "__main__":
    main()
