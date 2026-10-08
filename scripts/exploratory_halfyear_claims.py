"""EXPLORATORY (not a planned analysis): claims 1 to 3 computed separately for January-June and July-December
2016, to examine why validation (July-December 2015) and test disagree. The split was chosen AFTER seeing
the test results. Reads results/test/ only. Writes results/analyses/exploratory_halfyear_claims.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analysis_common import Test, load_cfgs  # noqa: E402

from escal import gates as G  # noqa: E402
from escal.bootstrap import paired_diff  # noqa: E402


def main() -> None:
    cfg, acfg = load_cfgs()
    T = Test(cfg, acfg)
    seeds = T.seeds
    E = {s: T.W("edge", s) for s in seeds}
    C = {s: T.W("cloud", s) for s in seeds}
    TG, TA, CAP = T.W("trees_ground", "all"), T.W("trees_all", "all"), T.W("trees_ground_cap10x", "all")

    def R(p, sel):
        return G.avg_metrics(T.y, p, T.sp, sel)["RMSE"]

    out = {"label": "EXPLORATORY. Not a planned analysis (docs/FREEZE.md section c). The January-June / "
                    "July-December split was chosen after seeing the 2016 results. Frozen verdict rules applied "
                    "per half for description only; they do not replace the full-year verdicts.",
           "halves": {}}
    for name, (a, b) in acfg["exploratory_halves"].items():
        inh = ((T.ts >= pd.Timestamp(a)) & (T.ts < pd.Timestamp(b) + pd.Timedelta(days=1))).values
        sel = T.sel & inh[:, None]
        h = {"bounds": [a, b], "primary_cells": int(sel.sum()), "issue_times": int((T.ev & inh).sum()),
             "satellite_available_share_of_daylight_issue_times":
                 float(T.truth["sat_available"].values[(T.truth["part"] == "test").values & T.day.any(1) & inh].mean())}
        ra, rg = R(TA, sel), R(TG, sel)
        iv1 = paired_diff(T.y, TG, TA, sel, T.ts, cfg)
        h["claim1"] = {"rmse_trees_all": ra, "rmse_trees_ground": rg, "input_effect": 1 - ra / rg,
                       "interval_rmse_ground_minus_all": iv1,
                       "would_count_against": bool(1 - ra / rg >= 0.02 and iv1["lo"] > 0)}
        rc = {s: R(C[s], sel) for s in seeds}
        rcap = R(CAP, sel)
        iv2 = [{"cloud_seed": s, **paired_diff(T.y, CAP, C[s], sel, T.ts, cfg)} for s in seeds]
        above = sum(i["lo"] > 0 for i in iv2)
        d = float(np.mean([rcap - rc[s] for s in seeds]) / np.mean(list(rc.values())))
        h["claim2"] = {"rmse_cap10x": rcap, "rmse_cloud_seed_mean": float(np.mean(list(rc.values()))),
                       "relative_difference_d": d, "seeds_entirely_above_zero": int(above),
                       "intervals": iv2,
                       "verdict_rule_applied": "worse" if above >= 3 else ("matches" if abs(d) <= 0.02 else "not distinguishable")}
        re_ = {s: R(E[s], sel) for s in seeds}
        iv3 = [{"seed": s, **paired_diff(T.y, E[s], C[s], sel, T.ts, cfg)} for s in seeds]
        h["claim3"] = {"rmse_edge_seed_mean": float(np.mean(list(re_.values()))),
                       "rmse_cloud_seed_mean": float(np.mean(list(rc.values()))),
                       "gain_mean_of_seed_rmse": float(1 - np.mean(list(rc.values())) / np.mean(list(re_.values()))),
                       "gain_per_seed": [1 - rc[s] / re_[s] for s in seeds],
                       "seeds_including_zero_or_below": int(sum(i["lo"] <= 0 for i in iv3)), "intervals": iv3}
        out["halves"][name] = h
    Path(acfg["out_dir"], "exploratory_halfyear_claims.json").write_text(json.dumps(out, indent=2, default=float))
    for n, h in out["halves"].items():
        print(n, h["primary_cells"], "C1", round(h["claim1"]["input_effect"], 4),
              {k: round(h["claim1"]["interval_rmse_ground_minus_all"][k], 2) for k in ("point", "lo", "hi")},
              "| C2 d", round(h["claim2"]["relative_difference_d"], 4), h["claim2"]["seeds_entirely_above_zero"],
              h["claim2"]["verdict_rule_applied"],
              "| C3 gain", round(h["claim3"]["gain_mean_of_seed_rmse"], 4), h["claim3"]["seeds_including_zero_or_below"])


if __name__ == "__main__":
    main()
