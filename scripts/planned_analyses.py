"""Stage 9.1: planned analyses A1 to A7 (docs/PLAN.md section 3) on the 2016 test run.
Reads results/test/ and the saved checkpoints only; nothing is retrained, retuned or rerun.
Writes results/analyses/a1_... to a7_... and results/analyses/planned_summary.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analysis_common import Test, load_cfgs, verify_decisions  # noqa: E402

from escal import gates as G  # noqa: E402
from escal.bootstrap import avg_rmse, day_index, day_sums, interval, weights  # noqa: E402


def rmse_avg_expected(y, edge, target, esc_prob, sel):
    """RMSE (per horizon, averaged) when escalated cells take target with probability esc_prob, else edge."""
    se = (y - edge) ** 2
    st = (y - target) ** 2
    m = esc_prob[:, None] * st + (1 - esc_prob[:, None]) * se
    return float(np.mean([np.sqrt(np.mean(m[sel[:, j], j])) for j in range(y.shape[1])]))


def main() -> None:
    cfg, acfg = load_cfgs()
    out = Path(acfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    T = Test(cfg, acfg)
    seeds, hz = T.seeds, T.hz
    scores = {s: T.gate_scores(s) for s in seeds}
    check = verify_decisions(T, scores)
    print("decision check", check, flush=True)
    if not check["reproduces_sweep"]:
        raise SystemExit("recomputed gate decisions do not reproduce sweep_test.csv; stopping")
    tau = T.thresholds()
    v = np.nonzero(T.ev)[0]
    yv, spv, selv, rampv = T.y[v], T.sp[v], T.sel[v], T.ramp[v]
    Ev = {s: T.W("edge", s)[v] for s in seeds}
    Cv = {s: T.W("cloud", s)[v] for s in seeds}
    summ = pd.read_csv(Path(acfg["gates_dir"]) / "summary_test.csv")
    sweep = pd.read_csv(Path(acfg["gates_dir"]) / "sweep_test.csv")
    res = {"decision_check": check}

    # A1 gate-to-oracle gap per budget
    s = summ[(summ.escalate_to == "cloud") & (summ.row_set == "primary")]
    piv = s.pivot(index="budget_target", columns="gate", values="RMSE_mean")
    rows = []
    for b in piv.index:
        for g in ("fixed_interval", "variability", "uncertainty", "learned"):
            r_or, r_rd, r_g = piv.loc[b, "oracle"], piv.loc[b, "random"], piv.loc[b, g]
            rows.append({"budget_target": b, "gate": g, "rmse_gate": r_g, "rmse_oracle": r_or, "rmse_random": r_rd,
                         "gap_to_oracle_Wm2": r_g - r_or,
                         "share_of_random_to_oracle_gap_closed": (r_rd - r_g) / (r_rd - r_or) if r_rd != r_or else np.nan})
    a1 = pd.DataFrame(rows)
    a1.to_csv(out / "a1_gate_oracle_gap.csv", index=False, float_format="%.4f")

    # A2 per-horizon breakdown
    cm = pd.read_csv("results/test/claims_models_test.csv")
    cm = cm[cm.row_set == "primary"]
    tiers = cm.groupby("model", sort=False)[[f"rmse_{h}" for h in hz]].mean()
    sw = sweep[(sweep.escalate_to == "cloud") & (sweep.row_set == "primary") & (sweep.budget_target == 0.25)]
    rows = []
    for j, h in enumerate(hz):
        re_, rc_ = tiers.loc["edge_fp32", f"rmse_{h}"], tiers.loc["cloud", f"rmse_{h}"]
        r = {"horizon": h, "edge_fp32": re_, "cloud": rc_, "trees_ground": tiers.loc["trees_ground", f"rmse_{h}"],
             "smart_persistence": tiers.loc["smart_persistence", f"rmse_{h}"], "edge_to_cloud_gain": 1 - rc_ / re_}
        for g in ("random", "variability", "uncertainty", "learned", "oracle"):
            gg = sw[sw.gate == g].groupby("seed")[f"rmse_{h}"].mean()
            ee = [G.avg_metrics(yv[:, [j]], Ev[x][:, [j]], spv[:, [j]], selv[:, [j]])["RMSE"] for x in seeds]
            cc = [G.avg_metrics(yv[:, [j]], Cv[x][:, [j]], spv[:, [j]], selv[:, [j]])["RMSE"] for x in seeds]
            r[f"rmse_{g}_25pct"] = float(gg.mean())
            r[f"share_kept_{g}_25pct"] = float(np.mean([(e - gv) / (e - c) for e, gv, c in zip(ee, gg.values, cc)]))
        rows.append(r)
    a2 = pd.DataFrame(rows)
    a2.to_csv(out / "a2_per_horizon.csv", index=False, float_format="%.4f")

    # A3 routing maps (uncertainty gate at 25% by default)
    rg, rb = acfg["routing"]["gate"], acfg["routing"]["budget"]
    esc_share = np.mean([T.decisions(rg, rb, s, scores[s], tau).astype(float) for s in seeds], axis=0)
    loc = T.local.iloc[v].reset_index(drop=True)
    df = pd.DataFrame({"month": loc.dt.month, "hour": loc.dt.hour, "esc": esc_share})
    hm = df.groupby(["month", "hour"]).agg(share_escalated=("esc", "mean"), issue_times=("esc", "size")).reset_index()
    hm.to_csv(out / "a3_routing_hour_month.csv", index=False, float_format="%.4f")
    sk = acfg["sky_regime"]
    b30 = T.truth["B(ghi_kt|30min)"].values
    dl = (T.local.dt.normalize()).values
    dd = pd.DataFrame({"day": dl, "kt": b30, "ok": T.day[:, 0] & (T.truth["part"] == "test").values})
    dd = dd[dd.ok]
    stats = dd.groupby("day")["kt"].agg(mean_kt="mean", var=lambda x: float(np.sqrt(np.mean(np.diff(x.values) ** 2))) if len(x) > 2 else np.nan)
    low = stats["var"] < sk["low_variability"]
    stats["regime"] = np.where(low & (stats.mean_kt >= sk["clear_min_mean_kt"]), "clear",
                               np.where(low & (stats.mean_kt < sk["overcast_max_mean_kt"]), "overcast", "partly_cloudy"))
    df["regime"] = pd.Series(dl[v]).map(stats["regime"]).values
    reg = df.groupby("regime").agg(share_escalated=("esc", "mean"), issue_times=("esc", "size")).reset_index()
    reg["days"] = reg["regime"].map(stats["regime"].value_counts())
    # benefit actually available per regime (oracle view): mean per-issue-time squared-error reduction, seed mean
    ben = np.mean([G.benefit(yv, Ev[s], Cv[s], selv) for s in seeds], axis=0)
    reg["mean_true_benefit_Wm2sq"] = reg["regime"].map(pd.Series(ben).groupby(df["regime"].values).mean())
    reg.to_csv(out / "a3_routing_sky_regime.csv", index=False, float_format="%.4f")

    # A4 link outage
    oc = acfg["outage"]
    rng = np.random.default_rng(oc["seed"])
    dv = day_index(T.ts.iloc[v])
    rows = []
    for s in seeds:
        re_, rc_ = (G.avg_metrics(yv, Ev[s], spv, selv)["RMSE"], G.avg_metrics(yv, Cv[s], spv, selv)["RMSE"])
        for g in oc["gates"]:
            for b in oc["budgets"]:
                esc = np.full(len(v), b) if g == "random" else T.decisions(g, b, s, scores[s], tau).astype(float)
                for q in oc["fractions"]:
                    r_ind = rmse_avg_expected(yv, Ev[s], Cv[s], esc * (1 - q), selv)
                    blk = []
                    for _ in range(oc["block_draws"] if 0 < q < 1 else 1):
                        down = rng.random(dv.max() + 1) < q
                        blk.append(rmse_avg_expected(yv, Ev[s], Cv[s], esc * (~down[dv]), selv))
                    rb_ = float(np.mean(blk))
                    rows.append({"seed": s, "gate": g, "budget_target": b, "outage_fraction": q,
                                 "rmse_independent_outages": r_ind, "rmse_day_block_outages": rb_,
                                 "share_kept_independent": (re_ - r_ind) / (re_ - rc_),
                                 "share_kept_day_block": (re_ - rb_) / (re_ - rc_)})
    a4 = pd.DataFrame(rows)
    a4.to_csv(out / "a4_link_outage.csv", index=False, float_format="%.4f")
    a4s = a4.groupby(["gate", "budget_target", "outage_fraction"])[["share_kept_independent", "share_kept_day_block"]].mean().reset_index()
    a4s.to_csv(out / "a4_link_outage_summary.csv", index=False, float_format="%.4f")

    # A5 paired day-block bootstrap of gate differences (RMSE_A - RMSE_B), report budgets
    w = weights(dv.max() + 1, cfg)
    rows = []
    for s in seeds:
        se = day_sums(yv, Ev[s], selv, dv, dv.max() + 1)
        sc = day_sums(yv, Cv[s], selv, dv, dv.max() + 1)

        def sums(g, b):
            if g == "random":
                return (b * sc[0] + (1 - b) * se[0], se[1])
            return day_sums(yv, G.blend(Ev[s], Cv[s], T.decisions(g, b, s, scores[s], tau)), selv, dv, dv.max() + 1)
        for b in cfg["budget"]["report_at"]:
            for ga, gb in acfg["gate_pairs"]:
                sa, sb = sums(ga, b), sums(gb, b)
                it = interval(avg_rmse(*sa) - avg_rmse(*sb), avg_rmse(*sa, w) - avg_rmse(*sb, w), cfg)
                rows.append({"seed": s, "budget_target": b, "gate_a": ga, "gate_b": gb, **it})
    a5 = pd.DataFrame(rows)
    a5.to_csv(out / "a5_gate_differences.csv", index=False, float_format="%.4f")
    a5s = (a5.groupby(["budget_target", "gate_a", "gate_b"])
           .agg(point_mean=("point", "mean"), lo_mean=("lo", "mean"), hi_mean=("hi", "mean"),
                seeds_excluding_zero=("includes_zero", lambda x: int((~x).sum()))).reset_index())
    a5s.to_csv(out / "a5_gate_differences_summary.csv", index=False, float_format="%.4f")

    # A6 input ablation (what the frozen run contains) next to the validation 4-way control
    cl = json.loads(Path("results/test/claims_test.json").read_text())
    tm = cm.groupby("model", sort=False)["RMSE"].agg(["mean", "min", "max"])
    ctl = pd.read_csv("results/tiers/control.csv")
    ctl = ctl[(ctl.row_set == "primary") & (ctl.horizon == "mean")].groupby("model")["RMSE"].mean()
    a6 = pd.DataFrame([
        {"model": "trees_ground (edge inputs)", "rmse_2016": tm.loc["trees_ground", "mean"], "rmse_validation": ctl["trees_ground"]},
        {"model": "trees ground+satellite", "rmse_2016": np.nan, "rmse_validation": ctl["trees_ground_sat"]},
        {"model": "trees ground+NAM", "rmse_2016": np.nan, "rmse_validation": ctl["trees_ground_nam"]},
        {"model": "trees all inputs", "rmse_2016": tm.loc["trees_all", "mean"], "rmse_validation": ctl["trees_all"]},
        {"model": "cloud tier (step-4 average)", "rmse_2016": tm.loc["cloud", "mean"], "rmse_validation": ctl["step4_average"]},
        {"model": "edge network fp32", "rmse_2016": tm.loc["edge_fp32", "mean"], "rmse_validation": ctl["edge_network"]},
    ])
    a6.to_csv(out / "a6_input_ablation.csv", index=False, float_format="%.4f")
    res["a6_note"] = ("The planned A6 (cloud tier without satellite, without NWP) needs two extra cloud trainings and "
                      "was not done: nothing is retrained after the freeze. Ground+satellite and ground+NAM trees were "
                      "not part of the frozen test run, so they have validation values only.")
    res["a6_input_effect_2016"] = cl["claim1_input_effect"]

    # A7 calibration of the uncertainty gate (predicted vs realised mean |edge error| per issue time)
    nb = acfg["calibration_bins"]
    rows, rho = [], {}
    for s in seeds:
        pred = scores[s]["uncertainty"][v] * cfg["gates"]["uncertainty_scale"]
        act = np.nanmean(np.where(selv, np.abs(yv - Ev[s]), np.nan), axis=1)
        ok = np.isfinite(act)
        q = pd.qcut(pd.Series(pred[ok]).rank(method="first"), nb, labels=False)
        for k in range(nb):
            m = (q == k).values
            rows.append({"seed": s, "bin": k, "n": int(m.sum()), "mean_predicted_Wm2": float(pred[ok][m].mean()),
                         "mean_realised_Wm2": float(act[ok][m].mean())})
        rho[s] = float(pd.Series(pred[ok]).corr(pd.Series(act[ok]), method="spearman"))
        slope = np.polyfit(pred[ok], act[ok], 1)
        rho[f"{s}_fit"] = {"slope": float(slope[0]), "intercept": float(slope[1])}
    pd.DataFrame(rows).to_csv(out / "a7_uncertainty_calibration.csv", index=False, float_format="%.4f")
    res["a7_spearman_and_fit"] = rho

    res["files"] = sorted(p.name for p in out.glob("a*_*.csv"))
    (out / "planned_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "a6_input_effect_2016"}, indent=1, default=float))


if __name__ == "__main__":
    main()
