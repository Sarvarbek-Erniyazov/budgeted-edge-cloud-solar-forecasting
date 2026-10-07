"""The single test run (docs/FREEZE.md section c) and its dry run.

  python scripts/test_run.py predictions [--dry-run]
  python scripts/test_run.py gates [--dry-run]
  python scripts/test_run.py claims [--dry-run]

Test mode needs protocol.frozen: true and ESCAL_UNLOCK_TEST=1 (otherwise LockedTestYear is raised before
anything is read). It evaluates on 2016, refits the gates on all of 2015, and writes results/test/.
Dry-run mode never touches 2016: validation plays the "test" period, gates are fitted on gate_fit only,
output goes to results/dryrun/, and `claims --dry-run` ends with an exact comparison against the committed
development results (results/dryrun/comparison.json).
Tiers are never retrained: edge and step-3 network from checkpoints, trees from checkpoints/trees/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gates import run_gates  # noqa: E402
from tiers import CKPT, Run, tile_summary  # noqa: E402

from escal import gates as G  # noqa: E402
from escal.audit import minute_coverage  # noqa: E402
from escal.benchmark import anchor_predict  # noqa: E402
from escal.bootstrap import paired_diff  # noqa: E402
from escal.data import horizon_minutes, load_config, read  # noqa: E402
from escal.features import apply_standardiser, fit_standardiser  # noqa: E402
from escal.models import CloudNet, EdgeNet  # noqa: E402
from escal.splits import LockedTestYear, _bounds, test_unlocked  # noqa: E402
from escal.train import predict, set_seed  # noqa: E402

TREES = Path("checkpoints/trees")
MODES = {
    "dry": dict(include_test=False, eval_part="val", eval_split="validation", fit_parts=("gate_fit",),
                label="validation", out=Path("results/dryrun"), gate_ckpt=Path("checkpoints/gates_dryrun"),
                thresholds=Path("results/dryrun/gate_thresholds.yaml")),
    "test": dict(include_test=True, eval_part="test", eval_split="test", fit_parts=("gate_fit", "val"),
                 label="test", out=Path("results/test"), gate_ckpt=Path("checkpoints/gates_test"),
                 thresholds=Path("configs/gate_thresholds_test.yaml")),
}
TIER_FILES = ["edge", "edge_int8", "cloud", "step3_network", "trees_all", "trees_ground", "trees_ground_cap10x"]
REPORT_GATES = ["random", "fixed_interval", "variability", "uncertainty", "learned", "oracle"]


def _jsonable(o):
    return o.tolist() if hasattr(o, "tolist") else float(o)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


# ---------------------------------------------------------------- predictions
def data_quality(run: Run, cfg: dict, m: dict, nam: pd.DataFrame) -> dict:
    ev = (run.df["part"] == m["eval_part"]).values
    day = run.tg["day"]
    t = run.df.loc[ev, "timestamp"]
    s, e = _bounds(cfg, m["eval_split"])
    all_days = pd.date_range(s, e - pd.Timedelta(days=1), freq="D")
    local_days = pd.DatetimeIndex((t - pd.Timedelta(hours=8)).dt.normalize().unique())
    steps = t.diff().dropna()
    big = steps.sort_values(ascending=False).head(5)
    irr = read("Folsom_irradiance.csv", cfg, m["include_test"])
    irr = irr[(irr["timestamp"] >= s) & (irr["timestamp"] < e)]
    dv = ev & day.any(1)
    nam_rows = {}
    for j, h in enumerate(run.hz):
        cells = ev & day[:, j]
        nam_rows[h] = {"daylight_cells": int(cells.sum()),
                       "share_missing": float(nam[f"nam_missing_{h}"].values[cells].mean()),
                       "share_interpolated": float(nam[f"nam_interp_{h}"].values[cells].mean())}
    sif_ok = run.df[run.base.sif_cols].notna().all(axis=1).values
    return {
        "period": m["eval_split"], "bounds": [str(s), str(e)],
        "issue_times": int(ev.sum()), "daylight_issue_times": int(dv.sum()),
        "daylight_cells": int((ev[:, None] & day).sum()),
        "days_in_period": int(len(all_days)), "local_days_with_issue_times": int(len(local_days)),
        "largest_gaps_between_issue_times": [{"after": str(t.loc[i - 1]) if i - 1 in t.index else None,
                                              "length": str(v)} for i, v in big.items()],
        "irradiance_minutes": {**minute_coverage(irr), "first": str(irr["timestamp"].min()),
                               "last": str(irr["timestamp"].max())} if len(irr) else {"rows": 0},
        "satellite_available_15min_share_of_daylight_issue_times": float(run.avail[dv].mean()),
        "satellite_available_0min_share_of_daylight_issue_times": float(run.avail0[dv].mean()),
        "benchmark_satellite_feature_share_of_daylight_issue_times": float(sif_ok[dv].mean()),
        "primary_cells": int((ev[:, None] & day & run.avail[:, None]).sum()),
        "nam_coverage": nam_rows,
    }


def stage_predictions(cfg: dict, m: dict) -> None:
    out, pred = m["out"], m["out"] / "predictions"
    pred.mkdir(parents=True, exist_ok=True)
    run = Run(cfg, include_test=m["include_test"])
    tc, hz, seeds = run.tc, run.hz, cfg["seeds"]
    v3 = {**tc["base_variant"], **cfg["ladder"]["step3"]}
    d3 = run.cloud_data(v3, run.lag)

    # data quality first, before any prediction or metric
    dq = data_quality(run, cfg, m, d3["nam_raw"])
    (out / "data_quality.json").write_text(json.dumps(dq, indent=2))
    print("data quality written", flush=True)

    norms = np.load(TREES / "norms.npz", allow_pickle=True)
    g_mu = pd.Series(norms["ground_mu"], index=norms["ground_cols"])
    g_sd = pd.Series(norms["ground_sd"], index=norms["ground_cols"])
    t_mu = pd.Series(norms["step3_mu"], index=norms["step3_cols"])
    t_sd = pd.Series(norms["step3_sd"], index=norms["step3_cols"])
    rg_mu, rg_sd = fit_standardiser(run.ground, run.tr)
    rt_mu, rt_sd = fit_standardiser(d3["tab"], run.tr)
    norm_check = {"ground_stats_recomputed_equal_saved": bool(np.array_equal(rg_mu.values, g_mu.values)
                                                               and np.array_equal(rg_sd.values, g_sd.values)),
                  "step3_stats_recomputed_equal_saved": bool(np.array_equal(rt_mu.values, t_mu.values)
                                                              and np.array_equal(rt_sd.values, t_sd.values))}
    Xg = apply_standardiser(run.ground, g_mu, g_sd)
    X3 = apply_standardiser(d3["tab"], t_mu, t_sd)
    clip = tc["kt_clip"]
    dev = tc["device"] if torch.cuda.is_available() else "cpu"
    arrs = {k: {} for k in TIER_FILES}
    for s in seeds:
        set_seed(s)
        e = EdgeNet(Xg.shape[1], len(hz), tc["edge"]["hidden"])
        e.load_state_dict(torch.load(CKPT / "base" / f"edge_seed{s}.pt", map_location=dev))
        e.to(dev)
        arrs["edge"][s] = np.clip(predict(e, {"x": Xg, "tiles": None}), *clip)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(f"checkpoints/footprint/edge_seed{s}.int8.onnx", so, providers=["CPUExecutionProvider"])
        arrs["edge_int8"][s] = np.clip(np.concatenate([sess.run(None, {"x": Xg[k:k + 4096]})[0]
                                                       for k in range(0, len(Xg), 4096)]), *clip)
        set_seed(s)
        c = tc["cloud"]
        net = CloudNet(X3.shape[1], len(hz), d3["tiles"].shape[1], c["hidden"], c["cnn_channels"], c["fusion_hidden"])
        net.load_state_dict(torch.load(CKPT / "step3" / f"cloud_seed{s}.pt", map_location=dev))
        net.to(dev)
        arrs["step3_network"][s] = np.clip(predict(net, {"x": X3, "tiles": d3["tiles"]}), *clip)
    Xall = np.concatenate([X3, tile_summary(d3["tiles"])], axis=1)
    for name, X in (("trees_all", Xall), ("trees_ground", Xg), ("trees_ground_cap10x", Xg)):
        models = pickle.loads((TREES / f"{name}.pkl").read_bytes())
        o = np.zeros((len(X), len(hz)), np.float32)
        for j, mdl in enumerate(models):
            o[:, j] = mdl.predict(X)
        arrs[name]["all"] = np.clip(o, *clip)
    arrs["cloud"] = {s: (arrs["trees_all"]["all"] + arrs["step3_network"][s]) / 2 for s in seeds}

    # anchors, fitted on models_train only exactly as in Stage 3 / Stage 5
    ms, me = _bounds(cfg, "models_train")
    max_m = max(horizon_minutes(h) for h in hz)
    fit_rows = ((run.df["timestamp"] >= ms) & (run.df["timestamp"] + pd.Timedelta(minutes=max_m) < me)).values
    endo = [c for c in run.base.endo_cols if "ghi" in c]
    allf = endo + run.base.sif_cols
    lasso = {"lasso_endo": np.stack([anchor_predict(run.df, fit_rows, endo, allf, "ghi", h, cfg) for h in hz], 1),
             "lasso_exo": np.stack([anchor_predict(run.df, fit_rows, allf, allf, "ghi", h, cfg) for h in hz], 1)}

    keep = run.df["part"].isin(list(m["fit_parts"]) + [m["eval_part"]]).values
    ts = run.df.loc[keep, "timestamp"].reset_index(drop=True)
    tg = run.tg
    truth = pd.DataFrame({"timestamp": ts, "part": run.df.loc[keep, "part"].values,
                          "sat_available": run.avail[keep],
                          "B(ghi_kt|30min)": run.df.loc[keep, "B(ghi_kt|30min)"].values})
    for j, h in enumerate(hz):
        truth[f"ghi_{h}"] = tg["ghi"][keep, j]
        truth[f"kt_{h}"] = run.df.loc[keep, f"ghi_kt_{h}"].values
        truth[f"clear_{h}"] = tg["clear"][keep, j]
        truth[f"sp_{h}"] = tg["sp"][keep, j]
        truth[f"day_{h}"] = tg["day"][keep, j]
    truth.to_parquet(pred / "truth.parquet", index=False)
    devin = run.ground.loc[keep].reset_index(drop=True)
    devin.insert(0, "timestamp", ts)
    devin.to_parquet(pred / "on_device_inputs.parquet", index=False)
    for name, d in arrs.items():
        frames = []
        for s, a in d.items():
            f = pd.DataFrame(a[keep], columns=[f"kt_{h}" for h in hz])
            f.insert(0, "seed", s)
            f.insert(0, "timestamp", ts)
            frames.append(f)
        pd.concat(frames, ignore_index=True).to_parquet(pred / f"{name}.parquet", index=False)
    an = pd.DataFrame({"timestamp": ts})
    for k, a in lasso.items():
        for j, h in enumerate(hz):
            an[f"{k}_{h}"] = a[keep, j]
    an.to_parquet(pred / "anchors_wm2.parquet", index=False)
    meta = {"mode": "dry run" if m is MODES["dry"] else "test", "git_head": git_head(), **norm_check,
            "rows_saved": int(keep.sum()), "parts": truth["part"].value_counts().to_dict()}
    (out / "predictions_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


# ---------------------------------------------------------------- gates
def stage_gates(cfg: dict, m: dict) -> None:
    run_gates(cfg, pred=m["out"] / "predictions", out=m["out"] / "gates", fit_parts=m["fit_parts"],
              eval_part=m["eval_part"], label=m["label"], thresholds_path=m["thresholds"], ckpt=m["gate_ckpt"],
              logs=Path("logs") / ("dryrun" if m is MODES["dry"] else "test") / "gates")


# ---------------------------------------------------------------- claims
def stage_claims(cfg: dict, m: dict) -> dict:
    pred, out, label = m["out"] / "predictions", m["out"], m["label"]
    truth = pd.read_parquet(pred / "truth.parquet")
    hz = [c[4:] for c in truth.columns if c.startswith("ghi_")]
    seeds = cfg["seeds"]
    y = truth[[f"ghi_{h}" for h in hz]].values
    clear = truth[[f"clear_{h}" for h in hz]].values
    sp = truth[[f"sp_{h}" for h in hz]].values
    day = truth[[f"day_{h}" for h in hz]].values.astype(bool)
    ev = (truth["part"] == m["eval_part"]).values
    pop = truth["sat_available"].values & day.any(1)
    prim = day & (ev & pop)[:, None]
    alld = day & ev[:, None]
    kt = truth[[f"kt_{h}" for h in hz]].values
    prev = np.column_stack([truth["B(ghi_kt|30min)"].values, kt[:, :-1]])
    ramp = prim & (np.abs(kt - prev) >= cfg["ramp"]["delta_kt"])
    ts = truth["timestamp"]

    def W(name, seed):
        d = pd.read_parquet(pred / f"{name}.parquet")
        d = d[d["seed"].astype(str) == str(seed)]
        return np.where(day, d[[f"kt_{h}" for h in hz]].values * clear, np.nan)

    E = {s: W("edge", s) for s in seeds}
    Ei = {s: W("edge_int8", s) for s in seeds}
    C = {s: W("cloud", s) for s in seeds}
    TG, TA, CAP = W("trees_ground", "all"), W("trees_all", "all"), W("trees_ground_cap10x", "all")
    an = pd.read_parquet(pred / "anchors_wm2.parquet")
    LE = an[[f"lasso_endo_{h}" for h in hz]].values
    LX = an[[f"lasso_exo_{h}" for h in hz]].values
    anch = prim & np.isfinite(LX)

    def R(p, sel):
        return G.avg_metrics(y, p, sp, sel)["RMSE"]

    # model table: every model on identical cells per row set
    rows = []
    sets = {"primary": prim, "all_daylight": alld, "ramp": ramp, "primary_anchor_rows": anch}
    for rs, sel in sets.items():
        models = {"smart_persistence|na": sp, "lasso_endo|na": LE, "trees_ground|na": TG, "trees_all|na": TA,
                  "trees_ground_cap10x|na": CAP,
                  **{f"edge_fp32|{s}": E[s] for s in seeds}, **{f"edge_int8 (secondary)|{s}": Ei[s] for s in seeds},
                  **{f"cloud|{s}": C[s] for s in seeds}}
        if rs == "primary_anchor_rows":
            models["lasso_exo|na"] = LX
        for k, p in models.items():
            name, seed = k.split("|")
            mt = G.avg_metrics(y, p, sp, sel)
            rows.append({"row_set": rs, "model": name, "seed": seed, "cells": int(sel.sum()),
                         "RMSE": mt["RMSE"], "MAE": mt["MAE"], "skill": mt["skill"],
                         **{f"rmse_{h}": v for h, v in zip(hz, mt["rmse_per_horizon"])}})
    table = pd.DataFrame(rows)
    table.to_csv(out / f"claims_models_{label}.csv", index=False, float_format="%.4f")

    c = {"mode": "dry run (validation as test, gates fitted on gate_fit)" if m is MODES["dry"] else "test",
         "git_head": git_head(), "primary_cells": int(prim.sum()), "anchor_cells": int(anch.sum()),
         "ramp_cells": int(ramp.sum())}
    # claim 1
    ra, rg = R(TA, prim), R(TG, prim)
    iv = paired_diff(y, TG, TA, prim, ts, cfg)
    eff = 1 - ra / rg
    c["claim1_input_effect"] = {"rmse_trees_all": ra, "rmse_trees_ground": rg, "input_effect": eff,
                                "interval_rmse_ground_minus_all": iv,
                                "counts_against": bool(eff >= 0.02 and iv["lo"] > 0)}
    # claim 2
    rc = {s: R(C[s], prim) for s in seeds}
    rcap = R(CAP, prim)
    ivs = [{"cloud_seed": s, **paired_diff(y, CAP, C[s], prim, ts, cfg)} for s in seeds]
    above = sum(i["lo"] > 0 for i in ivs)
    d = float(np.mean([rcap - rc[s] for s in seeds]) / np.mean(list(rc.values())))
    verdict2 = "worse" if above >= 3 else ("matches" if abs(d) <= 0.02 else "not distinguishable")
    c["claim2_on_device_sufficiency"] = {"rmse_cap10x": rcap, "rmse_cloud_per_seed": rc, "relative_difference_d": d,
                                         "intervals_rmse_cap10x_minus_cloud": ivs, "seeds_entirely_above_zero": int(above),
                                         "verdict": verdict2, "counts_against": verdict2 == "worse"}
    # claim 3 (fp32 primary, int8 secondary)
    for tag, EE in (("fp32", E), ("int8_secondary", Ei)):
        re = {s: R(EE[s], prim) for s in seeds}
        gains = [1 - rc[s] / re[s] for s in seeds]
        ivs3 = [{"seed": s, **paired_diff(y, EE[s], C[s], prim, ts, cfg)} for s in seeds]
        incl = sum(i["lo"] <= 0 for i in ivs3)
        c[f"claim3_edge_vs_cloud_{tag}"] = {
            "rmse_edge_per_seed": re, "gain_per_seed": gains, "gain_mean_of_seed_rmse": float(1 - np.mean(list(rc.values())) / np.mean(list(re.values()))),
            "gain_median_seed": float(np.median(gains)), "intervals_rmse_edge_minus_cloud": ivs3,
            "seeds_including_zero_or_below": int(incl), "below_5pct_threshold": bool(np.mean(gains) < 0.05),
            "counts_against": bool(incl >= 3) if tag == "fp32" else "not judged (secondary row)"}
    # claim 4
    summ = pd.read_csv(out / "gates" / f"summary_{label}.csv")
    sweep = pd.read_csv(out / "gates" / f"sweep_{label}.csv")
    rb = cfg["budget"]["report_at"]
    sc = summ[(summ.escalate_to == "cloud") & (summ.row_set == "primary") & summ.budget_target.isin(rb)]
    c4 = {"table": sc[["gate", "budget_target", "realised_rate_mean", "share_mean", "share_median", "share_min",
                       "share_max"]].to_dict("records")}
    sw = sweep[(sweep.escalate_to == "cloud") & (sweep.row_set == "primary")]

    def share(g, b, s):
        return float(sw[(sw.gate == g) & (sw.budget_target == b) & (sw.seed == s)]["share_retained"].iloc[0])
    diff_ur = [share("uncertainty", 0.25, s) - share("random", 0.25, s) for s in seeds]
    c4["uncertainty_minus_random_share_at_25pct_per_seed"] = diff_ur
    c4["against_gate_beats_random"] = bool(sum(x < 0.10 for x in diff_ur) >= 3)

    def smean(g, b):
        return float(sc[(sc.gate == g) & (sc.budget_target == b)]["share_mean"].iloc[0])
    lu = {b: smean("learned", b) - smean("uncertainty", b) for b in rb}
    c4["learned_minus_uncertainty_seed_mean"] = lu
    c4["against_learned_not_better"] = bool(sum(v >= 0.10 for v in lu.values()) >= 2)
    dev_rate = {f"{g}@{b}": float(sc[(sc.gate == g) & (sc.budget_target == b)]["realised_rate_mean"].iloc[0] - b)
                for g in ("variability", "uncertainty", "learned") for b in rb}
    c4["realised_minus_target"] = dev_rate
    c4["against_refit_fixes_shortfall"] = bool(any(abs(v) > 0.05 for v in dev_rate.values()))
    c["claim4_gates"] = c4
    # claim 5
    re_r = {s: R(E[s], ramp) for s in seeds}
    rc_r = {s: R(C[s], ramp) for s in seeds}
    iv5 = [{"seed": s, **paired_diff(y, E[s], C[s], ramp, ts, cfg)} for s in seeds]
    swr = sweep[(sweep.escalate_to == "cloud") & (sweep.row_set == "ramp") & (sweep.budget_target == 0.25)]
    gate_ramp = {}
    for g in [x for x in REPORT_GATES if x != "oracle"]:
        v = swr[swr.gate == g]["share_retained"].values
        gate_ramp[g] = {"share_mean": float(v.mean()), "all_seeds_positive": bool((v > 0).all())}
    c["claim5_ramp"] = {"rmse_edge_ramp": re_r, "rmse_cloud_ramp": rc_r, "intervals_rmse_edge_minus_cloud_ramp": iv5,
                        "against_no_cloud_advantage": bool(sum(i["lo"] > 0 for i in iv5) >= 3),
                        "gate_share_at_25pct_ramp": gate_ramp,
                        "against_no_gate_helps": bool(any(v["share_mean"] >= 0.5 and v["all_seeds_positive"]
                                                          for v in gate_ramp.values())),
                        "note": "oracle excluded (headroom only)"}
    (out / f"claims_{label}.json").write_text(json.dumps(c, indent=2, default=_jsonable))
    print("claims written", flush=True)
    if m is MODES["dry"]:
        compare(cfg, m, c, table)
    return c


# ---------------------------------------------------------------- dry-run comparison
def compare(cfg: dict, m: dict, c: dict, table: pd.DataFrame) -> None:
    items = []

    def add(name, ref, got, kind="exact"):
        if kind == "bytes":
            same = ref == got
            items.append({"item": name, "kind": "byte-identical file", "identical": bool(same)})
            return
        ref_a, got_a = np.asarray(ref, float), np.asarray(got, float)
        diff = float(np.nanmax(np.abs(ref_a - got_a))) if ref_a.size else 0.0
        items.append({"item": name, "kind": kind, "reference": ref, "dry_run": got, "max_abs_diff": diff,
                      "identical": bool(diff == 0.0 and np.array_equal(np.isnan(ref_a), np.isnan(got_a)))})

    out, hz, seeds = m["out"], None, cfg["seeds"]
    # prediction files
    ref_p, dry_p = Path("results/tiers/predictions"), out / "predictions"
    for name in ("edge", "cloud", "trees_ground"):
        a, b = pd.read_parquet(ref_p / f"{name}.parquet"), pd.read_parquet(dry_p / f"{name}.parquet")
        cols = [x for x in a.columns if x.startswith("kt_")]
        same = a["timestamp"].equals(b["timestamp"]) and np.array_equal(a[cols].values, b[cols].values)
        items.append({"item": f"predictions/{name}.parquet values", "kind": "exact array equality", "identical": bool(same),
                      "max_abs_diff": float(np.abs(a[cols].values - b[cols].values).max())})
    for name in ("truth", "on_device_inputs"):
        if (ref_p / f"{name}.parquet").exists():
            a, b = pd.read_parquet(ref_p / f"{name}.parquet"), pd.read_parquet(dry_p / f"{name}.parquet")
            same = a.equals(b[a.columns])
            items.append({"item": f"predictions/{name}.parquet (local, not committed)", "kind": "exact frame equality",
                          "identical": bool(same)})
    # gate outputs and thresholds
    for f in ("sweep_validation.csv", "summary_validation.csv", "bootstrap_validation.csv",
              "edge_vs_cloud_bootstrap.json", "meta.json"):
        add(f"gates/{f}", Path("results/gates", f).read_bytes(), (out / "gates" / f).read_bytes(), "bytes")
    add("gate_thresholds.yaml vs configs/gate_thresholds.yaml", Path("configs/gate_thresholds.yaml").read_bytes(),
        m["thresholds"].read_bytes(), "bytes")
    for s in seeds:
        for g in ("uncertainty", "learned"):
            a = torch.load(Path("checkpoints/gates") / f"seed{s}" / f"{g}.pt")
            b = torch.load(m["gate_ckpt"] / f"seed{s}" / f"{g}.pt")
            items.append({"item": f"gate network {g} seed {s}", "kind": "exact tensor equality",
                          "identical": bool(all(torch.equal(a[k], b[k]) for k in a))})
    # claim values
    ce = json.loads(Path("results/tiers/control_effects.json").read_text())["primary"]
    add("claim 1 input effect (control_effects.json)", ce["input_effect (A = trees_all, B = trees_ground)"]["relative"]["mean"],
        c["claim1_input_effect"]["input_effect"])
    fp = json.loads(Path("results/tiers/footprint_trees.json").read_text())
    add("claim 2 RMSE cap10x (footprint_trees.json)", fp["validation_rmse_seed_mean"]["trees_ground_cap10x"]["mean"],
        c["claim2_on_device_sufficiency"]["rmse_cap10x"])
    add("claim 2 cloud RMSE seed mean (footprint_trees.json)", fp["validation_rmse_seed_mean"]["cloud_step4_average"]["mean"],
        float(np.mean(list(c["claim2_on_device_sufficiency"]["rmse_cloud_per_seed"].values()))))
    for k in ("point", "lo", "hi"):
        add(f"claim 2 interval {k} per seed (footprint_trees.json)",
            [x[k] for x in fp["match_vs_cloud"]["trees_ground_cap10x"]["rmse_trees_minus_cloud"]],
            [x[k] for x in c["claim2_on_device_sufficiency"]["intervals_rmse_cap10x_minus_cloud"]])
    gg = json.loads(Path("results/go_no_go.json").read_text())
    v4 = [v for v in gg["variants"] if v["name"] == "step4_gbt_plus_net"][0]
    add("claim 3 gain (go_no_go.json step4_gbt_plus_net)", v4["gain_avg"], c["claim3_edge_vs_cloud_fp32"]["gain_mean_of_seed_rmse"])
    items[-1]["note"] = ("Stage 5 averaged trees and network in W/m2 (float64); the saved prediction files, the gates and "
                         "this run average kt in float32, then multiply by ghi_clear_h. See the next item.")
    # the same gain with the Stage 5 convention: average of trees and network formed in W/m2, float64
    pred = out / "predictions"
    truth = pd.read_parquet(pred / "truth.parquet")
    hz = [x[4:] for x in truth.columns if x.startswith("ghi_")]
    yy = truth[[f"ghi_{h}" for h in hz]].values
    cl = truth[[f"clear_{h}" for h in hz]].values
    dy = truth[[f"day_{h}" for h in hz]].values.astype(bool)
    sl = dy & ((truth["part"] == m["eval_part"]).values & truth["sat_available"].values & dy.any(1))[:, None]

    def ktp(name, s):
        d = pd.read_parquet(pred / f"{name}.parquet")
        return d[d["seed"].astype(str) == str(s)][[f"kt_{h}" for h in hz]].values

    def w(k):
        return np.where(dy, k * cl, np.nan)

    def rr(p):
        return [np.sqrt(np.mean((yy[sl[:, j], j] - p[sl[:, j], j]) ** 2)) for j in range(len(hz))]
    ta = w(ktp("trees_all", "all"))
    e5 = np.array([rr(w(ktp("edge", s))) for s in seeds])
    c5 = np.array([rr((ta + w(ktp("step3_network", s))) / 2) for s in seeds])
    add("claim 3 gain, Stage 5 convention (W/m2 float64 average) vs go_no_go.json", v4["gain_avg"],
        float(1 - c5.mean(1).mean() / e5.mean(1).mean()))
    add("claim 3 gain per seed (go_no_go.json, stored to 4 dp)", v4["gain_avg_per_seed"],
        [round(x, 4) for x in c["claim3_edge_vs_cloud_fp32"]["gain_per_seed"]])
    eb = json.loads(Path("results/gates/edge_vs_cloud_bootstrap.json").read_text())
    for k in ("point", "lo", "hi"):
        add(f"claim 3 interval {k} per seed (gates/edge_vs_cloud_bootstrap.json)",
            [x["edge_minus_cloud"][k] for x in eb], [x[k] for x in c["claim3_edge_vs_cloud_fp32"]["intervals_rmse_edge_minus_cloud"]])
    add("edge fp32 RMSE per seed (gates/edge_vs_cloud_bootstrap.json)", [x["rmse"]["edge"] for x in eb],
        [c["claim3_edge_vs_cloud_fp32"]["rmse_edge_per_seed"][s] for s in seeds])
    add("cloud RMSE per seed (gates/edge_vs_cloud_bootstrap.json)", [x["rmse"]["cloud"] for x in eb],
        [c["claim2_on_device_sufficiency"]["rmse_cloud_per_seed"][s] for s in seeds])
    ms = json.loads(Path("results/footprint/measured.json").read_text())["edge_network"]["per_seed"]
    add("edge int8 RMSE per seed (measured.json)", [x["rmse_int8"] for x in ms],
        [c["claim3_edge_vs_cloud_int8_secondary"]["rmse_edge_per_seed"][s] for s in seeds])
    summ = pd.read_csv("results/gates/summary_validation.csv")
    sc = summ[(summ.escalate_to == "cloud") & (summ.row_set == "primary") & summ.budget_target.isin(cfg["budget"]["report_at"])]
    got = pd.DataFrame(c["claim4_gates"]["table"])
    add("claim 4 share_mean at report budgets (summary_validation.csv)", sc["share_mean"].values, got["share_mean"].values)
    add("claim 4 realised_rate_mean at report budgets", sc["realised_rate_mean"].values, got["realised_rate_mean"].values)
    sw = pd.read_csv("results/gates/sweep_validation.csv")
    rr = sw[(sw.escalate_to == "cloud") & (sw.row_set == "ramp") & (sw.gate == "random")]
    add("claim 5 ramp RMSE edge per seed (sweep, budget 0, 4 dp)", rr[rr.budget_target == 0.0]["RMSE"].values,
        np.round([c["claim5_ramp"]["rmse_edge_ramp"][s] for s in seeds], 4))
    add("claim 5 ramp RMSE cloud per seed (sweep, budget 1, 4 dp)", rr[rr.budget_target == 1.0]["RMSE"].values,
        np.round([c["claim5_ramp"]["rmse_cloud_ramp"][s] for s in seeds], 4))
    vc = pd.read_csv("results/tiers/validation.csv")
    base = vc[(vc.variant == "base") & (vc.horizon == "mean")]

    def tab(model, rs):
        return float(table[(table.model == model) & (table.row_set == rs)]["RMSE"].iloc[0])
    add("smart persistence RMSE, primary (tiers/validation.csv, 4 dp)",
        float(base[(base.row_set == "primary") & (base.model == "smart_persistence")]["RMSE"].iloc[0]), round(tab("smart_persistence", "primary"), 4))
    add("lasso_endo RMSE, primary (tiers/validation.csv, 4 dp)",
        float(base[(base.row_set == "primary") & (base.model == "lasso_endo")]["RMSE"].iloc[0]), round(tab("lasso_endo", "primary"), 4))
    add("lasso_exo RMSE, anchor rows (tiers/validation.csv, 4 dp)",
        float(base[(base.row_set == "primary_anchor_rows") & (base.model == "lasso_exo")]["RMSE"].iloc[0]),
        round(tab("lasso_exo", "primary_anchor_rows"), 4))
    add("anchor-row cells (tiers/validation.csv mean-row n)",
        int(base[(base.row_set == "primary_anchor_rows") & (base.model == "lasso_exo")]["n"].iloc[0]), int(c["anchor_cells"]))
    res = {"all_identical": bool(all(i["identical"] for i in items)), "n_items": len(items),
           "n_identical": int(sum(i["identical"] for i in items)), "items": items}
    (out / "comparison.json").write_text(json.dumps(res, indent=2, default=_jsonable))
    print(f"comparison: {res['n_identical']}/{res['n_items']} identical", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["predictions", "gates", "claims"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    m = MODES["dry" if a.dry_run else "test"]
    if not a.dry_run and not test_unlocked(cfg):
        raise LockedTestYear("test run refused: protocol.frozen must be true and ESCAL_UNLOCK_TEST=1 (docs/FREEZE.md)")
    m["out"].mkdir(parents=True, exist_ok=True)
    {"predictions": stage_predictions, "gates": stage_gates, "claims": stage_claims}[a.stage](cfg, m)


if __name__ == "__main__":
    main()
