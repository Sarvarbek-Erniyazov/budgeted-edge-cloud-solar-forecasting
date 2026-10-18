"""Post-hoc analyses A8 to A11 on the 2016 test run (not pre-specified; asked for after the paper draft).
Reads results/test/ and the saved gate checkpoints only; nothing is retrained, retuned or rerun, and nothing under
results/test/ is written. Gate scores are not stored as files: as in planned_analyses.py they are recomputed by
inference with the saved 2015-refit gate networks, and the decisions must reproduce sweep_test.csv before anything
is written. Every output file starts with the line HEADER. Writes results/analyses/posthoc/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analysis_common import Test, load_cfgs, verify_decisions  # noqa: E402

from escal import gates as G  # noqa: E402

HEADER = "post hoc, not pre-specified"
OUT = Path("results/analyses/posthoc")
BUDGET = 0.25
KT_COL = "ih_B(ghi_kt|5min)"                 # mean GHI clear-sky index over the 5 min before the issue time
KT_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, np.inf)]


def write(df: pd.DataFrame, stem: str, title: str, notes: list[str]) -> None:
    with open(OUT / f"{stem}.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {HEADER}\n")
        df.to_csv(f, index=False, float_format="%.4f")
    cols = list(df.columns)
    fmt = lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)  # noqa: E731
    md = [HEADER, "", f"# {title}", ""] + notes + ["", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    (OUT / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def seed_stats(vals) -> dict:
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def main() -> None:
    cfg, acfg = load_cfgs()
    OUT.mkdir(parents=True, exist_ok=True)
    gd = Path(acfg["gates_dir"])
    sweep = pd.read_csv(gd / "sweep_test.csv")
    sweep = sweep[(sweep.escalate_to == "cloud") & (sweep.row_set == "primary")]
    summ = pd.read_csv(gd / "summary_test.csv")
    cm = pd.read_csv("results/test/claims_models_test.csv")
    cm = cm[cm.row_set == "primary"]
    hz = [c[5:] for c in sweep.columns if c.startswith("rmse_")]
    res = {"header": HEADER}

    # A8 per-horizon RMSE, seed mean and range (smart persistence has no seed)
    series = {"smart_persistence": cm[cm.model == "smart_persistence"],
              "edge_fp32": cm[cm.model == "edge_fp32"], "cloud": cm[cm.model == "cloud"],
              "uncertainty_25pct": sweep[(sweep.gate == "uncertainty") & (sweep.budget_target == BUDGET)],
              "learned_25pct": sweep[(sweep.gate == "learned") & (sweep.budget_target == BUDGET)]}
    rows = []
    for h in hz:
        r = {"horizon": h}
        for name, d in series.items():
            st = seed_stats(d[f"rmse_{h}"].values)
            r[f"{name}_mean"] = st["mean"]
            if name != "smart_persistence":
                r[f"{name}_min"], r[f"{name}_max"] = st["min"], st["max"]
        rows.append(r)
    a8 = pd.DataFrame(rows)
    write(a8, "a8_per_horizon_rmse", "A8: per-horizon RMSE on 2016 (W/m2)", [
        "Primary cells (daylight, satellite frame available), 39,655 cells over 7,457 issue times. "
        "Seed mean and seed range (min, max) over 5 seeds; smart persistence has no seed.",
        "Sources: results/test/claims_models_test.csv (smart persistence, edge_fp32, cloud) and "
        "results/test/gates/sweep_test.csv (escalate_to = cloud, row_set = primary, budget_target = 0.25)."])

    # A9 uncertainty gate @25% by clear-sky-index bin at issue time
    T = Test(cfg, acfg)
    scores = {s: T.gate_scores(s) for s in T.seeds}
    check = verify_decisions(T, scores)
    print("decision check", check, flush=True)
    if not check["reproduces_sweep"]:
        raise SystemExit("recomputed gate decisions do not reproduce sweep_test.csv; stopping")
    res["decision_check"] = check
    tau = T.thresholds()
    v = np.nonzero(T.ev)[0]
    yv, selv = T.y[v], T.sel[v]
    kt0 = T.inp[KT_COL].values[v]
    if np.isnan(kt0).any():
        raise SystemExit(f"{KT_COL} missing at {int(np.isnan(kt0).sum())} evaluation issue times")
    per_seed = []
    for s in T.seeds:
        esc = T.decisions("uncertainty", BUDGET, s, scores[s], tau)
        E, C = T.W("edge", s)[v], T.W("cloud", s)[v]
        for lo, hi in KT_BINS:
            m = (kt0 >= lo) & ((kt0 <= hi) if hi == 1.0 else (kt0 < hi))
            sel = selv & m[:, None]
            ok = [j for j in range(len(T.hz)) if sel[:, j].any()]
            rmse = lambda P: float(np.mean([np.sqrt(np.mean((yv[sel[:, j], j] - P[sel[:, j], j]) ** 2)) for j in ok]))  # noqa: E731
            per_seed.append({"seed": s, "bin": lo, "issue_times": int(m.sum()), "cells": int(sel.sum()),
                             "share_escalated": float(esc[m].mean()), "share_of_escalations": float(esc[m].sum() / esc.sum()),
                             "rmse_edge": rmse(E), "rmse_cloud": rmse(C)})
    ps = pd.DataFrame(per_seed)
    ps["cloud_minus_edge"] = ps.rmse_cloud - ps.rmse_edge
    rows = []
    for lo, hi in KT_BINS:
        d = ps[ps.bin == lo]
        lab = f"{lo:.1f}-{hi:.1f}" if np.isfinite(hi) else f">{lo:.1f}"
        r = {"kt_bin": lab, "issue_times": int(d.issue_times.iloc[0]), "cells": int(d.cells.iloc[0])}
        for c in ("share_escalated", "share_of_escalations", "rmse_edge", "rmse_cloud", "cloud_minus_edge"):
            st = seed_stats(d[c])
            r[f"{c}_mean"], r[f"{c}_min"], r[f"{c}_max"] = st["mean"], st["min"], st["max"]
        rows.append(r)
    a9 = pd.DataFrame(rows)
    write(a9, "a9_escalation_by_kt", "A9: uncertainty gate @25% by clear-sky index at issue time (2016)", [
        f"Clear-sky index at issue time = `{KT_COL}` (mean GHI kt over the 5 min before the issue time, an on-device "
        "input; capped at 1.2). Bins are [lo, hi) except 0.8-1.0, which includes 1.0; >1.0 is strictly above 1.0.",
        "share_escalated: escalated issue times / issue times in the bin. share_of_escalations: escalated issue times "
        "in the bin / all escalated issue times. RMSE over the bin's daylight cells, per horizon then averaged over "
        "horizons that have cells; cloud_minus_edge < 0 means the cloud tier is better in that bin. Seed mean and range "
        "over 5 seeds. Gate decisions recomputed by inference with the saved gate networks and thresholds and checked "
        "to reproduce results/test/gates/sweep_test.csv exactly (4 dp)."])

    # A10 target vs realised escalation rate
    s10 = summ[(summ.escalate_to == "cloud") & (summ.row_set == "primary") & (summ.budget_target.isin([0.10, 0.25, 0.50]))]
    a10 = s10[["gate", "budget_target", "realised_rate_mean", "realised_rate_min", "realised_rate_max"]].copy()
    a10["realised_minus_target_mean"] = a10.realised_rate_mean - a10.budget_target
    a10 = a10.sort_values(["budget_target", "gate"]).reset_index(drop=True)
    write(a10, "a10_target_vs_realised_rate", "A10: target vs realised escalation rate on 2016", [
        "Share of the 7,457 evaluation issue times escalated, seed mean and range over 5 seeds. Source: "
        "results/test/gates/summary_test.csv (escalate_to = cloud, row_set = primary). Thresholds were fitted on the "
        "2015 gate-fit population, so realised rates on 2016 can differ from the target; random and oracle hit the "
        "target by construction, fixed_interval by its per-day schedule."])

    # A11 int8 vs fp32 gate decisions
    a11 = ("not available: no int8 gate scores for 2016 are stored. The int8 gate ONNX files in checkpoints/footprint/ "
           "were exported from the validation-time gates (checkpoints/gates/), not from the 2015-refit gates used on "
           "2016 (checkpoints/gates_test/), and were never run on 2016.")
    (OUT / "a11_int8_gate_agreement.md").write_text(f"{HEADER}\n\n# A11: int8 vs fp32 gate decisions @25%\n\n{a11}\n", encoding="utf-8")
    res["a11"] = a11

    res["files"] = sorted(p.name for p in OUT.glob("a*_*.*"))
    (OUT / "posthoc_summary.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(a8.to_string(), a9.to_string(), a10.to_string(), sep="\n\n")


if __name__ == "__main__":
    main()
