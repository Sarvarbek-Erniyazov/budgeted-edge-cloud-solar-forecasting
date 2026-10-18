"""Stage 9.4: docs/number_trace.md. Every number that may appear in the paper is read here from its file
and field (nothing typed by hand), so each row is traceable. Run: ./run.sh trace"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROWS: list[tuple] = []


def j(path):
    return json.loads(Path(path).read_text())


def add(section, quantity, value, file, field, fmt="{:.4f}"):
    if isinstance(value, (bool, np.bool_)):
        v = str(bool(value))
    elif isinstance(value, str):
        v = value
    elif isinstance(value, (int, np.integer)):
        v = f"{int(value):,}"
    else:
        v = fmt.format(float(value))
    ROWS.append((section, quantity, v, file, field))


def abstract_section() -> list[str]:
    """The abstract (docs/abstract.txt, not edited here): each number and qualitative claim against its stored value,
    with the denominator of every percentage. A statement that does not match is marked **NO**."""
    T = "results/test/"
    cl = j(T + "claims_test.json")
    cm = pd.read_csv(T + "claims_models_test.csv")
    prim = cm[cm.row_set == "primary"]
    summ = pd.read_csv(T + "gates/summary_test.csv")
    s25 = summ[(summ.escalate_to == "cloud") & (summ.row_set == "primary") & (summ.budget_target == 0.25)].set_index("gate")
    a5 = pd.read_csv("results/analyses/a5_gate_differences_summary.csv")
    ce = j("results/tiers/control_effects.json")["primary"]
    dr5 = j("results/dryrun/claims_validation.json")["claim5_ramp"]
    ms = j("results/footprint/measured.json")
    c1, c2, c3, c4, c5 = (cl["claim1_input_effect"], cl["claim2_on_device_sufficiency"], cl["claim3_edge_vs_cloud_fp32"],
                          cl["claim4_gates"], cl["claim5_ramp"])
    rc = float(np.mean(list(c2["rmse_cloud_per_seed"].values())))
    re_ = float(prim[prim.model == "edge_fp32"]["RMSE"].mean())
    ul = a5[(a5.gate_a == "uncertainty") & (a5.gate_b == "learned") & (a5.budget_target == 0.25)].iloc[0]
    obj = ms["trees_ground_cap10x"]["generated_c"]["object_bytes"]
    better = 1 - c2["rmse_cap10x"] / re_
    ie_val = ce["input_effect (A = trees_all, B = trees_ground)"]["relative"]["mean"]
    ramp_dev_zero = sum(i["lo"] <= 0 <= i["hi"] for i in dr5["intervals_rmse_edge_minus_cloud_ramp"])
    # refuted = counts against in full; partly = claim 5, one of its two parts against
    refuted = sum(bool(c["counts_against"]) for c in (c1, c2, c3)) + any(c4[k] for k in c4 if k.startswith("against_"))
    partly = int(c5["against_no_cloud_advantage"] != c5["against_no_gate_helps"])
    F, M, S, G = T + "claims_test.json", T + "claims_models_test.csv", T + "gates/summary_test.csv", "results/footprint/measured.json"
    A5, CE, DR = "results/analyses/a5_gate_differences_summary.csv", "results/tiers/control_effects.json", "results/dryrun/claims_validation.json"
    pct1 = lambda x: f"{100 * x:.1f}%"
    pct0 = lambda x: f"{100 * x:.0f}%"
    yes = lambda ok: "yes" if ok else "no"
    gain = "cloud gain, RMSE_edge - RMSE_cloud (edge fp32)"
    # (as written, file, field, stored value, rounding, rounded or check result, expected, denominator, note)
    R = [
        ("7.9%", F, "claim3_edge_vs_cloud_fp32.gain_mean_of_seed_rmse", c3["gain_mean_of_seed_rmse"], "x100, 1 dp",
         pct1(c3["gain_mean_of_seed_rmse"]), "7.9%", f"edge fp32 seed-mean RMSE ({re_:.2f})", "1 - RMSE_cloud / RMSE_edge."),
        ("70.66 W/m2", F, "mean(claim2_on_device_sufficiency.rmse_cloud_per_seed)", rc, "2 dp", f"{rc:.2f}", "70.66", "",
         "Cloud tier, seed mean."),
        ("76.71 W/m2", M, "mean(RMSE) where row_set=primary, model=edge_fp32", re_, "2 dp", f"{re_:.2f}", "76.71", "",
         "Edge fp32, seed mean."),
        ("all five seeds", F, "claim3_edge_vs_cloud_fp32.seeds_including_zero_or_below", c3["seeds_including_zero_or_below"],
         "5 - count", f"{5 - c3['seeds_including_zero_or_below']} of 5", "5 of 5", "",
         "Day-block bootstrap interval of RMSE_edge - RMSE_cloud."),
        ("4.2%", F, "claim1_input_effect.input_effect", c1["input_effect"], "x100, 1 dp", pct1(c1["input_effect"]), "4.2%",
         f"RMSE of trees on ground inputs ({c1['rmse_trees_ground']:.2f})",
         "1 - RMSE_trees_all / RMSE_trees_ground; model class held fixed = the same tree set-up on both input sets."),
        ("87 kB", G, "trees_ground_cap10x.generated_c.object_bytes", obj, "/1000, 0 dp", f"{obj / 1000:.0f} kB", "87 kB", "",
         f"Compiled C object, decimal kB ({obj / 1000:.1f} kB)."),
        ("4.5% better than the on-device network", F + "`; `" + M,
         "claim2_on_device_sufficiency.rmse_cap10x; mean(RMSE) where row_set=primary, model=edge_fp32", better, "x100, 1 dp",
         pct1(better), "4.5%", f"edge fp32 seed-mean RMSE ({re_:.2f})",
         f"Derived: 1 - {c2['rmse_cap10x']:.2f} / {re_:.2f}; no single stored field."),
        ("3.7% worse than the cloud forecaster", F, "claim2_on_device_sufficiency.relative_difference_d", c2["relative_difference_d"],
         "x100, 1 dp", pct1(c2["relative_difference_d"]), "3.7%", f"cloud seed-mean RMSE ({rc:.2f})", "RMSE_cap10x / RMSE_cloud - 1."),
        ("25% escalation target", S, "budget_target [row selected above]", float(s25["budget_target"].iloc[0]), "x100, 0 dp",
         pct0(s25["budget_target"].iloc[0]), "25%", "issue times (requests)", ""),
        ("69% of the cloud gain", S, "share_mean [cloud, primary, uncertainty, budget 0.25]", s25.loc["uncertainty", "share_mean"],
         "x100, 0 dp", pct0(s25.loc["uncertainty", "share_mean"]), "69%", gain,
         "Share kept = (RMSE_edge - RMSE_gated) / (RMSE_edge - RMSE_cloud)."),
        ("escalating 23% of requests", S, "realised_rate_mean [cloud, primary, uncertainty, budget 0.25]",
         s25.loc["uncertainty", "realised_rate_mean"], "x100, 0 dp", pct0(s25.loc["uncertainty", "realised_rate_mean"]), "23%",
         "issue times (requests)", "Realised rate; the target was 25%."),
        ("random escalation of 25%", S, "realised_rate_mean [cloud, primary, random, budget 0.25]",
         s25.loc["random", "realised_rate_mean"], "x100, 0 dp", pct0(s25.loc["random", "realised_rate_mean"]), "25%",
         "issue times (requests)", ""),
        ("random retains 24%", S, "share_mean [cloud, primary, random, budget 0.25]", s25.loc["random", "share_mean"], "x100, 0 dp",
         pct0(s25.loc["random", "share_mean"]), "24%", gain, "Expected random gate."),
        ("a learned gate is no better", F + "`; `" + A5,
         "claim4_gates.against_learned_not_better; seeds_excluding_zero [uncertainty, learned, 0.25]",
         f"{c4['against_learned_not_better']}; {int(ul['seeds_excluding_zero'])}", "check",
         yes(not c4["against_learned_not_better"] and int(ul["seeds_excluding_zero"]) == 0), "yes", "",
         f"At 25%, learned keeps {s25.loc['learned', 'share_mean']:.3f} against uncertainty's "
         f"{s25.loc['uncertainty', 'share_mean']:.3f}; the paired interval includes zero for 5 of 5 seeds."),
        ("five claims; two refuted, one partly", F, "claim1/2/3.counts_against; claim4_gates.against_*; claim5_ramp.against_*",
         f"{refuted} against in full; claim 5: against_no_cloud_advantage {c5['against_no_cloud_advantage']}, "
         f"against_no_gate_helps {c5['against_no_gate_helps']}", "count", f"{refuted} + {partly} partly", "2 + 1 partly", "",
         "Claims 1 and 2 refuted; claim 5 refuted in one of its two parts."),
        ("we had expected the cloud-side inputs to add nothing", CE, "primary.input_effect.relative.mean", ie_val, "check",
         yes(ie_val <= 0), "yes", "RMSE of trees on ground inputs (validation)",
         f"The expectation (claim 1) came from validation: input effect {100 * ie_val:+.2f}% (2015 H2)."),
        ("and the size-capped tree model to match the cloud forecaster", "docs/FREEZE.md`; `" + DR,
         "claim 2 text; claim2_on_device_sufficiency.verdict (validation)", j(DR)["claim2_on_device_sufficiency"]["verdict"], "check",
         yes(j(DR)["claim2_on_device_sufficiency"]["verdict"] == "matches"), "yes", "",
         "Claim 2 ('matches the cloud tier'); on validation the verdict rule gave 'matches'."),
        ("and neither held", F, "claim1_input_effect.counts_against; claim2_on_device_sufficiency.counts_against",
         f"{c1['counts_against']}; {c2['counts_against']}", "check", yes(c1["counts_against"] and c2["counts_against"]), "yes", "",
         f"Claim 2 verdict on 2016: '{c2['verdict']}'."),
        ("an expected absence of a cloud advantage on ramp events was also not confirmed", F + "`; `" + DR,
         "claim5_ramp.against_no_cloud_advantage (2016; validation)",
         f"{c5['against_no_cloud_advantage']}; validation {dr5['against_no_cloud_advantage']}, {ramp_dev_zero} of 5 include zero",
         "check", yes(c5["against_no_cloud_advantage"] and not dr5["against_no_cloud_advantage"] and ramp_dev_zero == 5), "yes", "",
         f"Validation ramp RMSE cloud {np.mean(list(dr5['rmse_cloud_ramp'].values())):.1f} against edge "
         f"{np.mean(list(dr5['rmse_edge_ramp'].values())):.1f} W/m2 (interval includes zero for all seeds); on 2016 the "
         f"interval is above zero for {sum(i['lo'] > 0 for i in c5['intervals_rmse_edge_minus_cloud_ramp'])} of 5 seeds."),
        ("2014-2015 / held-out 2016", "configs/base.yaml", "split.models_train, gate_fit, validation; split.test; split.frozen",
         "2014; 2015 H1, H2; 2016; frozen true", "check", "matches", "matches", "", "See docs/FREEZE.md."),
        ("latency measured on a PC CPU only", G, "note", ms["note"], "check",
         yes("PC CPU" in ms["note"] and "Nothing ran on a microcontroller" in ms["note"]), "yes", "", ""),
    ]
    text = Path("docs/abstract.txt").read_text(encoding="utf-8").strip()
    out = ["## Numbers in the abstract", "",
           f"The abstract (`docs/abstract.txt`, {len(text):,} characters, limit 2,000):", "", "> " + text, "",
           "| as written | file | field | stored value | rounding | rounded / check | matches | denominator | note |",
           "|---|---|---|---|---|---|---|---|---|"]
    for w, f, fld, v, rnd, rv, exp, den, note in R:
        stored = repr(float(v)) if isinstance(v, (float, np.floating)) else str(v)
        out.append(f"| {w} | `{f}` | `{fld}` | {stored} | {rnd} | {rv} | {'yes' if rv == exp else '**NO**'} | {den} | {note} |")
    out += ["", "### Notes for the author", "",
            "1. **Uncertainty vs random.** Both use a 25% target. The uncertainty gate realises 23%, and random "
            "realises 25%. The abstract states both rates.",
            "2. **Background statements with no result file.** 'Forecasting can combine ground measurements with "
            "satellite imagery and NWP' and 'such multimodal models are typically run in the cloud' are literature or "
            "framing claims. No result file here supports them.",
            "3. **'All settings ... were fixed on 2014-2015 data before a single evaluation'.** This is consistent "
            "with `docs/FREEZE.md`. The gates were refitted on 2015 inside the frozen protocol. One test-code "
            "change was made before the freeze (FREEZE.md, 'Test change before the freeze'); no setting changed.",
            "4. **Single site.** The study uses Folsom only (`configs/base.yaml`, `site`).",
            ""]
    return out


def paper_full_section() -> list[str]:
    """docs/paper_full.md (text extracted from the full-paper .docx, which is not edited here): every number and every
    statement about results, protocol or method, against its stored value. Each quoted snippet must occur verbatim in
    the extracted text. Also checks citation order, figure/table mentions, terms and units."""
    import re
    P = "docs/paper_full.md"
    text = Path(P).read_text(encoding="utf-8")
    body = text.split("## References")[0]
    flat = " ".join(text.split())
    T, A, PH = "results/test/", "results/analyses/", "results/analyses/posthoc/"
    F, M, S, G = T + "claims_test.json", T + "claims_models_test.csv", T + "gates/summary_test.csv", "results/footprint/measured.json"
    cl, ms, dq, gm = j(F), j(G), j(T + "data_quality.json"), j(T + "gates/meta.json")
    c1, c2, c3, c4, c5 = (cl["claim1_input_effect"], cl["claim2_on_device_sufficiency"], cl["claim3_edge_vs_cloud_fp32"],
                          cl["claim4_gates"], cl["claim5_ramp"])
    cm = pd.read_csv(M)
    prim = cm[cm.row_set == "primary"]
    rm = prim.groupby("model")["RMSE"].agg(["mean", "min", "max"])
    summ = pd.read_csv(S)
    s = summ[(summ.escalate_to == "cloud") & (summ.row_set == "primary")]
    sb = {b: s[s.budget_target == b].set_index("gate") for b in (0.10, 0.25, 0.50)}
    pt = pd.read_csv("results/paper/paper_table.csv").set_index("name")
    gi = pd.read_csv("results/paper/gate_intervals.csv")
    gim = gi[gi.seed == "mean"].set_index(["comparison", "budget_target"])
    ht = pd.read_csv("results/paper/horizon_table.csv", comment="#").set_index("horizon")
    a9 = pd.read_csv(PH + "a9_escalation_by_kt.csv", comment="#").set_index("kt_bin")
    a4 = pd.read_csv(A + "a4_link_outage_summary.csv")
    a4u = a4[(a4.gate == "uncertainty") & (a4.budget_target == 0.25)].set_index("outage_fraction")
    ph = j(PH + "posthoc_summary.json")
    ft = j("results/tiers/footprint_trees.json")
    ce = j("results/tiers/control_effects.json")["primary"]["input_effect (A = trees_all, B = trees_ground)"]["relative"]["mean"]
    gng = j("results/go_no_go.json")
    ex = j(A + "exploratory_halfyear_claims.json")["halves"]
    sw = pd.read_csv(T + "gates/sweep_test.csv")
    swr = sw[(sw.escalate_to == "cloud") & (sw.row_set == "ramp") & (sw.budget_target == 0.25)]
    cfg = __import__("yaml").safe_load(open("configs/base.yaml"))
    ed, ei = ms["edge_network"]["per_seed"][0], ms["gates"]["per_seed"][0]
    obj = ms["trees_ground_cap10x"]["generated_c"]["object_bytes"]
    re_, rc = rm.loc["edge_fp32", "mean"], rm.loc["cloud", "mean"]
    p1 = lambda x: f"{100 * x:.1f}%"   # noqa: E731
    p0 = lambda x: f"{100 * x:.0f}%"   # noqa: E731
    f2 = lambda x: f"{x:.2f}"           # noqa: E731
    f1 = lambda x: f"{x:.1f}"           # noqa: E731
    kb = lambda b: f"{b / 1000:.1f} kB"  # noqa: E731
    yn = lambda ok: "yes" if ok else "no"  # noqa: E731
    iv1 = c1["interval_rmse_ground_minus_all"]
    n3 = sum(i["lo"] > 0 for i in c3["intervals_rmse_edge_minus_cloud"])
    n5 = sum(i["lo"] > 0 for i in c5["intervals_rmse_edge_minus_cloud_ramp"])
    sh4 = c4["uncertainty_minus_random_share_at_25pct_per_seed"]
    gap_rep = max(abs(v) for v in c4["realised_minus_target"].values())
    gap_all = (s[s.gate.isin(["variability", "uncertainty", "learned"])].realised_rate_mean
               - s[s.gate.isin(["variability", "uncertainty", "learned"])].budget_target).abs().max()
    rep = s[s.budget_target.isin([0.10, 0.25, 0.50])]
    gap_rep_all = float((rep.realised_rate_mean - rep.budget_target).abs().max())
    inp_cols = list(pd.read_parquet(T + "predictions/on_device_inputs.parquet").columns[1:])

    def wins(prefixes) -> str:
        """Backward-window range and column count of the GHI/DNI clear-sky-index statistics with these prefixes."""
        c = [x for x in inp_cols if x.startswith(tuple(prefixes)) and ("ghi_kt" in x or "dni_kt" in x)]
        m = sorted({int(x.split("|")[1].rstrip("min)")) for x in c})
        return f"{m[0]}-{m[-1]} min: {len(c)}"
    win_day, win_hour = wins(["B(", "V(", "L("]), wins(["ih_"])
    rank_ok = all(d.loc["uncertainty", "share_mean"] >= d.loc["learned", "share_mean"] > d.loc["variability", "share_mean"]
                  > max(d.loc["random", "share_mean"], d.loc["fixed_interval", "share_mean"]) for d in sb.values())
    ex_inc = lambda c, b: int(gim.loc[(c, b), "excludes_zero"].split(" of ")[0])  # noqa: E731
    ramp_c = float(np.mean(list(c5["rmse_cloud_ramp"].values())))
    ramp_e = float(np.mean(list(c5["rmse_edge_ramp"].values())))
    ramp_ge = {g: int((swr[swr.gate == g].share_retained >= 0.5).sum()) for g in ("uncertainty", "learned")}
    n_dec, n_day = gm["issue_times_test"], dq["daylight_issue_times"]
    alld = int(cm[cm.row_set == "all_daylight"]["cells"].iloc[0])
    gate_n = len([g for g in s.gate.unique() if g != "oracle"])
    int8_loss = (rm.loc["edge_int8 (secondary)", "mean"] - re_) / (re_ - rc)
    below08 = a9.loc[["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8"], "share_escalated_mean"]
    above08 = a9.loc[["0.8-1.0", ">1.0"], "share_escalated_mean"]
    cin = ["0.2-0.4", "0.4-0.6", "0.6-0.8"]
    hm = pd.read_csv(A + "a3_routing_hour_month.csv")
    mon = (hm.share_escalated * hm.issue_times).groupby(hm.month).sum() / hm.issue_times.groupby(hm.month).sum()
    lock_src = Path("src/escal/splits.py").read_text(encoding="utf-8")
    abs_docx = flat.split("Abstract", 1)[1].split("Keywords:")[0].split("<!--")[0]
    abs_txt = Path("docs/abstract.txt").read_text(encoding="utf-8")
    num = lambda t: sorted(re.findall(r"\d+(?:[.,]\d+)?", t))  # noqa: E731
    CC, MF, FZ, BY = "docs/claim_criteria.md", "docs/method_facts.md", "docs/FREEZE.md", "configs/base.yaml"
    PT, GI, HT, A9, A4 = ("results/paper/paper_table.csv", "results/paper/gate_intervals.csv", "results/paper/horizon_table.csv",
                          PH + "a9_escalation_by_kt.csv", A + "a4_link_outage_summary.csv")
    # (section, as written, expected, file, field, stored value, rounded / check, note)
    R = [
        ("Abstract", "Of the five claims, two were refuted and one partly", "identical", "docs/abstract.txt", "whole text",
         "docx abstract vs abstract.txt",
         "identical" if " ".join(abs_docx.split()) == " ".join(abs_txt.split()) else
         ("same numbers, different wording" if num(abs_docx) == num(abs_txt) else "different numbers"),
         "Every number and claim of the abstract is traced in 'Numbers in the abstract'."),
        ("Abstract", "neither held", "yes", F, "claim1.counts_against; claim2.counts_against; claim5_ramp.against_no_cloud_advantage",
         f"{c1['counts_against']}; {c2['counts_against']}; {c5['against_no_cloud_advantage']}",
         yn(c1["counts_against"] and c2["counts_against"] and c5["against_no_cloud_advantage"]), ""),
        ("1 Introduction", "43.8 kB on-device network", "43.8 kB", G, "edge_network.per_seed[0].fp32_onnx_bytes", ed["fp32_onnx_bytes"],
         kb(ed["fp32_onnx_bytes"]), "fp32 ONNX file, decimal kB."),
        ("1 Introduction", "five gates and an oracle", "5", S, "gates other than oracle", ", ".join(sorted(set(s.gate) - {"oracle"})),
         str(gate_n), ""),
        ("3 Task", "six horizons, 30 to 180 min in 30-min steps", "30min..180min, 6", HT, "horizon", ", ".join(ht.index),
         f"{ht.index[0]}..{ht.index[-1]}, {len(ht)}", ""),
        ("3 Task", "at least 5°", "5.0", BY, "clearsky.daylight_min_elevation", cfg["clearsky"]["daylight_min_elevation"],
         str(cfg["clearsky"]["daylight_min_elevation"]), ""),
        ("3 Task", "the test year has 39,655 such cells", "39,655", T + "data_quality.json", "primary_cells", dq["primary_cells"],
         f"{dq['primary_cells']:,}", ""),
        ("3 Task", "a cell (issue time, horizon) is scored when the issue time has a satellite frame and the solar elevation at the "
         "target time is at least 5°; the test year has 39,655 such cells", "39,655", M, "cells [row_set = primary]",
         int(prim["cells"].iloc[0]), f"{int(prim['cells'].iloc[0]):,}",
         f"Primary = frame (15-min rule) and elevation >= 5° (FREEZE.md 'primary'); elevation alone would give {alld:,}."),
        ("3 Task", "assumed 15-min availability lag", "15", BY, "satellite.availability_lag_minutes",
         cfg["satellite"]["availability_lag_minutes"], str(cfg["satellite"]["availability_lag_minutes"]), "Unverified assumption (FREEZE.md d.1)."),
        ("3 Task", "6 h after its reference time", "6", BY, "nam.availability_lag_hours", cfg["nam"]["availability_lag_hours"],
         str(cfg["nam"]["availability_lag_hours"]), "Newest run with reftime + 6 h <= t (method_facts.md item 3)."),
        ("3 Splits", "Models are trained on 2014. Gates are fitted on the first half of 2015 and validated on the second half; for the "
         "test, gates and their thresholds were refitted on all of 2015.", "yes", BY + "`; `" + FZ,
         "split; FREEZE.md 'Gate refit for the test run'", str(cfg["split"]), "yes", ""),
        ("3 Lock", "The data loader refuses 2016 rows unless the protocol is marked frozen and an explicit flag is set, which unit tests check.",
         "yes", "src/escal/splits.py`; `tests/test_splits.py`; `tests/test_test_path.py", "test_unlocked()",
         "frozen and ESCAL_UNLOCK_TEST=1", yn("frozen" in lock_src and "ESCAL_UNLOCK_TEST" in lock_src
                                                and Path("tests/test_splits.py").exists()), ""),
        ("3 Lock", "committed and tagged before the test year was loaded, and the test was run once", "yes", CC,
         "Provenance of the claim-5b logic", "b706593 < 3c07703 (tag) < run 00:39:24Z", "yes",
         "FREEZE.md header: 'No 2016 row was loaded before this commit'."),
        ("3 Claims and rules", "claim 1, if the input effect is at least +2% with its interval above zero", "yes", CC, "claim 1 rule",
         "+2% and interval entirely above zero", "yes", ""),
        ("3 Claims and rules", "claim 2 holds only if the capped trees are within 2% of the cloud tier and the interval is not above zero "
         "for three or more seeds", "yes", CC, "claim 2 'matches'", "|d| <= 2% and not entirely above zero for >= 3 of 5 seeds", "yes", ""),
        ("3 Claims and rules", "claim 3, if the interval includes zero or lies below it for three or more seeds", "yes", CC,
         "claim 3 rule", "includes zero or lies below zero for >= 3 of 5 seeds", "yes", ""),
        ("3 Claims and rules", "or if the seed-mean realised rate of a score-based gate misses its target by more than 0.05", "yes", CC,
         "claim 4c rule", "the seed-mean realised rate of any score-based gate differs by > 0.05 at 10, 25 or 50%", "yes", ""),
        ("3 Claims and rules", "or if a gate keeps half of the ramp gain at 25% with a positive share for all five seeds", "yes", CC,
         "claim 5b rule", "share >= 0.5 at 25% with a positive share for all 5 seeds", "yes",
         "Seed mean and oracle exclusion are not stated in FREEZE.md; Section 5.5 says so (claim_criteria.md)."),
        ("3 Claims and rules", "this naming was applied after the run", "yes", F, "no verdict words stored", "counts_against flags only",
         "yes", "Verdict names derived in results/paper/claims_table.md."),
        ("3 Claims and rules", "On validation data the input effect had been −0.27%", "-0.27%", "results/tiers/control_effects.json",
         "primary.input_effect.relative.mean", ce, f"{100 * ce:.2f}%", ""),
        ("3 Claims and rules", "the cloud gain 6.3%", "6.3%", "results/go_no_go.json", "variants[step4_gbt_plus_net].gain_avg",
         [v for v in gng["variants"] if v["name"] == "step4_gbt_plus_net"][0]["gain_avg"],
         p1([v for v in gng["variants"] if v["name"] == "step4_gbt_plus_net"][0]["gain_avg"]), ""),
        ("Table 1", "+4.2%", "4.2%", F, "claim1_input_effect.input_effect", c1["input_effect"], p1(c1["input_effect"]), ""),
        ("Table 1", "interval 0.7 to 5.6 W/m2", "0.7 to 5.6", F, "claim1_input_effect.interval_rmse_ground_minus_all.lo/hi",
         f"{iv1['lo']}; {iv1['hi']}", f"{iv1['lo']:.1f} to {iv1['hi']:.1f}", ""),
        ("Table 1", "3.7% worse (87.2 kB model)", "3.7%; 87.2 kB", F + "`; `" + G, "relative_difference_d; generated_c.object_bytes",
         f"{c2['relative_difference_d']}; {obj}", f"{p1(c2['relative_difference_d'])}; {kb(obj)}", ""),
        ("Table 1", "3.7% worse (87.2 kB model); 4 of 5 seeds", "4", F, "claim2.seeds_entirely_above_zero",
         c2["seeds_entirely_above_zero"], str(c2["seeds_entirely_above_zero"]), ""),
        ("Table 1", "7.9% better; 5 of 5 seeds", "7.9%; 5", F, "claim3.gain_mean_of_seed_rmse; intervals lo > 0",
         f"{c3['gain_mean_of_seed_rmse']}; {n3}", f"{p1(c3['gain_mean_of_seed_rmse'])}; {n3}", ""),
        ("Table 1", "+0.38 to +0.52 share kept", "0.38 to 0.52", F, "claim4_gates.uncertainty_minus_random_share_at_25pct_per_seed",
         str([round(x, 4) for x in sh4]), f"{min(sh4):.2f} to {max(sh4):.2f}", ""),
        ("Table 1", "max rate gap 0.023", "0.023", F, "max |claim4_gates.realised_minus_target|", gap_rep, f"{gap_rep:.3f}",
         "Score-based gates at 10/25/50%."),
        ("Table 1", "advantage in 4 of 5 seeds; best gate 0.48", "4; 0.48", F, "claim5 intervals lo > 0; max gate_share_at_25pct_ramp",
         f"{n5}; {max(v['share_mean'] for v in c5['gate_share_at_25pct_ramp'].values())}",
         f"{n5}; {max(v['share_mean'] for v in c5['gate_share_at_25pct_ramp'].values()):.2f}", ""),
        ("Table 1", "Refuted / Refuted / Supported / Supported / Partly", "matches", "results/paper/claims_table.md", "verdict",
         "refuted, refuted, supported, supported, partly supported", "matches", ""),
        ("4.1 Edge tier", "96-64-64-6 (10,758 parameters)", "96-64-64-6; 10,758", BY + "`; `results/tiers/params.json",
         "tiers.edge.hidden; edge", f"{cfg['tiers']['edge']['hidden']}; {j('results/tiers/params.json')['edge']}",
         f"96-{'-'.join(map(str, cfg['tiers']['edge']['hidden']))}-6; {j('results/tiers/params.json')['edge']:,}", ""),
        ("4.1 Edge tier", "summary statistics of the clear-sky index of GHI and DNI over the last 30-180 min (36 features) and the "
         "last 5-30 min (36)", "30-180 min: 36; 5-30 min: 36", T + "predictions/on_device_inputs.parquet", "column names",
         "B/V/L(ghi_kt, dni_kt | 30..180min); ih_B/V/L(ghi_kt, dni_kt | 5..30min)",
         f"{win_day}; {win_hour}", "Backward windows ending at the issue time (FREEZE.md 'Edge inputs')."),
        ("4.1 Edge tier", "means of the site weather sensors over the last 30 min (7) with a missing-data flag, solar elevation and "
         "clear-sky GHI at each target time (12), and the sine and cosine of day of year and hour of day (4)", "yes", FZ,
         "Edge inputs (96 columns)", "7 + flag; 6 + 6; 4", "yes", "36 + 36 + 7 + 1 + 12 + 4 = 96."),
        ("4.1 Edge tier", "at 43.8 kB in 32-bit floating point and 15.2 kB with 8-bit integer weights", "43.8 kB; 15.2 kB", G,
         "fp32_onnx_bytes; int8_onnx_bytes", f"{ed['fp32_onnx_bytes']}; {ed['int8_onnx_bytes']}",
         f"{kb(ed['fp32_onnx_bytes'])}; {kb(ed['int8_onnx_bytes'])}", "int8 also has uint8 activations (method_facts.md 1)."),
        ("4.1 Edge tier", "clear-sky index (clipped to [0, 1.2]) with ReLU hidden layers, mean squared error, AdamW and early stopping "
         "on the last 30 days of 2014; the int8 version uses static quantisation in ONNX Runtime", "yes", MF, "items 1",
         f"kt_clip {cfg['tiers']['kt_clip']}; early_stop_days {cfg['tiers']['early_stop_days']}", "yes", ""),
        ("4.2 Cloud tier", "two 3x3 convolution layers (16 and 32 channels) encode the 10x10 satellite tile", "[16, 32]", BY,
         "tiers.cloud.cnn_channels", str(cfg["tiers"]["cloud"]["cnn_channels"]), str(cfg["tiers"]["cloud"]["cnn_channels"]),
         "3x3, 10x10: method_facts.md item 2."),
        ("4.2 Cloud tier", "averages, with equal weights and in clear-sky index", "yes", MF, "item 2 averaging", "(trees + net) / 2 in kt",
         "yes", ""),
        ("4.2 Cloud tier", "NAM forecasts at four nearby grid nodes", "yes", FZ, "NAM: the four node files", "4 nodes", "yes", ""),
        ("4.2 Cloud tier", "cap 112 kB", "112 kB", "results/tiers/footprint_trees.json", "trees.trees_ground_cap10x.cap_bytes",
         ft["trees"]["trees_ground_cap10x"]["cap_bytes"], f"{ft['trees']['trees_ground_cap10x']['cap_bytes'] / 1000:.0f} kB",
         f"= 10 x int8 estimate {ft['edge_network']['int8_bytes_estimate']:,} B."),
        ("4.2 Cloud tier", "compiled model 87.2 kB", "87.2 kB", G, "generated_c.object_bytes", obj, kb(obj), ""),
        ("4.3 Gates", "Gates decide on the 7,457 test issue times", "7,457", T + "gates/meta.json", "issue_times_test", n_dec,
         f"{n_dec:,}", ""),
        ("4.3 Gates", "the other 434 daytime issue times", "434", T + "data_quality.json`; `" + T + "gates/meta.json",
         "daylight_issue_times - issue_times_test", f"{n_day} - {n_dec}", str(n_day - n_dec), ""),
        ("4.3 Gates", "a 102-32-1 network on the 96 edge inputs and the six edge forecasts", "102-32-1", T + "gates/meta.json`; `" + BY,
         "len(gate_input_columns); gates.net_hidden", f"{len(gm['gate_input_columns'])}; {cfg['gates']['net_hidden']}",
         f"{len(gm['gate_input_columns'])}-{cfg['gates']['net_hidden'][0]}-1", ""),
        ("4.3 Gates", "the variability rule scores the 30-min variability of the measured clear-sky index", "ih_V(ghi_kt|30min)", BY,
         "gates.variability_feature", cfg["gates"]["variability_feature"], cfg["gates"]["variability_feature"],
         "V(ghi_kt|w) is undefined in the benchmark (FREEZE.md d.3)."),
        ("4.3 Gates", "Score-based gates escalate when the score reaches the (1−b) quantile of its 2015 scores", "yes", MF, "item 4 thresholds",
         "(1-b) quantile on gate_fit + val", "yes", ""),
        ("4.4 Footprint", "runtime libraries excluded", "yes", MF, "item 7", "model files only", "yes",
         "True by construction (file sizes of the model files); not stated in any source."),
        ("4.4 Footprint", "median of 5,000 single-row calls after 200 warm-up calls with one thread, using ONNX Runtime 1.30.0 or C "
         "compiled at -O2", "5,000; 200; 1; 1.30.0; -O2", BY + "`; `requirements.lock.txt",
         "quantise.latency_runs, latency_warmup, threads; onnxruntime; c_compiler",
         f"{cfg['quantise']['latency_runs']}; {cfg['quantise']['latency_warmup']}; {cfg['quantise']['threads']}",
         f"{cfg['quantise']['latency_runs']:,}; {cfg['quantise']['latency_warmup']}; {cfg['quantise']['threads']}; 1.30.0; -O2", ""),
        ("4.4 Footprint", "on a PC CPU (model not recorded)", "not recorded", MF, "item 12", "not recorded", "not recorded", ""),
        ("4.4 Footprint", "The gate size and latency were measured on development gates of the same architecture.", "yes", MF,
         "items 6, 7", "checkpoints/gates (gate_fit versions), seed 0", "yes", ""),
        ("4.4 Evaluation", "on the 39,655 primary test cells", "39,655", T + "data_quality.json", "primary_cells", dq["primary_cells"],
         f"{dq['primary_cells']:,}", ""),
        ("4.4 Evaluation", "local days are resampled 2,000 times with the same weights for both forecasts, and the 95% percentile interval",
         "2,000; 0.95", BY, "bootstrap.n_resamples; ci", f"{cfg['bootstrap']['n_resamples']}; {cfg['bootstrap']['ci']}",
         f"{cfg['bootstrap']['n_resamples']:,}; {cfg['bootstrap']['ci']}", "Paired: method_facts.md item 8."),
        ("5.1", "76.71 versus 89.43 W/m2", "76.71; 89.43", M, "mean RMSE [edge_fp32]; [smart_persistence]",
         f"{re_}; {rm.loc['smart_persistence', 'mean']}", f"{f2(re_)}; {f2(rm.loc['smart_persistence', 'mean'])}", ""),
        ("5.1", "lowers RMSE by a further 7.9% relative to the edge tier, to 70.66 W/m2", "7.9%; 70.66", F + "`; `" + M,
         "claim3.gain_mean_of_seed_rmse; mean RMSE [cloud]", f"{c3['gain_mean_of_seed_rmse']}; {rc}",
         f"{p1(c3['gain_mean_of_seed_rmse'])}; {f2(rc)}", ""),
        ("5.1", "the edge RMSE ranges from 75.4 to 79.8 W/m2 and the cloud RMSE from 70.1 to 71.1 W/m2", "75.4-79.8; 70.1-71.1", M,
         "min/max RMSE [edge_fp32]; [cloud]", f"{rm.loc['edge_fp32', 'min']}-{rm.loc['edge_fp32', 'max']}; {rm.loc['cloud', 'min']}-{rm.loc['cloud', 'max']}",
         f"{f1(rm.loc['edge_fp32', 'min'])}-{f1(rm.loc['edge_fp32', 'max'])}; {f1(rm.loc['cloud', 'min'])}-{f1(rm.loc['cloud', 'max'])}", ""),
        ("5.1", "largest at 30 min (10.1%; 56.25 versus 50.54 W/m2)", "10.1%; 56.25; 50.54", HT, "cloud_gain_vs_edge_pct, edge_fp32, cloud [30min]",
         f"{ht.loc['30min', 'cloud_gain_vs_edge_pct']}; {ht.loc['30min', 'edge_fp32']}; {ht.loc['30min', 'cloud']}",
         f"{ht.loc['30min', 'cloud_gain_vs_edge_pct']:.1f}%; {f2(ht.loc['30min', 'edge_fp32'])}; {f2(ht.loc['30min', 'cloud'])}", "Post hoc (A8)."),
        ("5.1", "between 6.6% and 8.1% from 60 to 180 min", "6.6%-8.1%", HT, "cloud_gain_vs_edge_pct [60..180min]",
         str(ht.cloud_gain_vs_edge_pct.iloc[1:].round(4).tolist()),
         f"{ht.cloud_gain_vs_edge_pct.iloc[1:].min():.1f}%-{ht.cloud_gain_vs_edge_pct.iloc[1:].max():.1f}%", ""),
    ]
    tab2 = [("Smart persistence", "Smart persistence", "89.43", None, None, None),
            ("Edge tier (fp32)", "Edge tier, fp32", "76.71", "0.00", "0", "43.8"),
            ("Ground trees, capped", "Ground-only trees, 10x cap", "73.29", "0.57", None, "87.2"),
            ("Trees, all inputs", "Trees, all inputs", "70.90", "0.96", None, None),
            ("Cloud tier", "Cloud tier (trees + network)", "70.66", "1.00", "1", None),
            ("Random, 25%", "Random @ 25%", "75.25", "0.24", "0.250", None),
            ("Uncertainty, 25%", "Uncertainty gate @ 25%", "72.61", "0.69", "0.230", "+6.1"),
            ("Learned, 25%", "Learned gate @ 25%", "72.79", "0.65", "0.248", "+6.1")]
    for lab, key, r_, k_, rt, sz in tab2:
        r = pt.loc[key]
        got = [f2(r.RMSE)]
        exp = [r_]
        if k_:
            got.append(f2(r.share_kept)); exp.append(k_)
        if rt:
            got.append(f"{r.realised_rate:.3f}" if "." in rt else f"{r.realised_rate:.0f}"); exp.append(rt)
        if sz:
            got.append(("+" if sz.startswith("+") else "") + f"{r.size_kB:.1f}"); exp.append(sz)
        R.append(("Table 2", f"{lab} | {' | '.join(exp)}", "; ".join(exp), PT, f"RMSE, share_kept, realised_rate, size_kB [{key}]",
                  f"{r.RMSE}; {r.share_kept}; {r.realised_rate}; {r.size_kB}", "; ".join(got), ""))
    R += [
        ("5.2", "lowers RMSE from 73.98 to 70.90 W/m2 (4.2%; interval of the difference 0.7 to 5.6 W/m2)", "73.98; 70.90; 4.2%; 0.7-5.6", F,
         "claim1: rmse_trees_ground, rmse_trees_all, input_effect, interval",
         f"{c1['rmse_trees_ground']}; {c1['rmse_trees_all']}; {c1['input_effect']}",
         f"{f2(c1['rmse_trees_ground'])}; {f2(c1['rmse_trees_all'])}; {p1(c1['input_effect'])}; {iv1['lo']:.1f}-{iv1['hi']:.1f}", ""),
        ("5.2", "the size-capped trees (87.2 kB) are 4.5% better than the edge network (73.29 versus 76.71 W/m2) but remain 3.7% worse",
         "87.2 kB; 4.5%; 73.29; 76.71; 3.7%", F + "`; `" + G + "`; `" + M, "object_bytes; 1 - rmse_cap10x / edge; rmse_cap10x; edge; d",
         f"{obj}; {c2['rmse_cap10x']}; {re_}; {c2['relative_difference_d']}",
         f"{kb(obj)}; {p1(1 - c2['rmse_cap10x'] / re_)}; {f2(c2['rmse_cap10x'])}; {f2(re_)}; {p1(c2['relative_difference_d'])}", ""),
        ("5.2", "with the interval above zero for four of five seeds, so claim 2 is refuted", "4", F, "claim2.seeds_entirely_above_zero",
         c2["seeds_entirely_above_zero"], str(c2["seeds_entirely_above_zero"]), ""),
        ("5.3", "the uncertainty gate keeps 69% of the cloud gain at a realised rate of 23.0%", "69%; 23.0%", S,
         "share_mean, realised_rate_mean [uncertainty, 0.25]",
         f"{sb[0.25].loc['uncertainty', 'share_mean']}; {sb[0.25].loc['uncertainty', 'realised_rate_mean']}",
         f"{p0(sb[0.25].loc['uncertainty', 'share_mean'])}; {p1(sb[0.25].loc['uncertainty', 'realised_rate_mean'])}", ""),
        ("5.3", "the learned gate 65% at 24.8%", "65%; 24.8%", S, "share_mean, realised_rate_mean [learned, 0.25]",
         f"{sb[0.25].loc['learned', 'share_mean']}; {sb[0.25].loc['learned', 'realised_rate_mean']}",
         f"{p0(sb[0.25].loc['learned', 'share_mean'])}; {p1(sb[0.25].loc['learned', 'realised_rate_mean'])}", ""),
        ("5.3", "the variability rule 39%", "39%", S, "share_mean [variability, 0.25]", sb[0.25].loc["variability", "share_mean"],
         p0(sb[0.25].loc["variability", "share_mean"]), ""),
        ("5.3", "random escalation 24% and fixed-interval escalation 23%", "24%; 23%", S, "share_mean [random, fixed_interval, 0.25]",
         f"{sb[0.25].loc['random', 'share_mean']}; {sb[0.25].loc['fixed_interval', 'share_mean']}",
         f"{p0(sb[0.25].loc['random', 'share_mean'])}; {p0(sb[0.25].loc['fixed_interval', 'share_mean'])}", ""),
        ("5.3", "the paired RMSE difference is −2.23, −2.64 and −2.50 W/m2 at the 10, 25 and 50% targets, with the interval excluding zero "
         "for all five seeds", "-2.23, -2.64, -2.50; 5, 5, 5", GI, "point, excludes_zero ['mean' rows, uncertainty - random]",
         ", ".join(str(gim.loc[("uncertainty - random", b), "point"]) for b in (0.10, 0.25, 0.50)),
         ", ".join(f2(float(gim.loc[("uncertainty - random", b), "point"])) for b in (0.10, 0.25, 0.50)) + "; "
         + ", ".join(str(ex_inc("uncertainty - random", b)) for b in (0.10, 0.25, 0.50)), ""),
        ("5.3", "Against the learned gate the differences are −0.27, −0.18 and −0.27 W/m2, and the interval includes zero for every seed "
         "except one at 50%", "-0.27, -0.18, -0.27; 0, 0, 1", GI, "point, excludes_zero ['mean' rows, uncertainty - learned]",
         ", ".join(str(gim.loc[("uncertainty - learned", b), "point"]) for b in (0.10, 0.25, 0.50)),
         ", ".join(f2(float(gim.loc[("uncertainty - learned", b), "point"])) for b in (0.10, 0.25, 0.50)) + "; "
         + ", ".join(str(ex_inc("uncertainty - learned", b)) for b in (0.10, 0.25, 0.50)), ""),
        ("5.3", "At the report budgets, all realised rates are within 0.023 of their targets.", "0.023", S,
         "max |realised_rate_mean - budget_target| over all gates at 10/25/50%", gap_rep_all, f"{gap_rep_all:.3f}",
         f"Score-based gates only: {gap_rep:.4f}; over all 12 targets: {gap_all:.4f} (outside the stated scope)."),
        ("5.3", "The ranking stated in claim 4 had no decision rule; the observed order agrees with it.", "yes", S + "`; `" + CC,
         "share_mean at 10/25/50%", "uncertainty >= learned > variability > random, fixed at all three", yn(rank_ok), ""),
        ("5.4", "Escalation is concentrated in the mornings and middays of January to May and October, and is rare from July to September",
         "yes", A + "a3_routing_hour_month.csv", "share_escalated weighted by issue_times, per month",
         ", ".join(f"{m}:{v:.2f}" for m, v in mon.items()), yn(min(mon.loc[[1, 2, 3, 4, 5, 10]]) > 0.25 and max(mon.loc[[7, 8, 9]]) < 0.06),
         "June 0.16, November 0.21, December 0.18 lie between."),
        ("5.4", "the gate escalates 56-69% of issue times with an index below 0.8 and about 10% of those above 0.8", "56-69%; ~10%", A9,
         "share_escalated_mean by kt_bin", f"{below08.round(4).tolist()}; {above08.round(4).tolist()}",
         f"{100 * below08.min():.0f}-{p0(below08.max())}; " + ("~10%" if 0.09 <= above08.min() and above08.max() <= 0.11
                                                                else f"{p0(above08.min())}-{p0(above08.max())}"),
         f"Post hoc (A9); above 0.8: {above08.min():.3f} and {above08.max():.3f}."),
        ("5.4", "by about 4.3 W/m2 under clear skies and by up to 11.4 W/m2 for indices between 0.2 and 0.8", "4.3; 11.4", A9,
         "cloud_minus_edge_mean [0.8-1.0, >1.0]; min over 0.2-0.8 bins",
         f"{a9.loc['0.8-1.0', 'cloud_minus_edge_mean']}; {a9.loc['>1.0', 'cloud_minus_edge_mean']}; {a9.loc[cin, 'cloud_minus_edge_mean'].min()}",
         f"{-a9.loc['0.8-1.0', 'cloud_minus_edge_mean']:.1f}; {-a9.loc[cin, 'cloud_minus_edge_mean'].min():.1f}",
         "The >1.0 bin is 4.6 W/m2; 'about 4.3' fits the 0.8-1.0 bin."),
        ("5.4", "the gate spends most of its budget where the gain is largest", "yes", A9, "share_of_escalations_mean, kt < 0.8",
         a9.loc[["0.0-0.2"] + cin, "share_of_escalations_mean"].sum(), yn(a9.loc[["0.0-0.2"] + cin, "share_of_escalations_mean"].sum() > 0.5), ""),
        ("5.4", "reproduce the stored decisions exactly", "True", PH + "posthoc_summary.json", "decision_check.reproduces_sweep",
         ph["decision_check"]["reproduces_sweep"], str(ph["decision_check"]["reproduces_sweep"]), ""),
        ("5.5", "changes by at least 0.25 from the previous horizon, 30 min earlier (2,551 in 2016)", "0.25; 2,551", BY + "`; `" + F,
         "ramp.delta_kt; ramp_cells", f"{cfg['ramp']['delta_kt']}; {cl['ramp_cells']}", f"{cfg['ramp']['delta_kt']}; {cl['ramp_cells']:,}", ""),
        ("5.5", "140.1 versus 145.9 W/m2, seed means", "140.1; 145.9", F, "mean(claim5_ramp.rmse_cloud_ramp); mean(rmse_edge_ramp)",
         f"{ramp_c}; {ramp_e}", f"{f1(ramp_c)}; {f1(ramp_e)}", ""),
        ("5.5", "with the interval above zero for four of five seeds", "4", F, "claim5 intervals lo > 0", n5, str(n5), ""),
        ("5.5", "The best gate keeps 0.48 of the ramp gain at 25% (uncertainty gate, seed mean)", "0.48", F,
         "claim5_ramp.gate_share_at_25pct_ramp.uncertainty.share_mean", c5["gate_share_at_25pct_ramp"]["uncertainty"]["share_mean"],
         f2(c5["gate_share_at_25pct_ramp"]["uncertainty"]["share_mean"]), ""),
        ("5.5", "the uncertainty gate reaches 0.5 in two of five seeds and the learned gate in one", "2; 1", T + "gates/sweep_test.csv",
         "share_retained >= 0.5 [ramp, cloud, 0.25]", f"{ramp_ge['uncertainty']}; {ramp_ge['learned']}",
         f"{ramp_ge['uncertainty']}; {ramp_ge['learned']}", ""),
        ("5.5", "the code committed before the test used the seed mean and excluded the oracle", "yes", CC, "Provenance of the claim-5b logic",
         "b706593, before tag 3c07703", "yes", ""),
        ("5.6", "its RMSE rises from 76.71 to 77.21 W/m2 (seed range 75.8 to 81.0 W/m2)", "76.71; 77.21; 75.8-81.0", M,
         "RMSE [edge_fp32]; [edge_int8 (secondary)] mean, min, max",
         f"{re_}; {rm.loc['edge_int8 (secondary)', 'mean']}",
         f"{f2(re_)}; {f2(rm.loc['edge_int8 (secondary)', 'mean'])}; {f1(rm.loc['edge_int8 (secondary)', 'min'])}-{f1(rm.loc['edge_int8 (secondary)', 'max'])}", ""),
        ("5.6", "a loss of 8% of the edge-to-cloud gain", "8%", M, "(RMSE_int8 - RMSE_fp32) / (RMSE_edge - RMSE_cloud)", int8_loss, p0(int8_loss),
         "Derived."),
        ("5.6", "7.5 µs for the fp32 edge network, 9.6 µs for the int8 version, 8.6 µs for the uncertainty gate, 8.5 µs for the learned gate "
         "and 3.5 µs for the compiled capped trees", "7.5; 9.6; 8.6; 8.5; 3.5", G, "latency_*.median_us",
         "", "; ".join(f1(x) for x in (ed["latency_fp32"]["median_us"], ed["latency_int8"]["median_us"],
                                       ei["uncertainty"]["latency_int8"]["median_us"], ei["learned"]["latency_int8"]["median_us"],
                                       ms["trees_ground_cap10x"]["generated_c"]["latency_single_row"]["median_us"])), ""),
        ("5.6", "With whole days offline at random with probability 0.25 (200 draws), the uncertainty gate at a 25% target keeps 0.514",
         "0.514", A4, "share_kept_day_block [uncertainty, 0.25, 0.25]", a4u.loc[0.25, "share_kept_day_block"],
         f"{a4u.loc[0.25, 'share_kept_day_block']:.3f}", "200 draws: configs/analyses.yaml outage.block_draws."),
        ("5.6", "against 0.688 with the link always available; independent refusals with the same probability give 0.512", "0.688; 0.512", A4,
         "share_kept_day_block [.., 0.0]; share_kept_independent [.., 0.25]",
         f"{a4u.loc[0.0, 'share_kept_day_block']}; {a4u.loc[0.25, 'share_kept_independent']}",
         f"{a4u.loc[0.0, 'share_kept_day_block']:.3f}; {a4u.loc[0.25, 'share_kept_independent']:.3f}", ""),
        ("5.6", "This analysis was named in the plan before the test run, but its setup was fixed after it.", "yes", MF, "item 11",
         "PLAN.md:57 at tag; configs/analyses.yaml fixed after the run", "yes", ""),
        ("6", "The input effect changed from −0.27% on the 2015 validation half-year to 4.2% on 2016", "-0.27%; 4.2%",
         "results/tiers/control_effects.json`; `" + F, "input effect validation; test", f"{ce}; {c1['input_effect']}",
         f"{100 * ce:.2f}%; {p1(c1['input_effect'])}", ""),
        ("6", "an exploratory split of 2016 did not explain why", "both halves positive", A + "exploratory_halfyear_claims.json",
         "halves.H1/H2.claim1.input_effect", f"{ex['H1']['claim1']['input_effect']}; {ex['H2']['claim1']['input_effect']}",
         "both halves positive" if min(ex["H1"]["claim1"]["input_effect"], ex["H2"]["claim1"]["input_effect"]) > 0 else "mixed",
         "Interpretation; both halves show a positive input effect (2.8%, 8.8%), so the split does not reproduce -0.27%."),
        ("6", "The 434 daytime issue times without a satellite frame are outside the primary evaluation.", "434", T + "data_quality.json`; `"
         + T + "gates/meta.json", "daylight_issue_times - issue_times_test", n_day - n_dec, str(n_day - n_dec), ""),
        ("7", "a 43.8 kB on-device network with a 6.1 kB gate kept 69% of the gain", "43.8 kB; 6.1 kB; 69%", G + "`; `" + S,
         "fp32_onnx_bytes; gates int8_onnx_bytes; share_mean [uncertainty, 0.25]",
         f"{ed['fp32_onnx_bytes']}; {ei['uncertainty']['int8_onnx_bytes']}; {sb[0.25].loc['uncertainty', 'share_mean']}",
         f"{kb(ed['fp32_onnx_bytes'])}; {kb(ei['uncertainty']['int8_onnx_bytes'])}; {p0(sb[0.25].loc['uncertainty', 'share_mean'])}", ""),
        ("7", "while escalating 23% of requests", "23%", S, "realised_rate_mean [uncertainty, 0.25]",
         sb[0.25].loc["uncertainty", "realised_rate_mean"], p0(sb[0.25].loc["uncertainty", "realised_rate_mean"]), ""),
        ("7", "size-capped ground-only trees were 4.5% better than the on-device network", "4.5%", F + "`; `" + M,
         "1 - rmse_cap10x / edge", 1 - c2["rmse_cap10x"] / re_, p1(1 - c2["rmse_cap10x"] / re_), ""),
        ("Fig. 2 caption", "bands: seed range over five seeds", "yes", "scripts/paper_figures.py", "fig_budget_sweep fill_between RMSE_min..RMSE_max",
         "seed min..max", "yes", ""),
        ("Fig. 3 caption", "mean of five seeds", "yes", "scripts/planned_analyses.py", "A3 esc_share = mean over seeds", "mean over 5 seeds", "yes", ""),
    ]
    # each quoted snippet must be in the extracted paper (Table 2 rows are checked cell by cell)
    norm = lambda t: " ".join(t.replace("–", "-").split())  # noqa: E731
    for sec, w, *_ in R:
        if sec.startswith("Table"):   # table rows: each cell separately
            assert all(norm(c) in norm(text) for c in re.split(r" \| | / ", w)), w
        elif sec != "Abstract" or w != "Of the five claims, two were refuted and one partly":
            assert norm(w) in norm(text), w
    yes_vals = {"yes", "True", "matches", "same numbers"}
    out = ["", "## Numbers and statements in the full paper", "",
           f"Source: `{P}`, extracted read-only from `ICONI2026_Erniyazov_Lim_full.docx` (python-docx; body, tables and "
           "captions). Each 'as written' snippet occurs verbatim in that file. 'matches' compares the stored value, rounded as "
           "the paper rounds it, with the paper's value, or records a check against `docs/claim_criteria.md` / "
           "`docs/method_facts.md`. Literature statements (Sections 1-2) are not traced.", "",
           "| section | as written | paper value | file | field | stored value | rounded / check | matches | note |",
           "|---|---|---|---|---|---|---|---|---|"]
    n_ok = 0
    for sec, w, exp, f, fld, v, rv, note in R:
        ok = rv == exp or (exp == "yes" and rv in yes_vals)
        n_ok += ok
        sv = repr(float(v)) if isinstance(v, (float, np.floating)) else str(v)
        cell = lambda t: str(t).replace("|", "\\|")  # noqa: E731
        out.append(f"| {sec} | {cell(w)} | {cell(exp)} | `{f}` | `{cell(fld)}` | {cell(sv)} | {cell(rv)} | "
                   f"{'yes' if ok else '**NO**'} | {cell(note)} |")
    out += ["", f"**{n_ok} of {len(R)} rows match; {len(R) - n_ok} do not.**", ""]

    # (a) citations
    cites = []
    for m in re.finditer(r"\[(\d+)\](?:\s*[–-]\s*\[(\d+)\])?", body.replace("[image]", "")):
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        cites += list(range(a, b + 1))
    first = list(dict.fromkeys(cites))
    refs = re.findall(r"^\[(\d+)\] ", text, flags=re.M)
    out += ["### (a) In-text citations", "",
            f"- Reference list: {len(refs)} entries (numbers restored from Word list order).",
            f"- Order of first appearance: {', '.join(f'[{n}]' for n in first)}.",
            f"- In order 1 to {len(refs)}: {'yes' if first == list(range(1, len(refs) + 1)) else '**no**'}; "
            f"every reference cited: {'yes' if set(first) == set(range(1, len(refs) + 1)) else '**no**'}.", ""]
    # (b) figures and tables
    lines = text.splitlines()
    out += ["### (b) Figures and tables", "", "| item | first mention (line) | caption (line) | mentioned before caption | picture |",
            "|---|---|---|---|---|"]
    for item in ("Fig. 1", "Fig. 2", "Fig. 3", "Table 1", "Table 2"):
        cap = next(i for i, ln in enumerate(lines, 1) if ln.startswith(item + "."))
        men = next((i for i, ln in enumerate(lines, 1) if item in ln and not ln.startswith(item + ".")), None)
        near = " ".join(lines[max(0, cap - 4):cap - 1])   # the three lines above the caption
        ins = re.search(r"\[INSERT[^\]]*\]", near)
        pic = "table" if item.startswith("Table") else ("placeholder: " + ins.group(0) if ins else
                                                       ("image above caption" if "[image]" in near else "none"))
        out.append(f"| {item} | {men} | {cap} | {'yes' if men and men < cap else '**no**'} | {pic} |")
    out += ["", "### (c) Terms and units (whole-word counts in the body, references excluded)", "",
            "| term | count | lines |", "|---|---|---|"]
    terms = [("edge tier", r"\bedge tiers?\b"), ("edge network", r"\bedge network\b"), ("on-device network", r"\bon-device network\b"),
             ("cloud tier", r"\bcloud tiers?\b"), ("cloud forecaster", r"\bcloud forecaster\b"), ("share kept", r"\bshare kept\b"),
             ("keeps / kept (of the gain)", r"\b(?:keeps|kept)\b[^.]{0,40}\bgain\b"), ("retains / retained", r"\bretain(?:s|ed)?\b"),
             ("Kept (table column)", r"\| Kept \|"), ("realised rate", r"\brealised (?:escalation )?rates?\b"),
             ("realized (US spelling)", r"\breali[sz]ed\b(?<!realised)"), ("Rate (table column)", r"\| Rate \|"),
             ("W/m2", r"W/m2"), ("W/m²", r"W/m²"), ("kB", r"\d kB\b|\bkB\b"), ("KB (upper case)", r"\bKB\b"),
             ("µs", r"µs"), ("ms (milliseconds)", r"\d\s?ms\b")]
    blines = body.splitlines()
    for lab, rx in terms:
        hits = [i for i, ln in enumerate(blines, 1) if re.search(rx, ln, flags=0 if lab[0].isupper() or "/" in lab or lab in ("kB", "µs") else re.I)]
        cnt = sum(len(re.findall(rx, ln, flags=0 if lab[0].isupper() or "/" in lab or lab in ("kB", "µs") else re.I)) for ln in blines)
        out.append(f"| {lab} | {cnt} | {', '.join(map(str, hits)) if hits else '-'} |")
    out.append("")
    return out


def main() -> None:
    T = "results/test/"
    dq, cl = j(T + "data_quality.json"), j(T + "claims_test.json")
    # data quality
    s = "Data quality (2016)"
    add(s, "issue times", dq["issue_times"], T + "data_quality.json", "issue_times")
    add(s, "daylight issue times", dq["daylight_issue_times"], T + "data_quality.json", "daylight_issue_times")
    add(s, "primary cells", dq["primary_cells"], T + "data_quality.json", "primary_cells")
    add(s, "satellite available, 15-min rule (share of daylight issue times)",
        dq["satellite_available_15min_share_of_daylight_issue_times"], T + "data_quality.json",
        "satellite_available_15min_share_of_daylight_issue_times", "{:.3f}")
    add(s, "NAM missing, 30-min horizon (share of daylight cells)", dq["nam_coverage"]["30min"]["share_missing"],
        T + "data_quality.json", "nam_coverage.30min.share_missing", "{:.3f}")
    add(s, "irradiance missing minutes", dq["irradiance_minutes"]["missing_minutes"], T + "data_quality.json",
        "irradiance_minutes.missing_minutes")
    # claims
    F = T + "claims_test.json"
    c1 = cl["claim1_input_effect"]
    s = "Claim 1, input effect (2016)"
    add(s, "input effect (1 - RMSE_all/RMSE_ground)", c1["input_effect"], F, "claim1_input_effect.input_effect")
    add(s, "RMSE trees all inputs", c1["rmse_trees_all"], F, "claim1_input_effect.rmse_trees_all", "{:.2f}")
    add(s, "RMSE trees ground only", c1["rmse_trees_ground"], F, "claim1_input_effect.rmse_trees_ground", "{:.2f}")
    for k in ("point", "lo", "hi"):
        add(s, f"interval RMSE_ground - RMSE_all, {k}", c1["interval_rmse_ground_minus_all"][k], F,
            f"claim1_input_effect.interval_rmse_ground_minus_all.{k}", "{:.2f}")
    add(s, "verdict counts against", c1["counts_against"], F, "claim1_input_effect.counts_against")
    c2 = cl["claim2_on_device_sufficiency"]
    s = "Claim 2, on-device sufficiency (2016)"
    add(s, "RMSE 10x-capped ground trees", c2["rmse_cap10x"], F, "claim2_on_device_sufficiency.rmse_cap10x", "{:.2f}")
    add(s, "RMSE cloud tier, seed mean", np.mean(list(c2["rmse_cloud_per_seed"].values())), F,
        "mean(claim2_on_device_sufficiency.rmse_cloud_per_seed)", "{:.2f}")
    add(s, "relative difference d", c2["relative_difference_d"], F, "claim2_on_device_sufficiency.relative_difference_d")
    add(s, "seeds with interval entirely above zero", c2["seeds_entirely_above_zero"], F,
        "claim2_on_device_sufficiency.seeds_entirely_above_zero")
    add(s, "verdict", c2["verdict"], F, "claim2_on_device_sufficiency.verdict")
    c3 = cl["claim3_edge_vs_cloud_fp32"]
    s = "Claim 3, edge vs cloud (2016)"
    add(s, "gain (seed-mean RMSE)", c3["gain_mean_of_seed_rmse"], F, "claim3_edge_vs_cloud_fp32.gain_mean_of_seed_rmse")
    add(s, "gain, median seed", c3["gain_median_seed"], F, "claim3_edge_vs_cloud_fp32.gain_median_seed")
    add(s, "gain, min seed", min(c3["gain_per_seed"]), F, "min(claim3_edge_vs_cloud_fp32.gain_per_seed)")
    add(s, "gain, max seed", max(c3["gain_per_seed"]), F, "max(claim3_edge_vs_cloud_fp32.gain_per_seed)")
    add(s, "seeds whose interval includes zero", c3["seeds_including_zero_or_below"], F,
        "claim3_edge_vs_cloud_fp32.seeds_including_zero_or_below")
    add(s, "verdict counts against", c3["counts_against"], F, "claim3_edge_vs_cloud_fp32.counts_against")
    add(s, "gain vs int8 edge (secondary)", cl["claim3_edge_vs_cloud_int8_secondary"]["gain_mean_of_seed_rmse"], F,
        "claim3_edge_vs_cloud_int8_secondary.gain_mean_of_seed_rmse")
    c4 = cl["claim4_gates"]
    s = "Claim 4, gates (2016)"
    for r in c4["table"]:
        if r["gate"] in ("uncertainty", "learned", "variability", "random", "fixed_interval", "oracle"):
            add(s, f"{r['gate']} @ {int(r['budget_target'] * 100)}%: share kept (seed mean)", r["share_mean"], F,
                f"claim4_gates.table[gate={r['gate']},budget={r['budget_target']}].share_mean", "{:.3f}")
            add(s, f"{r['gate']} @ {int(r['budget_target'] * 100)}%: realised rate", r["realised_rate_mean"], F,
                f"claim4_gates.table[gate={r['gate']},budget={r['budget_target']}].realised_rate_mean", "{:.3f}")
    add(s, "verdict: against 'gate beats random'", c4["against_gate_beats_random"], F, "claim4_gates.against_gate_beats_random")
    add(s, "verdict: against 'learned not better'", c4["against_learned_not_better"], F, "claim4_gates.against_learned_not_better")
    add(s, "verdict: against 'refit fixes shortfall'", c4["against_refit_fixes_shortfall"], F,
        "claim4_gates.against_refit_fixes_shortfall")
    add(s, "max abs(realised - target) over score gates", max(abs(v) for v in c4["realised_minus_target"].values()), F,
        "max(abs(claim4_gates.realised_minus_target))", "{:.3f}")
    c5 = cl["claim5_ramp"]
    s = "Claim 5, ramps (2016)"
    add(s, "ramp cells", cl["ramp_cells"], F, "ramp_cells")
    add(s, "RMSE edge on ramps, seed mean", np.mean(list(c5["rmse_edge_ramp"].values())), F, "mean(claim5_ramp.rmse_edge_ramp)", "{:.2f}")
    add(s, "RMSE cloud on ramps, seed mean", np.mean(list(c5["rmse_cloud_ramp"].values())), F, "mean(claim5_ramp.rmse_cloud_ramp)", "{:.2f}")
    add(s, "seeds with ramp interval entirely above zero",
        sum(i["lo"] > 0 for i in c5["intervals_rmse_edge_minus_cloud_ramp"]), F, "count(claim5_ramp.intervals...lo>0)")
    add(s, "verdict: against 'no cloud advantage on ramps'", c5["against_no_cloud_advantage"], F, "claim5_ramp.against_no_cloud_advantage")
    add(s, "uncertainty gate share kept on ramps @25%", c5["gate_share_at_25pct_ramp"]["uncertainty"]["share_mean"], F,
        "claim5_ramp.gate_share_at_25pct_ramp.uncertainty.share_mean", "{:.3f}")
    add(s, "verdict: against 'no gate helps on ramps'", c5["against_no_gate_helps"], F, "claim5_ramp.against_no_gate_helps")
    # models table
    M = T + "claims_models_test.csv"
    cm = pd.read_csv(M)
    s = "Models on identical rows (2016)"
    for rs, model in [("primary", "smart_persistence"), ("primary", "lasso_endo"), ("primary_anchor_rows", "lasso_exo"),
                      ("primary", "edge_fp32"), ("primary", "edge_int8 (secondary)"), ("primary", "cloud"),
                      ("primary", "trees_ground"), ("primary", "trees_all"), ("primary", "trees_ground_cap10x")]:
        d = cm[(cm.row_set == rs) & (cm.model == model)]
        add(s, f"{model} RMSE ({rs}, seed mean)", d["RMSE"].mean(), M, f"mean(RMSE) where row_set={rs}, model={model}", "{:.2f}")
        add(s, f"{model} skill ({rs})", d["skill"].mean(), M, f"mean(skill) where row_set={rs}, model={model}", "{:.3f}")
    add(s, "anchor-row cells", cl["anchor_cells"], F, "anchor_cells")
    # footprint
    ms = j("results/footprint/measured.json")
    G = "results/footprint/measured.json"
    s = "Footprint (PC CPU; nothing on a microcontroller)"
    e0 = ms["edge_network"]["per_seed"][0]
    add(s, "edge int8 ONNX size (B)", e0["int8_onnx_bytes"], G, "edge_network.per_seed[0].int8_onnx_bytes")
    add(s, "edge fp32 ONNX size (B)", e0["fp32_onnx_bytes"], G, "edge_network.per_seed[0].fp32_onnx_bytes")
    add(s, "edge int8 latency median (us)", e0["latency_int8"]["median_us"], G, "edge_network.per_seed[0].latency_int8.median_us", "{:.1f}")
    add(s, "edge int8 latency p95 (us)", e0["latency_int8"]["p95_us"], G, "edge_network.per_seed[0].latency_int8.p95_us", "{:.1f}")
    add(s, "gate int8 ONNX size (B)", ms["gates"]["per_seed"][0]["uncertainty"]["int8_onnx_bytes"], G, "gates.per_seed[0].uncertainty.int8_onnx_bytes")
    tc = ms["trees_ground_cap10x"]
    add(s, "10x-capped trees, compiled C object (B)", tc["generated_c"]["object_bytes"], G, "trees_ground_cap10x.generated_c.object_bytes")
    add(s, "10x-capped trees, estimate (B)", tc["estimate_bytes_12_per_node"], G, "trees_ground_cap10x.estimate_bytes_12_per_node")
    add(s, "10x-capped trees, C latency median (us)", tc["generated_c"]["latency_single_row"]["median_us"], G,
        "trees_ground_cap10x.generated_c.latency_single_row.median_us", "{:.1f}")
    add(s, "10x-capped trees, ONNX size (B)", tc["onnx"]["total_bytes"], G, "trees_ground_cap10x.onnx.total_bytes")
    add(s, "edge parameters", j("results/tiers/params.json")["edge"], "results/tiers/params.json", "edge")
    d8 = np.mean([r["rmse_int8"] - r["rmse_fp32"] for r in ms["edge_network"]["per_seed"]])
    add(s, "int8 minus fp32 edge RMSE, validation (W/m2, seed mean)", d8, G, "mean(rmse_int8 - rmse_fp32)", "{:.2f}")
    # validation counterparts
    s = "Validation counterparts (development, 2015 H2)"
    gg = [v for v in j("results/go_no_go.json")["variants"] if v["name"] == "step4_gbt_plus_net"][0]
    add(s, "go/no-go gain, step-4 average", gg["gain_avg"], "results/go_no_go.json", "variants[step4_gbt_plus_net].gain_avg")
    ce = j("results/tiers/control_effects.json")["primary"]
    add(s, "input effect (validation)", ce["input_effect (A = trees_all, B = trees_ground)"]["relative"]["mean"],
        "results/tiers/control_effects.json", "primary.input_effect.relative.mean")
    add(s, "model effect (validation)", ce["model_effect (A = trees_ground, B = edge_network)"]["relative"]["mean"],
        "results/tiers/control_effects.json", "primary.model_effect.relative.mean")
    dr = j("results/dryrun/claims_validation.json")
    add(s, "claim-2 d (validation)", dr["claim2_on_device_sufficiency"]["relative_difference_d"],
        "results/dryrun/claims_validation.json", "claim2_on_device_sufficiency.relative_difference_d")
    sa = j("results/tiers/satellite_availability.json")
    add(s, "satellite available, 15-min rule (validation)", sa["lag_15min_primary"]["overall_share_of_daylight_issue_times"],
        "results/tiers/satellite_availability.json", "lag_15min_primary.overall_share_of_daylight_issue_times", "{:.3f}")
    # planned analyses
    A = "results/analyses/"
    s = "Planned analyses A1-A7 (2016)"
    a1 = pd.read_csv(A + "a1_gate_oracle_gap.csv")
    r = a1[(a1.gate == "uncertainty") & (a1.budget_target == 0.25)].iloc[0]
    add(s, "A1 uncertainty @25%: share of random-to-oracle gap closed", r["share_of_random_to_oracle_gap_closed"],
        A + "a1_gate_oracle_gap.csv", "share_of_random_to_oracle_gap_closed [uncertainty, 0.25]", "{:.3f}")
    a4 = pd.read_csv(A + "a4_link_outage_summary.csv")
    r = a4[(a4.gate == "uncertainty") & (a4.budget_target == 0.25) & (a4.outage_fraction == 0.25)].iloc[0]
    add(s, "A4 uncertainty @25%, 25% day-block outages: share kept", r["share_kept_day_block"],
        A + "a4_link_outage_summary.csv", "share_kept_day_block [uncertainty, 0.25, 0.25]", "{:.3f}")
    a5 = pd.read_csv(A + "a5_gate_differences_summary.csv")
    r = a5[(a5.gate_a == "uncertainty") & (a5.gate_b == "learned") & (a5.budget_target == 0.25)].iloc[0]
    add(s, "A5 uncertainty - learned @25%: seeds excluding zero", int(r["seeds_excluding_zero"]),
        A + "a5_gate_differences_summary.csv", "seeds_excluding_zero [uncertainty, learned, 0.25]")
    r = a5[(a5.gate_a == "uncertainty") & (a5.gate_b == "random") & (a5.budget_target == 0.25)].iloc[0]
    add(s, "A5 uncertainty - random @25%: seeds excluding zero", int(r["seeds_excluding_zero"]),
        A + "a5_gate_differences_summary.csv", "seeds_excluding_zero [uncertainty, random, 0.25]")
    ps = j(A + "planned_summary.json")
    sp = [v for k, v in ps["a7_spearman_and_fit"].items() if not k.endswith("_fit")]
    add(s, "A7 uncertainty calibration Spearman, min over seeds", min(sp), A + "planned_summary.json", "min(a7_spearman_and_fit[seed])", "{:.3f}")
    add(s, "A7 uncertainty calibration Spearman, max over seeds", max(sp), A + "planned_summary.json", "max(a7_spearman_and_fit[seed])", "{:.3f}")
    a2 = pd.read_csv(A + "a2_per_horizon.csv")
    add(s, "A2 edge-to-cloud gain at 30 min", a2.loc[a2.horizon == "30min", "edge_to_cloud_gain"].iloc[0], A + "a2_per_horizon.csv", "edge_to_cloud_gain [30min]", "{:.3f}")
    add(s, "A2 edge-to-cloud gain at 180 min", a2.loc[a2.horizon == "180min", "edge_to_cloud_gain"].iloc[0], A + "a2_per_horizon.csv", "edge_to_cloud_gain [180min]", "{:.3f}")
    # post hoc A8-A11: every value in every file
    P = A + "posthoc/"
    s = "POST HOC, not pre-specified: A8-A11 (2016)"
    for stem, key in (("a8_per_horizon_rmse", "horizon"), ("a9_escalation_by_kt", "kt_bin"),
                      ("a10_target_vs_realised_rate", None)):
        d = pd.read_csv(P + stem + ".csv", comment="#")
        tag = stem.split("_")[0].upper()
        for _, r in d.iterrows():
            rk = r[key] if key else f"{r['gate']}, {r['budget_target']:.2f}"
            for c in d.columns:
                if c in (key, "gate", "budget_target"):
                    continue
                add(s, f"{tag} [{rk}] {c}", r[c] if c not in ("issue_times", "cells") else int(r[c]),
                    P + stem + ".csv", f"{c} [{rk}]")
    ht = pd.read_csv("results/paper/horizon_table.csv", comment="#")
    for r in ht.itertuples():
        add(s, f"horizon table [{r.horizon}] cloud gain vs edge (%)", r.cloud_gain_vs_edge_pct,
            "results/paper/horizon_table.csv", f"cloud_gain_vs_edge_pct [{r.horizon}] = (1 - cloud_mean / edge_fp32_mean) * 100 from A8",
            "{:.1f}")
    ph = j(P + "posthoc_summary.json")
    add(s, "A9 recomputed decisions reproduce sweep_test.csv", ph["decision_check"]["reproduces_sweep"],
        P + "posthoc_summary.json", "decision_check.reproduces_sweep")
    add(s, "A11 int8 vs fp32 gate agreement @25%", "not available", P + "posthoc_summary.json", "a11")
    # exploratory
    X = A + "exploratory_halfyear_claims.json"
    ex = j(X)["halves"]
    s = "EXPLORATORY: half-year split (chosen after seeing test results)"
    for h in ("H1", "H2"):
        add(s, f"{h} input effect", ex[h]["claim1"]["input_effect"], X, f"halves.{h}.claim1.input_effect")
        add(s, f"{h} claim-2 d", ex[h]["claim2"]["relative_difference_d"], X, f"halves.{h}.claim2.relative_difference_d")
        add(s, f"{h} claim-2 verdict rule applied", ex[h]["claim2"]["verdict_rule_applied"], X, f"halves.{h}.claim2.verdict_rule_applied")
        add(s, f"{h} edge-to-cloud gain", ex[h]["claim3"]["gain_mean_of_seed_rmse"], X, f"halves.{h}.claim3.gain_mean_of_seed_rmse")

    lines = ["# Number trace", "",
             "Every number that may appear in the paper, with the file and field it comes from. Generated by "
             "`scripts/number_trace.py` (`./run.sh trace`): each value below is read from its file at generation "
             "time, not typed by hand. RMSE in W/m2; shares and gains as fractions.", ""] + abstract_section() + paper_full_section()
    cur = None
    for sec, q, v, f, fld in ROWS:
        if sec != cur:
            lines += ["", f"## {sec}", "", "| quantity | value | file | field |", "|---|---|---|---|"]
            cur = sec
        lines.append(f"| {q} | {v} | `{f}` | `{fld}` |")
    Path("docs/number_trace.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(ROWS)} numbers traced")


if __name__ == "__main__":
    main()
