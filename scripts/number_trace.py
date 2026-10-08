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
        ("development: cloud-side inputs appeared to add nothing", CE, "primary.input_effect.relative.mean", ie_val, "check",
         yes(ie_val <= 0), "yes", "RMSE of trees on ground inputs (validation)", f"Validation input effect {100 * ie_val:+.2f}% (2015 H2)."),
        ("development: no reliable cloud advantage on ramps", DR,
         "claim5_ramp.against_no_cloud_advantage; claim5_ramp.intervals_rmse_edge_minus_cloud_ramp",
         f"{dr5['against_no_cloud_advantage']}; {ramp_dev_zero} of 5 include zero", "check",
         yes(not dr5["against_no_cloud_advantage"] and ramp_dev_zero == 5), "yes", "",
         f"Validation ramp RMSE cloud {np.mean(list(dr5['rmse_cloud_ramp'].values())):.1f} against edge "
         f"{np.mean(list(dr5['rmse_edge_ramp'].values())):.1f} W/m2: slightly better, interval includes zero for all seeds."),
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
             "time, not typed by hand. RMSE in W/m2; shares and gains as fractions.", ""] + abstract_section()
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
