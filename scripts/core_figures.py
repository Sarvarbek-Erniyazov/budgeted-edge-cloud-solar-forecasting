"""Stage 9.3: core figures (a) architecture, (b) RMSE vs escalation budget, (c) routing map, and the core
table. Reads results/ files only. Writes figures/core/*.{svg,png} and results/paper/core_table.{csv,md}."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from escal import plotstyle as S

FIG = Path("figures/core")
PAPER = Path("results/paper")


def fig_architecture() -> list[str]:
    fig, ax = plt.subplots(figsize=(8.9, 3.2))
    ax.set_xlim(0, 14.1)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=S.MUTED, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color=S.INK,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=S.MUTED))

    for x, w, label in ((0.1, 6.9, "On device (PV site)"), (9.1, 4.9, "Cloud")):
        ax.add_patch(FancyBboxPatch((x, 0.25), w, 3.7, boxstyle="round,pad=0.02,rounding_size=0.1", fc="none",
                                    ec=S.MUTED, lw=0.8, ls=(0, (3, 2))))
        ax.text(x + 0.15, 3.72, label, fontsize=8, color=S.MUTED, fontweight="bold")
    box(0.3, 2.25, 2.05, 1.1, "Ground sensors\nGHI, DNI, weather\n+ clear-sky terms", "#f2f2f2")
    box(2.6, 2.25, 2.05, 1.1, "Edge tier, MLP\n43.8 kB\n(int8: 15.2 kB)", "#cfe3f3", bold=True)
    box(4.9, 2.25, 1.9, 1.1, "Gate\nint8, 6.1 kB\nescalate or not", "#fde2c8", bold=True)
    box(2.6, 0.5, 4.2, 0.95, "Forecast, 30 to 180 min ahead\n(edge, or cloud if escalated)", "#f2f2f2")
    box(11.2, 2.25, 2.6, 1.1, "Cloud tier\ntrees + network\n(averaged in kt)", "#cfe3f3", bold=True)
    box(11.2, 0.5, 2.6, 0.95, "GOES-15 tile\nNAM, 4 nodes", "#f2f2f2")
    arrow(2.35, 2.8, 2.6, 2.8)
    arrow(4.65, 2.8, 4.9, 2.8)
    arrow(3.6, 2.25, 3.6, 1.45)
    arrow(6.8, 3.05, 11.2, 3.05)
    arrow(12.5, 1.45, 12.5, 2.25)
    arrow(11.2, 2.45, 6.8, 1.05)
    # labels sit in the gap between the two groups, clear of every box, border and arrow
    ax.text(8.05, 3.17, "request\n(share b of\nissue times)", ha="center", va="bottom", fontsize=7, color=S.MUTED)
    ax.text(8.05, 2.05, "cloud forecast", ha="center", va="center", fontsize=7, color=S.MUTED)
    bad = S.layout_problems(fig)
    assert not bad, bad
    return S.save(fig, FIG / "architecture")


def fig_budget_sweep() -> list[str]:
    summ = pd.read_csv("results/test/gates/summary_test.csv")
    boot = pd.read_csv("results/test/gates/bootstrap_test.csv")
    s = summ[(summ.escalate_to == "cloud") & (summ.row_set == "primary")]
    b = boot[(boot.escalate_to == "cloud") & (boot.metric == "rmse")]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    r0 = s[(s.gate == "random") & (s.budget_target == 0.0)]["RMSE_mean"].iloc[0]
    r1 = s[(s.gate == "random") & (s.budget_target == 1.0)]["RMSE_mean"].iloc[0]
    for t in (0.10, 0.25, 0.50):
        ax.axvline(t, color=S.GRID, lw=1.0, ls=":", zorder=0)
    ax.axhline(r0, color=S.MUTED, lw=0.8, ls="--")
    ax.axhline(r1, color=S.MUTED, lw=0.8, ls="--")
    ax.text(0.62, r0 + 0.25, f"edge only {r0:.1f}", ha="left", va="bottom", fontsize=7, color=S.MUTED)
    ax.text(0.62, r1 - 0.3, f"cloud only {r1:.1f}", ha="left", va="top", fontsize=7, color=S.MUTED)
    for g in ["random", "fixed_interval", "variability", "uncertainty", "learned", "oracle"]:
        col, mk, ls = S.GATE_STYLE[g]
        d = s[s.gate == g].sort_values("budget_target")
        x, y = d["realised_rate_mean"].values, d["RMSE_mean"].values
        ax.fill_between(x, d["RMSE_min"].values, d["RMSE_max"].values, color=col, alpha=0.12, lw=0)
        ax.plot(x, y, color=col, ls=ls, lw=1.6, label=S.GATE_LABEL[g])
        rep = d[d.budget_target.isin([0.10, 0.25, 0.50])]
        bi = b[b.gate == g].groupby("budget_target")[["lo", "hi", "point"]].mean()
        yerr = np.vstack([rep["RMSE_mean"].values - bi.loc[rep.budget_target, "lo"].values,
                          bi.loc[rep.budget_target, "hi"].values - rep["RMSE_mean"].values])
        ax.errorbar(rep["realised_rate_mean"].values, rep["RMSE_mean"].values, yerr=yerr, fmt=mk, color=col, ms=6,
                    mec="white", mew=0.8, elinewidth=0.8, capsize=2, alpha=0.75)
    ax.set_xlabel("Realised escalation rate (share of issue times sent to the cloud); dotted lines: targets 10, 25, 50%")
    ax.set_ylabel("RMSE, W/m² (mean over 6 horizons)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("2016 test year: RMSE against escalation budget", loc="left")
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.text(0.0, -0.36, "Lines: seed mean; bands: range over 5 seeds; error bars at report budgets: 95% day-block "
                        "bootstrap interval of the RMSE (mean over seeds).", transform=ax.transAxes, fontsize=6.8, color=S.MUTED)
    return S.save(fig, FIG / "budget_sweep")


def fig_routing_map() -> list[str]:
    hm = pd.read_csv("results/analyses/a3_routing_hour_month.csv")
    piv = hm.pivot(index="hour", columns="month", values="share_escalated").sort_index()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.grid(False)
    im = ax.imshow(piv.values, aspect="auto", cmap=S.SEQ_CMAP, vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(piv.columns)), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"][:len(piv.columns)])
    ax.set_yticks(range(len(piv.index)), [f"{h:02d}" for h in piv.index])
    ax.set_xlabel("Month (2016)")
    ax.set_ylabel("Local hour (PST) of issue time")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Share of issue times escalated")
    cb.outline.set_edgecolor(S.MUTED)
    ax.set_title("Where the uncertainty gate escalates (25% budget, seed mean)", loc="left", fontsize=9)
    return S.save(fig, FIG / "routing_map")


def core_table() -> pd.DataFrame:
    cm = pd.read_csv("results/test/claims_models_test.csv")
    prim = cm[cm.row_set == "primary"]
    anch = cm[cm.row_set == "primary_anchor_rows"]
    ms = json.loads(Path("results/footprint/measured.json").read_text())
    tv = json.loads(Path("results/freeze/trees_verification.json").read_text())
    summ = pd.read_csv("results/test/gates/summary_test.csv")
    s = summ[(summ.escalate_to == "cloud") & (summ.row_set == "primary")]
    lat_e = ms["edge_network"]["per_seed"][0]
    lat_g = ms["gates"]["per_seed"][0]
    c = ms["trees_ground_cap10x"]
    re_ = prim[prim.model == "edge_fp32"]["RMSE"].mean()
    rc_ = prim[prim.model == "cloud"]["RMSE"].mean()

    def tier(model, label, rows, size, lat, rate, target_like=True):
        d = rows[rows.model == model]
        r = d["RMSE"].mean()
        return {"group": "tier/anchor", "name": label, "rows": "primary" if rows is prim else "anchor rows",
                "RMSE": r, "RMSE_seed_min": d["RMSE"].min(), "RMSE_seed_max": d["RMSE"].max(),
                "skill": d["skill"].mean(),
                "share_kept": (re_ - r) / (re_ - rc_) if (rows is prim and target_like) else np.nan,
                "realised_rate": rate, "size_bytes": size, "cpu_latency_median_us": lat}
    rows = [
        tier("smart_persistence", "Smart persistence", prim, 0, np.nan, np.nan, False),
        tier("lasso_endo", "Linear anchor, ground (lasso_endo)", prim, np.nan, np.nan, np.nan, False),
        tier("lasso_exo", "Linear anchor, ground + satellite (lasso_exo)", anch, np.nan, np.nan, np.nan, False),
        tier("edge_fp32", "Edge tier, fp32", prim, ms["edge_network"]["per_seed"][0]["fp32_onnx_bytes"],
             lat_e["latency_fp32"]["median_us"], 0.0),
        tier("edge_int8 (secondary)", "Edge tier, int8 (secondary)", prim, ms["edge_network"]["per_seed"][0]["int8_onnx_bytes"],
             lat_e["latency_int8"]["median_us"], 0.0),
        tier("trees_ground_cap10x", "Ground-only trees, 10x cap (C object)", prim, c["generated_c"]["object_bytes"],
             c["generated_c"]["latency_single_row"]["median_us"], np.nan),
        tier("trees_ground", "Ground-only trees, uncapped (pickle)", prim, tv["trees_ground"]["bytes"], np.nan, np.nan),
        tier("trees_all", "Trees, all inputs (pickle)", prim, tv["trees_all"]["bytes"], np.nan, np.nan),
        tier("cloud", "Cloud tier (trees + network)", prim, np.nan, np.nan, 1.0),
    ]
    for g in ["random", "fixed_interval", "variability", "uncertainty", "learned", "oracle"]:
        for bb in (0.10, 0.25, 0.50):
            d = s[(s.gate == g) & (s.budget_target == bb)].iloc[0]
            size = lat_g[g]["int8_onnx_bytes"] if g in ("uncertainty", "learned") else 0
            rows.append({"group": "gate", "name": f"{S.GATE_LABEL[g]} @ {int(bb * 100)}%", "rows": "primary",
                         "RMSE": d["RMSE_mean"], "RMSE_seed_min": d["RMSE_min"], "RMSE_seed_max": d["RMSE_max"],
                         "skill": d["skill_mean"], "share_kept": d["share_mean"], "realised_rate": d["realised_rate_mean"],
                         "size_bytes": size if g != "oracle" else np.nan,
                         "cpu_latency_median_us": lat_g[g]["latency_int8"]["median_us"] if g in ("uncertainty", "learned") else np.nan})
    t = pd.DataFrame(rows)
    PAPER.mkdir(parents=True, exist_ok=True)
    t.to_csv(PAPER / "core_table.csv", index=False, float_format="%.4f")
    md = ["# Core table (2016 test year)", "",
          "Generated by `scripts/core_figures.py`. Sources:",
          "- `results/test/claims_models_test.csv` and `results/test/gates/summary_test.csv`;",
          "- `results/footprint/measured.json` and `results/freeze/trees_verification.json`.",
          "",
          "RMSE is in W/m2, averaged over the 6 horizons; the RMSE is the mean over 5 seeds, with the seed range in "
          "brackets. Share kept = (RMSE_edge - RMSE) / (RMSE_edge - RMSE_cloud). Latency is single-row and "
          "one-thread on a PC CPU (ONNX Runtime or compiled C); nothing ran on a microcontroller. Gate latency is in addition to the edge "
          "tier. lasso_exo is on the anchor rows (29,948 cells), and every other row is on the primary rows (39,655 cells). "
          "Share kept is shown for tiers that could serve as an escalation target and for gates; n/a for the anchors. "
          "Cloud-tier size is not stated: the trees pickle alone is 5,917,302 B, plus the step-3 network.", "",
          "| name | RMSE [seed range] | skill | share kept | realised rate | size (B) | CPU latency (µs) |",
          "|---|---|---|---|---|---|---|"]

    def f(x, fmt):
        return "n/a" if pd.isna(x) else format(x, fmt)
    for r in rows:
        rng = "" if r["RMSE_seed_min"] == r["RMSE_seed_max"] else f" [{r['RMSE_seed_min']:.1f}-{r['RMSE_seed_max']:.1f}]"
        md.append(f"| {r['name']} | {r['RMSE']:.2f}{rng} | {r['skill']:.3f} | {f(r['share_kept'], '.2f')} | "
                  f"{f(r['realised_rate'], '.3f')} | {f(r['size_bytes'], ',.0f')} | {f(r['cpu_latency_median_us'], '.1f')} |")
    (PAPER / "core_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return t


def main() -> None:
    S.apply()
    files = fig_architecture() + fig_budget_sweep() + fig_routing_map()
    core_table()
    print("\n".join(files))


if __name__ == "__main__":
    main()
