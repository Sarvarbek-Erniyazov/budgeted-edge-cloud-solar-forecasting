"""Stage 9.5: paper versions of the core figures and the paper table. Reads results/ files only (nothing is
retrained or rerun). Figures are one template column wide (2.7 in), with no text below 7 pt and no in-figure
titles, saved as SVG and 600 dpi PNG in figures/paper/. The README keeps the figures/core/ versions.
Run: ./run.sh paper"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

from escal import plotstyle as S

FIG = Path("figures/paper")
PAPER = Path("results/paper")
WIDTH_IN, MIN_PT, DPI = 2.7, 7.0, 600
REPORT_BUDGETS = (0.10, 0.25, 0.50)


def apply_paper() -> None:
    S.apply()
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "lines.linewidth": 1.2, "lines.markersize": 4,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "svg.hashsalt": "paper"})   # stable SVG ids, so reruns give identical files


def save_paper(fig, name: str) -> list[str]:
    """Fixed width (no tight bbox), checked for small or colliding text before writing."""
    w = fig.get_size_inches()[0]
    assert abs(w - WIDTH_IN) < 1e-9, w
    bad = S.layout_problems(fig, min_pt=MIN_PT)
    assert not bad, bad
    FIG.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("svg", "png"):
        p = FIG / f"{name}.{ext}"
        fig.savefig(p, dpi=DPI, metadata={"Date": None} if ext == "svg" else None)
        out.append(p.as_posix())
    plt.close(fig)
    return out


SHORT_HEIGHT_IN = 2.1   # compact sweep for the 2-page paper: total height cap, legend included


def fig_budget_sweep(name: str = "budget_sweep", height: float = 3.1,
                     gates=("random", "fixed_interval", "variability", "uncertainty", "learned", "oracle"),
                     ylim: tuple[float, float] | None = None) -> list[str]:
    summ = pd.read_csv("results/test/gates/summary_test.csv")
    s = summ[(summ.escalate_to == "cloud") & (summ.row_set == "primary")]
    cm = pd.read_csv("results/test/claims_models_test.csv")
    cap = cm[(cm.row_set == "primary") & (cm.model == "trees_ground_cap10x")]["RMSE"].iloc[0]
    r0 = s[(s.gate == "random") & (s.budget_target == 0.0)]["RMSE_mean"].iloc[0]
    r1 = s[(s.gate == "random") & (s.budget_target == 1.0)]["RMSE_mean"].iloc[0]
    if ylim is not None:   # a fixed y-range must not clip any band, line or reference value that is drawn
        shown = s[s.gate.isin(gates)]
        lo, hi = min(shown.RMSE_min.min(), cap, r1), max(shown.RMSE_max.max(), cap, r0)
        assert ylim[0] <= lo and hi <= ylim[1], (ylim, lo, hi)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, height), layout="constrained")
    ax.axhline(r0, color=S.MUTED, lw=0.8, ls="--", label=f"Edge only ({r0:.1f})")
    ax.axhline(cap, color="#009E73", lw=0.9, ls="-.", label=f"Ground trees, 10x cap ({cap:.1f})")
    ax.axhline(r1, color=S.MUTED, lw=0.8, ls=":", label=f"Cloud only ({r1:.1f})")
    for g in gates:
        col, mk, ls = S.GATE_STYLE[g]
        d = s[s.gate == g].sort_values("budget_target")
        x, y = d["realised_rate_mean"].values, d["RMSE_mean"].values
        ax.fill_between(x, d["RMSE_min"].values, d["RMSE_max"].values, color=col, alpha=0.12, lw=0)
        rep = d[d.budget_target.isin(REPORT_BUDGETS)]
        ax.plot(x, y, color=col, ls=ls, label="Random" if g == "random" else S.GATE_LABEL[g])
        ax.plot(rep["realised_rate_mean"], rep["RMSE_mean"], mk, color=col, mec="white", mew=0.5)
    ax.set_xlabel("Realised escalation rate")
    ax.set_ylabel("RMSE (W/m²)")
    ax.set_xlim(-0.02, 1.02)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    h, lab = ax.get_legend_handles_labels()
    order = list(range(3, len(h))) + [0, 1, 2]   # gates first, then the three reference lines
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="outside lower center", ncol=2,
               handlelength=1.6, columnspacing=0.5, handletextpad=0.35, labelspacing=0.35)
    return save_paper(fig, name)


def fig_budget_sweep_short() -> list[str]:
    """Compact sweep for the 2-page paper: no oracle curve, y-axis fixed at 70-80 W/m2, 2.1 in tall with the
    legend inside the figure (constrained layout, no tight bbox, so the saved height is the figure height)."""
    return fig_budget_sweep("budget_sweep_short", SHORT_HEIGHT_IN,
                            ("random", "fixed_interval", "variability", "uncertainty", "learned"), (70.0, 80.0))


# Architecture geometry in inches: figure height, line spacing, the two dashed groups (y, h, label), the six boxes
# (x, y, w, h), the six arrows and the two gap labels (x, y). Box and label texts are shared by both versions.
ARCH_FULL = dict(
    H=3.2, ls=1.25, label_dy=0.12,
    groups=((1.42, 1.75, "On device (PV site)"), (0.04, 1.0, "Cloud")),
    boxes=((0.12, 2.2, 0.92, 0.66), (1.22, 2.2, 1.36, 0.66), (0.12, 1.52, 1.28, 0.5), (1.6, 1.52, 0.98, 0.5),
           (0.12, 0.13, 0.9, 0.62), (1.22, 0.13, 1.36, 0.62)),
    arrows=((1.06, 2.53, 1.2, 2.53), (2.09, 2.18, 2.09, 2.04), (1.31, 2.18, 1.31, 2.04), (1.04, 0.44, 1.2, 0.44),
            (2.3, 1.5, 2.3, 0.77), (1.31, 0.77, 1.31, 1.5)),
    gap_labels=((2.24, 1.235), (1.25, 1.235)))
ARCH_COMPACT = dict(   # 6-page paper: same boxes and labels, tighter gaps and line spacing, height <= 2.3 in
    H=2.3, ls=1.1, label_dy=0.09,
    groups=((1.06, 1.2, "On device (PV site)"), (0.04, 0.72, "Cloud")),
    boxes=((0.12, 1.6, 0.92, 0.48), (1.22, 1.6, 1.36, 0.48), (0.12, 1.12, 1.28, 0.32), (1.6, 1.12, 0.98, 0.32),
           (0.12, 0.1, 0.9, 0.44), (1.22, 0.1, 1.36, 0.44)),
    arrows=((1.06, 1.84, 1.2, 1.84), (2.09, 1.58, 2.09, 1.46), (1.31, 1.58, 1.31, 1.46), (1.04, 0.32, 1.2, 0.32),
            (2.3, 1.1, 2.3, 0.56), (1.31, 0.56, 1.31, 1.1)),
    gap_labels=((2.24, 0.91), (1.25, 0.91)))
ARCH_BOXES = (("Ground\nsensors: GHI,\nDNI, weather,\nclear-sky", "#f2f2f2", False),
              ("Edge tier, MLP\n43.8 kB\n(int8: 15.2 kB)", "#cfe3f3", True),
              ("Forecast,\n30 to 180 min", "#f2f2f2", False),
              ("Gate, int8\n6.1 kB\nescalate?", "#fde2c8", True),
              ("GOES-15 tile\nNAM, 4 nodes", "#f2f2f2", False),
              ("Cloud tier\ntrees + network\n(averaged in kt)", "#cfe3f3", True))


def fig_architecture(name: str = "architecture", L: dict = ARCH_FULL) -> list[str]:
    """Same content as figures/core/architecture, stacked for one column: device group above, cloud below.
    Coordinates are in inches. Arrows: sensors -> edge, edge -> gate, edge -> forecast, satellite/NAM -> cloud tier,
    request (gate -> cloud), cloud forecast (cloud -> device)."""
    H = L["H"]
    fig = plt.figure(figsize=(WIDTH_IN, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH_IN)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, text, fc, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=S.MUTED, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color=S.INK,
                fontweight="bold" if bold else "normal", linespacing=L["ls"])

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=S.MUTED))

    for y, h, label in L["groups"]:
        ax.add_patch(FancyBboxPatch((0.04, y), 2.62, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc="none",
                                    ec=S.MUTED, lw=0.6, ls=(0, (3, 2))))
        ax.text(0.12, y + h - L["label_dy"], label, va="center", fontsize=7.5, color=S.MUTED, fontweight="bold")
    for geom, (text, fc, bold) in zip(L["boxes"], ARCH_BOXES):
        box(*geom, text, fc, bold)
    for a in L["arrows"]:
        arrow(*a)
    # labels sit in the gap between the two groups, clear of every box, border and arrow
    for (x, y), text in zip(L["gap_labels"], ("request", "cloud forecast")):
        ax.text(x, y, text, ha="right", va="center", fontsize=7, color=S.MUTED)
    return save_paper(fig, name)


def fig_architecture_compact() -> list[str]:
    return fig_architecture("architecture_compact", ARCH_COMPACT)


def fig_routing_map(name: str = "routing_map", height: float = 2.3) -> list[str]:
    hm = pd.read_csv("results/analyses/a3_routing_hour_month.csv")
    piv = hm.pivot(index="hour", columns="month", values="share_escalated").sort_index()
    fig, ax = plt.subplots(figsize=(WIDTH_IN, height), layout="constrained")
    ax.grid(False)
    im = ax.imshow(piv.values, aspect="auto", cmap=S.SEQ_CMAP, vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(piv.columns)), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"][:len(piv.columns)])
    yt = [i for i, h in enumerate(piv.index) if h % 2 == 0]
    ax.set_yticks(yt, [f"{piv.index[i]:02d}" for i in yt])
    ax.set_xlabel("Month (2016)")
    ax.set_ylabel("Hour of issue time (PST)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Share escalated")
    cb.outline.set_edgecolor(S.MUTED)
    cb.ax.tick_params(labelsize=7)
    return save_paper(fig, name)


def fig_routing_map_compact() -> list[str]:
    """6-page paper: same data and colour bar, 1.9 in tall."""
    return fig_routing_map("routing_map_compact", 1.9)


def gate_intervals() -> pd.DataFrame:
    """Paired day-block intervals (A5) for uncertainty - random and uncertainty - learned at the report budgets,
    per seed and summarised, with each gate's realised rate, for citation in the text."""
    a5 = pd.read_csv("results/analyses/a5_gate_differences.csv")
    summ = pd.read_csv("results/test/gates/summary_test.csv")
    s = summ[(summ.escalate_to == "cloud") & (summ.row_set == "primary")].set_index(["gate", "budget_target"])
    pairs = [("uncertainty", "random"), ("uncertainty", "learned")]
    rows = []
    for ga, gb in pairs:
        for b in REPORT_BUDGETS:
            d = a5[(a5.gate_a == ga) & (a5.gate_b == gb) & (a5.budget_target.round(4) == b)].sort_values("seed")
            assert len(d) == 5, (ga, gb, b)
            base = {"comparison": f"{ga} - {gb}", "budget_target": b,
                    "rate_a": s.loc[(ga, b), "realised_rate_mean"], "rate_b": s.loc[(gb, b), "realised_rate_mean"]}
            for r in d.itertuples():
                rows.append({**base, "seed": str(r.seed), "point": r.point, "lo": r.lo, "hi": r.hi,
                             "excludes_zero": not r.includes_zero})
            rows.append({**base, "seed": "mean", "point": d.point.mean(), "lo": d.lo.mean(), "hi": d.hi.mean(),
                         "excludes_zero": f"{int((~d.includes_zero).sum())} of 5"})
    t = pd.DataFrame(rows)
    PAPER.mkdir(parents=True, exist_ok=True)
    t.to_csv(PAPER / "gate_intervals.csv", index=False, float_format="%.4f")
    md = ["# Paired gate differences at the report budgets (2016 test year)", "",
          "Generated by `scripts/paper_figures.py` from `results/analyses/a5_gate_differences.csv` (analysis A5) and "
          "`results/test/gates/summary_test.csv`. Nothing was recomputed.", "",
          "Each value is RMSE_A - RMSE_B in W/m2 on the primary rows (39,655 cells), RMSE averaged over the 6 horizons. "
          "Negative means gate A has the lower RMSE. Intervals are the paired day-block bootstrap 95% percentile "
          "intervals of `docs/FREEZE.md` (2,000 resamples, bootstrap seed 0), one per model seed. Random is the "
          "expected random gate (exact share b of issue times). The budget is the target; the realised rates of the "
          "two gates differ and are given. In the 'mean' row, point, lo and hi are means over the 5 seeds of the "
          "per-seed values; that row is a summary, not a 95% interval in its own right.", ""]
    for (cmp_, b), d in t.groupby(["comparison", "budget_target"], sort=False):
        ga, gb = cmp_.split(" - ")
        md += [f"## {ga} - {gb}, target {int(b * 100)}% (realised: {ga} {d.rate_a.iloc[0]:.3f}, "
               f"{gb} {d.rate_b.iloc[0]:.3f})", "",
               "| seed | difference | 95% interval | excludes zero |", "|---|---|---|---|"]
        for r in d.itertuples():
            ex = ("yes" if r.excludes_zero else "no") if isinstance(r.excludes_zero, bool) else r.excludes_zero
            iv = f"[{r.lo:.2f}, {r.hi:.2f}]" if r.seed != "mean" else f"mean bounds {r.lo:.2f} to {r.hi:.2f}"
            md.append(f"| {r.seed} | {r.point:.2f} | {iv} | {ex} |")
        md.append("")
    (PAPER / "gate_intervals.md").write_text("\n".join(md), encoding="utf-8")
    return t


PAPER_ROWS = [   # (name in core_table.csv, name in the paper table, size note, latency note)
    ("Smart persistence", "Smart persistence", "", ""),
    ("Linear anchor, ground (lasso_endo)", "Linear anchor, ground (lasso_endo)", "", ""),
    ("Edge tier, fp32", "Edge tier, fp32", "ONNX", ""),
    ("Edge tier, int8 (secondary)", "Edge tier, int8", "ONNX", ""),
    ("Ground-only trees, 10x cap (C object)", "Ground-only trees, 10x cap", "compiled C object", ""),
    ("Trees, all inputs (pickle)", "Trees, all inputs", "Python pickle", ""),
    ("Cloud tier (trees + network)", "Cloud tier (trees + network)", "", ""),
    ("Random (expected) @ 25%", "Random @ 25%", "plus edge tier", ""),
    ("Uncertainty gate @ 25%", "Uncertainty gate @ 25%", "int8 ONNX, plus edge tier", "plus edge tier"),
    ("Learned gate @ 25%", "Learned gate @ 25%", "int8 ONNX, plus edge tier", "plus edge tier"),
]


def paper_table() -> pd.DataFrame:
    """Ten rows of results/paper/core_table.csv (itself generated from the result files), with lasso_exo as a
    footnote because it is scored on different rows."""
    core = pd.read_csv(PAPER / "core_table.csv").set_index("name")
    rows = []
    for src, name, size_note, lat_note in PAPER_ROWS:
        r = core.loc[src]
        rows.append({"name": name, "RMSE": r.RMSE, "RMSE_seed_min": r.RMSE_seed_min, "RMSE_seed_max": r.RMSE_seed_max,
                     "share_kept": r.share_kept, "realised_rate": r.realised_rate, "size_kB": r.size_bytes / 1000,
                     "size_note": size_note, "cpu_latency_median_us": r.cpu_latency_median_us, "latency_note": lat_note})
    t = pd.DataFrame(rows)
    t.to_csv(PAPER / "paper_table.csv", index=False, float_format="%.4f")

    cm = pd.read_csv("results/test/claims_models_test.csv")
    an = cm[cm.row_set == "primary_anchor_rows"]
    ar = {m: an[an.model == m]["RMSE"].mean() for m in ("lasso_exo", "lasso_endo", "edge_fp32", "cloud")}
    n_anchor = int(an["cells"].iloc[0])

    def f(x, fmt):
        return "n/a" if pd.isna(x) else format(x, fmt)
    md = ["# Paper table (2016 test year)", "",
          "Generated by `scripts/paper_figures.py` from `results/paper/core_table.csv` (see `core_table.md` for its "
          "sources) and, for footnote a, `results/test/claims_models_test.csv`.", "",
          "| model | RMSE (W/m2) [seed range] | share kept | realised rate | size (kB) | CPU latency (µs) |",
          "|---|---|---|---|---|---|"]
    for r in t.itertuples():
        rng = "" if r.RMSE_seed_min == r.RMSE_seed_max else f" [{r.RMSE_seed_min:.1f}-{r.RMSE_seed_max:.1f}]"
        if r.size_kB == 0:   # nothing stored (smart persistence, random): shown as a dash, not 0.0
            size = "–"
        else:
            size = f(r.size_kB, ",.1f") + (f" ({r.size_note})" if r.size_note and not pd.isna(r.size_kB) else "")
        lat = f(r.cpu_latency_median_us, ".1f") + (f" ({r.latency_note})" if r.latency_note else "")
        sup = "<sup>a</sup>" if r.name.startswith("Linear anchor") else ""
        md.append(f"| {r.name}{sup} | {r.RMSE:.2f}{rng} | {f(r.share_kept, '.2f')} | {f(r.realised_rate, '.3f')} | "
                  f"{size} | {lat} |")
    md += ["",
           f"All rows are on the primary rows (39,655 cells). RMSE is averaged over the 6 horizons. For the edge, cloud "
           f"and gate rows it is the mean over 5 seeds, with the range over seeds in brackets; the other rows are "
           f"deterministic. Share kept = (RMSE_edge - RMSE) / (RMSE_edge - RMSE_cloud), from the seed-mean RMSEs, "
           f"with the edge tier in fp32; n/a for the anchors. Realised rate is the share of issue times escalated "
           f"(0 for the edge tier, 1 for the cloud tier, n/a where nothing is escalated). The gate rows use a 25% "
           f"target budget; random is the expected random gate. Size is in kB (1 kB = 1,000 B); – means no model "
           f"is stored (smart persistence and random). The tree sizes "
           f"are of different kinds: the 10x-capped trees are a compiled C object, and trees on all inputs are a "
           f"Python pickle. Cloud-tier size is not stated. Latency is the median single-row, one-thread time on a "
           f"PC CPU (ONNX Runtime or compiled C); nothing ran on a microcontroller. Gate size and latency are in "
           f"addition to the edge tier.", "",
           f"<sup>a</sup> lasso_exo (ground + satellite) needs the benchmark satellite feature, so it is scored only "
           f"on the {n_anchor:,} anchor rows where that feature exists: RMSE {ar['lasso_exo']:.2f}. On those same rows, "
           f"lasso_endo has {ar['lasso_endo']:.2f}, edge fp32 {ar['edge_fp32']:.2f} (seed mean) and the cloud tier "
           f"{ar['cloud']:.2f} (seed mean)."]
    (PAPER / "paper_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return t


CLAIMS = [   # (number, pre-specified wording from docs/FREEZE.md section b, shortened to one line)
    (1, "With the model type fixed, satellite and NAM inputs add nothing."),
    (2, "A ground-only tree model of about 85 KB matches the cloud tier."),
    (3, "The cloud tier beats the on-device (edge) network."),
    (4, "An on-device gate beats random, the learned gate is not better than the uncertainty gate, "
        "and refitting on all of 2015 fixes the budget shortfall."),
    (5, "On ramp cells there is no reliable cloud advantage, and no gate helps."),
]


def claims_table() -> list[str]:
    """Five claims: wording, verdict from the 'counts against' flags of claims_test.json (none against = supported,
    all against = refuted, otherwise partly supported), and one key number read from the same file."""
    c = json.loads(Path("results/test/claims_test.json").read_text())
    c1, c2, c3 = c["claim1_input_effect"], c["claim2_on_device_sufficiency"], c["claim3_edge_vs_cloud_fp32"]
    c4, c5 = c["claim4_gates"], c["claim5_ramp"]
    against = {1: [c1["counts_against"]], 2: [c2["counts_against"]], 3: [c3["counts_against"]],
               4: [c4["against_gate_beats_random"], c4["against_learned_not_better"], c4["against_refit_fixes_shortfall"]],
               5: [c5["against_no_cloud_advantage"], c5["against_no_gate_helps"]]}
    iv1 = c1["interval_rmse_ground_minus_all"]
    n2 = c2["seeds_entirely_above_zero"]
    n3 = sum(not i["includes_zero"] and i["lo"] > 0 for i in c3["intervals_rmse_edge_minus_cloud"])
    sh4 = c4["uncertainty_minus_random_share_at_25pct_per_seed"]
    n4 = sum(x >= 0.10 for x in sh4)
    n5 = sum(i["lo"] > 0 for i in c5["intervals_rmse_edge_minus_cloud_ramp"])
    best5 = max(c5["gate_share_at_25pct_ramp"].items(), key=lambda kv: kv[1]["share_mean"])
    key = {1: f"input effect {c1['input_effect'] * 100:.1f}%; RMSE_ground - RMSE_all {iv1['point']:.2f} W/m2 "
              f"[{iv1['lo']:.2f}, {iv1['hi']:.2f}]",
           2: f"d = +{c2['relative_difference_d'] * 100:.1f}% (trees worse); interval above zero for {n2} of 5 seeds",
           3: f"gain {c3['gain_mean_of_seed_rmse'] * 100:.1f}% (seed mean); interval excludes zero for {n3} of 5 seeds",
           4: f"@25%: uncertainty minus random share kept {min(sh4):.2f} to {max(sh4):.2f}; >= 0.10 for {n4} of 5 seeds",
           5: f"ramp RMSE_edge - RMSE_cloud interval above zero for {n5} of 5 seeds; best gate share on ramps "
              f"{best5[1]['share_mean']:.2f} ({best5[0]}) < 0.5"}

    def verdict(a):
        return "supported" if not any(a) else "refuted" if all(a) else "partly supported"
    md = ["# Claims table (2016 test year)", "",
          "Generated by `scripts/paper_figures.py` from `results/test/claims_test.json`; wording shortened from "
          "`docs/FREEZE.md` section b. The verdict follows that file's 'counts against' rules: supported if no rule "
          "counts against the claim, refuted if all do, partly supported otherwise (claim 4 has three rules, claim 5 "
          "two). Intervals are paired day-block bootstrap 95% intervals on the primary rows (ramp cells for claim 5).", "",
          "| claim | verdict | key number |", "|---|---|---|"]
    md += [f"| {n}. {w} | {verdict(against[n])} | {key[n]} |" for n, w in CLAIMS]
    (PAPER / "claims_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return md


def gate_intervals_compact() -> list[str]:
    g = pd.read_csv(PAPER / "gate_intervals.csv")
    g = g[g.seed == "mean"]
    md = ["# Gate differences, compact (2016 test year)", "",
          "From `results/paper/gate_intervals.csv` ('mean' rows). Each cell: seed-mean RMSE_A - RMSE_B in W/m2 "
          "(negative = uncertainty gate better), and in brackets the number of seeds whose paired day-block 95% "
          "interval excludes zero, of 5. Columns are target budgets.", "",
          "| comparison | 10% target | 25% target | 50% target |", "|---|---|---|---|"]
    for cmp_ in ("uncertainty - random", "uncertainty - learned"):
        cells = []
        for b in REPORT_BUDGETS:
            r = g[(g.comparison == cmp_) & (g.budget_target.round(4) == b)].iloc[0]
            cells.append(f"{float(r.point):.2f} ({r.excludes_zero.split(' of ')[0]})")
        md.append(f"| {cmp_} | " + " | ".join(cells) + " |")
    (PAPER / "gate_intervals_compact.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return md


def horizon_table() -> list[str]:
    """Post hoc (A8): per-horizon seed-mean RMSE and the cloud gain relative to the edge tier, in %."""
    a8 = pd.read_csv("results/analyses/posthoc/a8_per_horizon_rmse.csv", comment="#")
    t = pd.DataFrame({"horizon": a8.horizon, "smart_persistence": a8.smart_persistence_mean,
                      "edge_fp32": a8.edge_fp32_mean, "cloud": a8.cloud_mean,
                      "uncertainty_25pct": a8.uncertainty_25pct_mean})
    t["cloud_gain_vs_edge_pct"] = (1 - t.cloud / t.edge_fp32) * 100
    with open(PAPER / "horizon_table.csv", "w", encoding="utf-8", newline="") as f:
        f.write("# post hoc, not pre-specified\n")
        t.to_csv(f, index=False, float_format="%.4f")
    md = ["post hoc, not pre-specified", "", "# RMSE by horizon (2016 test year, post hoc)", "",
          "From `results/analyses/posthoc/a8_per_horizon_rmse.csv` (analysis A8, post hoc). RMSE in W/m2 on the primary "
          "rows; edge, cloud and the uncertainty gate (25% target) are means over 5 seeds. Cloud gain = "
          "1 - RMSE_cloud / RMSE_edge from the seed means, in %.", "",
          "| horizon | smart persistence | edge fp32 | cloud | uncertainty @25% | cloud gain vs edge (%) |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r.horizon.replace('min', ' min')} | {r.smart_persistence:.2f} | {r.edge_fp32:.2f} | {r.cloud:.2f} | "
           f"{r.uncertainty_25pct:.2f} | {r.cloud_gain_vs_edge_pct:.1f} |" for r in t.itertuples()]
    (PAPER / "horizon_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return md


def main() -> None:
    apply_paper()
    files = (fig_architecture() + fig_architecture_compact() + fig_budget_sweep() + fig_budget_sweep_short()
             + fig_routing_map() + fig_routing_map_compact())
    gate_intervals()
    paper_table()
    claims_table()
    gate_intervals_compact()
    horizon_table()
    print("\n".join(files))


if __name__ == "__main__":
    main()
