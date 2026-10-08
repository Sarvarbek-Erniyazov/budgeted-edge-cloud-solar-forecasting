# Captions for the paper figures and table

Figures are in `figures/paper/` (SVG and 600 dpi PNG, 2.7 in wide). The table is `results/paper/paper_table.md`.

**Architecture (`architecture.svg`).** The on-device edge tier (fp32 MLP, 43.8 kB; int8 version 15.2 kB) forecasts from ground sensors, and a 6.1 kB int8 gate decides for each issue time whether to request the cloud tier, which adds a GOES-15 tile and NAM forecasts at 4 nodes. A budget b fixes the share of issue times escalated, and an escalated issue time receives the cloud forecast instead of the edge forecast.

**Budget sweep (`budget_sweep.svg`).** RMSE (averaged over 6 horizons) on the 2016 primary rows (39,655 cells) against the realised share of issue times escalated, with lines at the mean and shaded bands over the range of 5 seeds; random is the expected random gate, and the horizontal lines are the edge tier only, the cloud tier only (seed means) and the deterministic 10x-capped ground-only trees. Markers are at the 10, 25 and 50% target budgets, plotted at the realised rate, which can differ from the target.

**Routing map (`routing_map.svg`).** Share of issue times escalated by the uncertainty gate at the 25% target budget, by month and local standard-time hour, averaged over 5 seeds. It covers the 7,457 test-year issue times the gate decides on (satellite frame available and at least one daylight horizon); blank cells have none.

**Table (`results/paper/paper_table.md`).** RMSE (averaged over 6 horizons), share of the cloud gain kept, realised escalation rate, size and single-row CPU latency on the 2016 primary rows (39,655 cells); edge, cloud and gate rows give the mean over 5 seeds with the seed range in brackets, the other models are deterministic, and gate rows use the 25% target budget. lasso_exo is in a footnote because it is scored on the 29,948 anchor rows only.
