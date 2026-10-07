# Stage 6a: model-type control (validation, Stage 5 rows and seeds)

Run with `./run.sh control` (`scripts/control.py`). The numbers are in
`results/tiers/control.csv` and `results/tiers/control_effects.json`. The protocol is in
`docs/PLAN.md` section 7.

## Set-up
- The same HistGradientBoosting set-up as ladder step 4 is trained on four input sets.
- The edge network and the step-3 network are the saved Stage 5 predictions.
- Two consistency checks against Stage 5 pass:
  - trees_all reproduces the Stage 5 step-4 trees (69.35 W/m2);
  - the step-4 average reproduces its per-seed RMSE.
- The tree models are deterministic, so their five seeds are identical. Spread over seeds comes
  only from the edge network and the step-3 network.

## RMSE (W/m2, seed mean), primary rows (20,125 cells)

| model | 30min | 60min | 90min | 120min | 150min | 180min | mean | range of mean over seeds |
|---|---|---|---|---|---|---|---|---|
| edge network | 51.88 | 64.05 | 72.89 | 79.26 | 83.35 | 88.26 | 73.28 | 71.78-77.27 |
| trees_ground (edge inputs) | 44.33 | 58.82 | 69.16 | 74.58 | 80.97 | 87.12 | 69.16 | identical |
| trees ground+sat | 43.57 | 59.28 | 69.10 | 74.69 | 81.43 | 86.17 | 69.04 | identical |
| trees ground+NAM | 44.79 | 59.92 | 68.74 | 75.88 | 80.82 | 86.92 | 69.51 | identical |
| trees all inputs | 44.84 | 59.87 | 69.28 | 75.34 | 79.73 | 87.05 | 69.35 | identical |
| step-4 average (cloud tier) | 46.77 | 60.08 | 68.66 | 73.82 | 78.70 | 83.91 | 68.66 | 68.11-69.23 |
| smart persistence | 54.25 | 70.40 | 84.39 | 95.46 | 105.62 | 115.19 | 87.55 | n/a |

All daylight cells give the same picture (`control.csv`, row_set `all_daylight`).

## The two effects
Effect = 1 - RMSE_A / RMSE_B, with RMSE per horizon averaged over horizons; positive means A is
better.

| effect | rows | mean | median | range over seeds | W/m2 (mean) |
|---|---|---|---|---|---|
| (a) input effect: trees all inputs vs trees_ground | primary | **-0.27%** | -0.27% | -0.27% to -0.27% | -0.19 |
| (b) model effect: trees_ground vs edge network | primary | **+5.55%** | +4.22% | +3.65% to +10.50% | +4.12 |
| (a) input effect | all daylight | -0.21% | -0.21% | -0.21% to -0.21% | -0.14 |
| (b) model effect | all daylight | +5.59% | +4.21% | +3.71% to +10.62% | +4.14 |

## What this says
1. **The Stage 5 pass is a model-type effect, not an input effect.** With the same tree model,
   adding satellite and NAM to the ground inputs changes averaged RMSE by -0.27%; the all-input
   trees are slightly worse. Satellite alone helps a little (69.04 against 69.16 W/m2); NAM
   alone hurts (69.51).
2. **Trees on ground data alone are almost as good as the chosen cloud tier.** trees_ground is
   69.16 W/m2 against 68.66 for the step-4 average. The remaining 0.7% comes from averaging with
   the network. Per horizon:
   - trees_ground is better at 30 and 60 min: 44.3 against 46.8 W/m2 at 30 min;
   - the step-4 average is better from 120 min on: 83.9 against 87.1 W/m2 at 180 min.
3. **The model effect is about as large as the go/no-go gain itself.** Its median is +4.2%, so
   on its own it falls below the 5% threshold. Edge seed 3, the outlier, again sets the top of
   the range.

As decided in `docs/PLAN.md` section 7, the go/no-go verdict, the chosen cloud tier and the edge
tier are not changed by this result.
