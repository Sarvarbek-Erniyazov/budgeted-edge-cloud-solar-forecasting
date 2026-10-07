# Stage 5: tiers and go/no-go (validation, 2015-07 to 2015-12)

Run with `./run.sh tiers`. Numbers come from `results/go_no_go.json`,
`results/tiers/validation.csv`, `results/tiers/summary.json`,
`results/tiers/satellite_availability.json` and `results/tiers/params.json`. The decisions taken
before training are in `docs/PLAN.md`, section 6.

Protocol:
- Training uses models_train (2014) only. 7,362 rows are used for fitting; the last 30 days of
  models_train (504 rows) are used only for early stopping.
- Evaluation uses validation only. The test year was never loaded.
- Seeds 0 to 4.
- Training is deterministic: re-running `./run.sh tiers` reproduces `validation.csv` byte for
  byte.

## Inputs and sizes
- **Edge tier:** an MLP (96 inputs → 64 → 64 → 6 outputs) with **10,758 parameters**, built
  from Linear+ReLU layers only, so it can be quantised to int8. Inputs:
  - benchmark ground features (intra-day and 5-min);
  - 30-min site weather means;
  - solar elevation and clear-sky value per horizon;
  - calendar terms.
- **Cloud tier, base:** 61,510 parameters. A CNN over the newest GOES-15 tile (stamped at or
  before t-15 min) is fused with an MLP over the same ground inputs plus NAM. NAM covers all
  four nodes and all eight fields, plus dwsw/clear-sky and interpolation/missing flags, at
  t+h-15 min, taken from the newest run available at t.
- **Satellite availability on validation, under the 15-minute rule:** 98.8% of the 3,857
  daylight issue times (99.5% with a 0-minute lag).

## Go/no-go
**Statistic:** gain = 1 - RMSE_cloud / RMSE_edge. RMSE is computed per horizon, averaged over
the six horizons, then averaged over seeds. The check is on the primary rows: daylight cells
where a satellite frame is available under the 15-minute rule, 20,125 cells. **Threshold:**
0.05.

| variant (ladder step) | built on | gain, averaged | seeds passing | result |
|---|---|---|---|---|
| base (0) | none | -0.0074 | 1/5 | fail |
| input check (1) | base | -0.0074 (unchanged) | n/a | inputs verified; fail |
| longer satellite window + tile differences (2) | base | -0.0605 | 0/5 | fail |
| richer NWP (3) | base | +0.0092 | 1/5 | fail |
| GBT alone (4) | step 3 features | +0.0537 | 2/5 | **pass** |
| GBT + step-3 network, averaged (4) | step 3 | **+0.0631** | 3/5 | **pass** |

**Verdict: GO via ladder step 4.** The first step that passes is the stronger cloud model. The
best variant is the GBT and the step-3 network, averaged. Step 5 was not needed.

Per-horizon RMSE (W/m2, seed mean), primary rows, best variant:

| model | 30min | 60min | 90min | 120min | 150min | 180min | mean |
|---|---|---|---|---|---|---|---|
| cloud (GBT + net) | 46.8 | 60.1 | 68.7 | 73.8 | 78.7 | 83.9 | 68.7 |
| edge | 51.9 | 64.0 | 72.9 | 79.3 | 83.4 | 88.3 | 73.3 |
| lasso_endo (anchor) | 53.6 | 66.7 | 77.8 | 85.6 | 92.1 | 98.0 | 79.0 |
| smart persistence | 54.2 | 70.4 | 84.4 | 95.5 | 105.6 | 115.2 | 87.6 |
| gain, cloud vs edge | 0.099 | 0.062 | 0.058 | 0.069 | 0.056 | 0.049 | 0.063 |

Skill over smart persistence, averaged over horizons: cloud 0.204, edge 0.147, lasso_endo 0.087.
On all daylight issue times, with the cloud tier receiving a satellite-missing flag, the picture
is the same: cloud 68.6 W/m2 against edge 73.2 W/m2.

## How solid is the pass: read before relying on it
1. **The margin is thin and depends on the seed.** The per-seed gains for the best variant are
   0.059, 0.048, 0.055, 0.114 and 0.036: the median is 0.055 and 3 of 5 seeds pass. Edge seed 3
   is an outlier (RMSE 77.3 against 71.8 to 73.1 for the other seeds), and it raises the mean.
   GBT alone has a median seed gain of 0.040 and passes on 2 of 5 seeds.
2. **The pass rests on the model class, not on the satellite tiles.** The neural cloud tier
   never beat the edge tier by 5%. The base network was -0.7%, and the longer satellite window
   made it worse (-6.1%). Only the tree model on the same inputs passed.
3. **Winner chosen on validation.** Six cloud variants were compared on validation and the best
   was kept, as the ladder prescribes. This selection makes the validation gain optimistic. The
   test year is the unbiased check.
4. **GBT seeds are not independent.** HistGradientBoosting with these settings is deterministic,
   so its five "seeds" give identical predictions. The spread across seeds comes from the edge
   tier, and from the network half of the averaged variant.
5. **The cloud network relies on NAM in a fragile way.** In the step-1 ablation for the base
   network (seed 0), average RMSE is:
   - 74.7 W/m2 with all inputs;
   - 85.8 W/m2 with the tiles zeroed;
   - 173.7 W/m2 with the NAM block set to its training mean.
   Both inputs are used. The NAM result shows over-reliance on a large block of 26-to-41-hour-old
   NWP inputs, not that NWP is valuable.

## Anchor check (same rows)
- **Edge vs lasso_endo**, primary rows: the edge tier beats its anchor, 73.3 against
  79.0 W/m2.
- **Cloud vs lasso_exo**, on the 15,104 primary cells where the benchmark's satellite feature
  exists:

  | cloud variant | cloud RMSE | lasso_exo RMSE | beats anchor? |
  |---|---|---|---|
  | base | 72.2 | 74.9 | yes |
  | step 2 | 76.0 | 74.9 | **no** |
  | step 3 | 70.9 | 74.9 | yes |
  | step 4, GBT | 67.9 | 74.9 | yes |
  | step 4, GBT + net | 67.1 | 74.9 | yes |

  The step-2 cloud network does not beat the linear satellite anchor. Note that lasso_exo uses
  the benchmark's 0-minute satellite convention, which is optimistic.

## Sensitivity: benchmark convention (0-minute satellite lag)
The base network was retrained with frames up to t and scored on the same primary rows. Gain
over the edge tier is +0.0117, against -0.0074 with the 15-minute lag. So the satellite latency
assumption is worth about 2 points of gain for the base network. Not used for the verdict.

## Run history (for transparency)
- **Run 1, void.** It collided with an orphaned process from a run I had stopped, and both
  processes wrote the same files. Its numbers are void and were deleted.
- **Fixes before the clean run.** Seeding happens before each model is built, and training is
  deterministic: deterministic algorithms are on, and adaptive pooling was replaced by an
  equivalent spatial mean.
- **Clean run and repeat.** The clean run was repeated after a label fix and gave identical
  numbers.
- No settings were changed between runs.
