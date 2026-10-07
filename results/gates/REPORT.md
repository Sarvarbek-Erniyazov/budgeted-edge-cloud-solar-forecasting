# Stage 6b: gates and budget sweep (validation, 2015-07 to 2015-12)

Run in order with `./run.sh predictions`, then `./run.sh gates`. The protocol is in
`docs/PLAN.md` sections 7 and 8. Numbers come from:
- `results/gates/sweep_validation.csv`: per seed, gate, budget and target;
- `summary_validation.csv`: mean, median and range over seeds;
- `bootstrap_validation.csv`;
- `edge_vs_cloud_bootstrap.json`;
- `meta.json`.

The thresholds are in `configs/gate_thresholds.yaml`, generated from gate_fit scores only.

## Set-up
- **Population:** 3,809 validation issue times, 20,125 daylight cells (the Stage 5 primary
  set). Budget = share of these issue times escalated. One escalation returns the target's
  forecast for all six horizons.
- **Predictions:** read from `results/tiers/predictions/`. All are out of sample, since both
  tiers were trained on 2014. The edge and cloud RMSE reproduce Stage 5 for every seed.
- **Gate inputs:** on-device only, meaning the edge inputs plus the edge kt forecast (102
  columns, listed in `meta.json`). The tests check that no column contains sat, nam, cloud or
  sif.
- **Gate networks** (uncertainty, learned): Linear+ReLU, one hidden layer of 32, one per seed.
  They were trained on 2,762 gate_fit issue times, with the last 30 days of gate_fit (765 issue
  times) used for early stopping. Thresholds are the (1-b) quantiles over the 3,527 gate_fit
  issue times.
- **Ramp subset:** 1,173 validation cells where |kt(t,h) - kt(t,h-30min)| >= 0.25.

## Edge vs cloud, paired day-block bootstrap (RMSE_edge - RMSE_cloud, W/m2, 95%)

| seed | point | interval | includes zero? |
|---|---|---|---|
| 0 | 4.34 | 1.23 to 7.69 | no |
| 1 | 3.49 | 0.64 to 6.34 | no |
| 2 | 3.98 | 0.97 to 7.04 | no |
| 3 | 8.78 | 5.59 to 12.00 | no |
| 4 | 2.55 | -0.01 to 5.08 | **yes** |

Edge against trees_ground (the reference target) excludes zero for all five seeds: points 2.62
to 8.11 W/m2, lower bounds 0.44 to 4.88.

## Main result: escalating to the cloud tier, primary rows
Values are the mean over 5 seeds, with the [min, max] over seeds in brackets.
- Edge alone: RMSE 73.28. Cloud alone: 68.66.
- Share of gain retained = (RMSE_edge - RMSE_gate) / (RMSE_edge - RMSE_cloud).

| gate | target budget | realised rate | RMSE (W/m2) | share of gain retained |
|---|---|---|---|---|
| random (expected) | 0.10 / 0.25 / 0.50 | 0.10 / 0.25 / 0.50 | 72.84 / 72.15 / 71.01 | 0.10 / 0.25 / 0.49 |
| fixed-interval | 0.10 / 0.25 / 0.50 | 0.076 / 0.233 / 0.490 | 72.84 / 72.15 / 71.04 | 0.10 / 0.24 / 0.49 |
| variability | 0.10 / 0.25 / 0.50 | 0.113 / 0.236 / 0.452 | 72.18 / 71.40 / 70.19 | 0.23 [0.08, 0.33] / 0.40 [0.19, 0.50] / 0.68 [0.55, 0.78] |
| uncertainty | 0.10 / 0.25 / 0.50 | 0.101 [0.07, 0.14] / 0.241 [0.21, 0.28] / 0.433 [0.32, 0.59] | 71.26 / 70.10 / 69.23 | 0.42 [0.27, 0.56] / 0.70 [0.58, 0.87] / 0.87 [0.75, 1.00] |
| learned | 0.10 / 0.25 / 0.50 | 0.074 [0.06, 0.09] / 0.209 [0.16, 0.24] / 0.378 [0.28, 0.46] | 71.64 / 70.53 / 69.52 | 0.36 [0.24, 0.47] / 0.61 [0.48, 0.72] / 0.83 [0.73, 1.00] |
| oracle | 0.10 / 0.25 / 0.50 | exact | 63.66 / 61.81 / 61.27 | 2.36 / 2.78 / 2.90 |

Findings:
1. **The uncertainty gate is the best on-device gate.** At a 25% target, at a realised 24%, it
   keeps 70% of the cloud tier's gain (seed range 58 to 87%). The learned gate keeps 61%, but
   at a lower realised rate (21%). The variability rule keeps 40%. Random and fixed-interval
   keep about their budget share, as they should.
2. **The learned gate is not better than the uncertainty gate.** It predicts edge error minus
   cloud error from on-device inputs alone, and it is at or below the uncertainty gate at every
   reported budget. Part of the gap comes from it escalating less than its target.
3. **Realised rates fall short of targets for the learned and uncertainty gates at larger
   budgets.** At a 50% target, the learned gate realises 0.38 and the uncertainty gate 0.43.
   Their thresholds come from gate_fit (January to June, a wetter half-year). Validation
   (July to December) is clearer, so its scores are lower. The variability gate and the fixed
   gates track their targets more closely.
4. **The oracle exceeds 100%.** Choosing the better tier for each issue time beats the cloud
   tier everywhere: RMSE 61.3 at 50%, against 68.7 for cloud alone. The two tiers' errors are
   far from perfectly correlated. This is a headroom figure, not something achievable.

## Reference: escalating to trees_ground instead (no cloud-side inputs)
The share is always measured against the cloud tier's gain.

| gate | 0.10 | 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|
| uncertainty, to trees_ground | 0.40 | 0.56 | 0.73 | 0.89 |
| uncertainty, to cloud | 0.42 | 0.70 | 0.87 | 1.00 |
| learned, to trees_ground | 0.38 | 0.59 | 0.71 | 0.89 |
| learned, to cloud | 0.36 | 0.61 | 0.83 | 1.00 |

Escalating everything to trees_ground keeps 89% of the cloud tier's gain (seed range 74 to
103%). At a 25% budget with the uncertainty gate, trees_ground delivers 56 points of the cloud
tier's 70. **So most of the escalation benefit, roughly four fifths, does not need cloud-side
inputs at all. It needs only a stronger model on the same ground data.** This agrees with
Stage 6a and the footprint result.

## Ramp subset (1,173 cells)
- On ramps, the cloud tier is barely better than the edge tier: RMSE 133.4 against 135.3 W/m2,
  seed mean. With so small a gap, share-retained ratios are unstable. At a 25% budget the seed
  medians are:
  - learned 0.37;
  - uncertainty 0.17;
  - variability -0.47 (worse than not escalating);
  - oracle 7.9.
- Seed ranges often cross zero (`summary_validation.csv`, row_set `ramp`). **No gate shows a
  reliable benefit on ramps.**

## Bootstrap intervals at the report budgets (`bootstrap_validation.csv`)
- **Gate RMSE intervals are wide**, about ±8 W/m2 around each value (for example uncertainty
  at 25%: 70.1, mean interval 62.0 to 77.8). They reflect day-to-day variation in absolute
  error, not differences between gates.
- **Share-retained intervals are very wide and often include zero.** For example, uncertainty
  at 25% has a mean interval of -0.03 to 0.92. This is because the denominator, the edge-cloud
  gap, is itself not reliably positive in every resample; for seed 4 it includes zero.
- A paired gate-against-edge interval would be the sharper test. It was not part of this
  stage's specification and was not computed.

## Proposal for the freeze: refit the gates on all of 2015 before the test run?
**Proposed: yes.**
- **What:** refit the uncertainty and learned gate networks, and all score thresholds, on the
  whole of 2015 (gate_fit + validation).
- **What stays fixed:** the architecture, the inputs, the early-stop rule (the last 30 days of
  the fitting period) and the budget grid, exactly as they are now. The tiers stay as trained
  on 2014, so their 2015 predictions are still out of sample.
- **Reason:**
  - The main defect seen on validation is budget miscalibration from the seasonal shift: the
    learned gate realises 0.38 at a 0.50 target.
  - Thresholds fitted on a half-year of one season do not transfer to the other half. A full
    year matches the seasonal mix of the 2016 test year.
- **Cost:**
  - The refitted gates have no out-of-sample check before the test year. Their realised rates
    on 2016 must therefore be reported next to their targets, as here.
  - The validation numbers in this file remain the development evidence, and they are not
    replaced.
