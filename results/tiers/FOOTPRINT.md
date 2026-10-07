# Footprint: ground-only trees against the edge network (validation)

Run with `./run.sh footprint` (`scripts/footprint_trees.py`). All numbers are in
`results/tiers/footprint_trees.json`; the rules are in `docs/PLAN.md` section 8.

## Sizes

| model | size | other |
|---|---|---|
| edge network | 10,758 parameters; float32 43,032 B; **int8 about 11,208 B** (estimate) | single-row CPU time 0.019 ms (PyTorch) |
| trees_ground (uncapped) | 517 trees, max depth 18, 31,537 nodes; compact about 378 KB (estimate), pickle 2.9 MB | 4.3 ms (sklearn) |
| trees_ground, 10x cap (112,080 B) | 473 trees (max 103 per horizon), 8 leaves per tree, max depth 7, 7,095 nodes; **compact about 85 KB (estimate)**, pickle 1.5 MB | 4.0 ms |
| trees_ground, 100x cap (1,120,800 B) | the cap did not bind: early stopping chose the same 517 trees as uncapped | 4.3 ms |

- Compact size = nodes x 12 bytes. This is an estimate for a plain array encoding, not a
  measured on-device file.
- CPU times are single-row times on a PC through Python. They are not microcontroller latency.

## Validation RMSE (W/m2), primary rows, 20,125 cells; seed mean where seeds apply

| model | 30min | 60min | 90min | 120min | 150min | 180min | mean |
|---|---|---|---|---|---|---|---|
| edge network | 51.88 | 64.05 | 72.89 | 79.26 | 83.35 | 88.26 | 73.28 |
| cloud (step-4 average) | 46.77 | 60.08 | 68.66 | 73.82 | 78.70 | 83.91 | 68.66 |
| trees_ground | 44.33 | 58.82 | 69.16 | 74.58 | 80.97 | 87.12 | 69.16 |
| trees_ground, 10x cap | 44.96 | 59.14 | 68.70 | 75.15 | 79.93 | 84.83 | 68.78 |
| trees_ground, 100x cap | 44.33 | 58.82 | 69.16 | 74.58 | 80.97 | 87.12 | 69.16 |

## Does a ground-only tree model of on-device size match the cloud tier?
**Yes, on validation.** The 10x-capped ground-only trees (about 85 KB estimated, 7.6 times the
edge network's int8 size) have an averaged RMSE of 68.78 W/m2, against 68.66 for the cloud tier.

The paired day-block bootstrap 95% interval of RMSE_trees - RMSE_cloud includes zero for all
five cloud seeds:

| cloud seed | point (W/m2) | 95% interval |
|---|---|---|
| 0 | +0.06 | -2.27 to +2.46 |
| 1 | +0.06 | -2.30 to +2.41 |
| 2 | +0.67 | -1.66 to +3.01 |
| 3 | +0.29 | -2.21 to +2.85 |
| 4 | -0.45 | -2.82 to +1.93 |

Uncapped trees_ground also matches the cloud tier on all five seeds, with points from -0.07 to
+1.05 W/m2.

Caveats:
- "On-device size" here is a size estimate. Nothing was run on a microcontroller.
- The tree model is about 7.6 times larger than the int8 edge network. Whether 85 KB fits a
  target device is a hardware question this study does not answer.

## Edge vs cloud, paired day-block bootstrap (RMSE_edge - RMSE_cloud, W/m2)

| seed | point | 95% interval | includes zero? |
|---|---|---|---|
| 0 | 4.34 | 1.27 to 7.83 | no |
| 1 | 3.49 | 0.65 to 6.51 | no |
| 2 | 3.98 | 1.14 to 7.10 | no |
| 3 | 8.78 | 5.61 to 11.98 | no |
| 4 | 2.55 | 0.04 to 5.14 | no (lower bound barely above zero) |
