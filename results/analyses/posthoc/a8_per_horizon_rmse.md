post hoc, not pre-specified

# A8: per-horizon RMSE on 2016 (W/m2)

Primary cells (daylight, satellite frame available), 39,655 cells over 7,457 issue times. Seed mean and seed range (min, max) over 5 seeds; smart persistence has no seed.
Sources: results/test/claims_models_test.csv (smart persistence, edge_fp32, cloud) and results/test/gates/sweep_test.csv (escalate_to = cloud, row_set = primary, budget_target = 0.25).

| horizon | smart_persistence_mean | edge_fp32_mean | edge_fp32_min | edge_fp32_max | cloud_mean | cloud_min | cloud_max | uncertainty_25pct_mean | uncertainty_25pct_min | uncertainty_25pct_max | learned_25pct_mean | learned_25pct_min | learned_25pct_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30min | 57.8120 | 56.2478 | 54.6111 | 60.1982 | 50.5414 | 49.8946 | 51.4022 | 53.0265 | 51.3829 | 54.9993 | 52.6603 | 51.4721 | 53.9977 |
| 60min | 76.1669 | 69.5773 | 67.8427 | 71.4232 | 63.9472 | 63.2101 | 64.5769 | 65.8836 | 65.2837 | 66.9379 | 65.8573 | 65.2581 | 66.4841 |
| 90min | 88.2236 | 76.9248 | 75.6580 | 80.2158 | 71.8124 | 70.7989 | 72.4720 | 73.4835 | 72.7330 | 75.0462 | 73.6215 | 73.1365 | 74.4345 |
| 120min | 97.1746 | 81.9512 | 80.5193 | 85.4774 | 75.8180 | 75.4103 | 76.2516 | 77.8957 | 76.8738 | 80.0781 | 78.0909 | 76.9733 | 79.7338 |
| 150min | 104.7329 | 86.0559 | 84.7066 | 89.3047 | 79.2507 | 78.9629 | 79.7530 | 81.0400 | 80.1345 | 82.1347 | 81.3885 | 80.4631 | 82.1698 |
| 180min | 112.4527 | 89.5037 | 88.6631 | 92.0373 | 82.6067 | 82.2521 | 82.8914 | 84.3268 | 83.7756 | 85.1724 | 85.1374 | 84.4596 | 85.6939 |
