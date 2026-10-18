post hoc, not pre-specified

# RMSE by horizon (2016 test year, post hoc)

From `results/analyses/posthoc/a8_per_horizon_rmse.csv` (analysis A8, post hoc). RMSE in W/m2 on the primary rows; edge, cloud and the uncertainty gate (25% target) are means over 5 seeds. Cloud gain = 1 - RMSE_cloud / RMSE_edge from the seed means, in %.

| horizon | smart persistence | edge fp32 | cloud | uncertainty @25% | cloud gain vs edge (%) |
|---|---|---|---|---|---|
| 30 min | 57.81 | 56.25 | 50.54 | 53.03 | 10.1 |
| 60 min | 76.17 | 69.58 | 63.95 | 65.88 | 8.1 |
| 90 min | 88.22 | 76.92 | 71.81 | 73.48 | 6.6 |
| 120 min | 97.17 | 81.95 | 75.82 | 77.90 | 7.5 |
| 150 min | 104.73 | 86.06 | 79.25 | 81.04 | 7.9 |
| 180 min | 112.45 | 89.50 | 82.61 | 84.33 | 7.7 |
