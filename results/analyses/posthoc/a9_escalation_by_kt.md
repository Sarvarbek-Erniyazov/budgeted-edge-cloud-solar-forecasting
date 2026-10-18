post hoc, not pre-specified

# A9: uncertainty gate @25% by clear-sky index at issue time (2016)

Clear-sky index at issue time = `ih_B(ghi_kt|5min)` (mean GHI kt over the 5 min before the issue time, an on-device input; capped at 1.2). Bins are [lo, hi) except 0.8-1.0, which includes 1.0; >1.0 is strictly above 1.0.
share_escalated: escalated issue times / issue times in the bin. share_of_escalations: escalated issue times in the bin / all escalated issue times. RMSE over the bin's daylight cells, per horizon then averaged over horizons that have cells; cloud_minus_edge < 0 means the cloud tier is better in that bin. Seed mean and range over 5 seeds. Gate decisions recomputed by inference with the saved gate networks and thresholds and checked to reproduce results/test/gates/sweep_test.csv exactly (4 dp).

| kt_bin | issue_times | cells | share_escalated_mean | share_escalated_min | share_escalated_max | share_of_escalations_mean | share_of_escalations_min | share_of_escalations_max | rmse_edge_mean | rmse_edge_min | rmse_edge_max | rmse_cloud_mean | rmse_cloud_min | rmse_cloud_max | cloud_minus_edge_mean | cloud_minus_edge_min | cloud_minus_edge_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0-0.2 | 367 | 1912 | 0.5613 | 0.4877 | 0.6403 | 0.1199 | 0.1054 | 0.1356 | 84.4402 | 82.2095 | 87.1223 | 80.1243 | 78.6711 | 82.0910 | -4.3159 | -6.9260 | -2.5100 |
| 0.2-0.4 | 447 | 2319 | 0.6649 | 0.6376 | 0.6890 | 0.1731 | 0.1653 | 0.1788 | 129.3447 | 128.1956 | 131.5828 | 118.1268 | 117.6813 | 118.4674 | -11.2179 | -13.1488 | -9.7282 |
| 0.4-0.6 | 537 | 2748 | 0.6898 | 0.6350 | 0.7095 | 0.2157 | 0.1978 | 0.2216 | 125.7208 | 124.0880 | 128.6358 | 116.1533 | 114.9952 | 116.8530 | -9.5676 | -11.9931 | -7.4063 |
| 0.6-0.8 | 447 | 2275 | 0.6004 | 0.5772 | 0.6174 | 0.1563 | 0.1497 | 0.1600 | 118.5825 | 116.6052 | 123.3019 | 107.1441 | 106.7090 | 107.5975 | -11.4384 | -16.5929 | -9.0486 |
| 0.8-1.0 | 2747 | 15564 | 0.1094 | 0.1045 | 0.1249 | 0.1749 | 0.1664 | 0.1990 | 55.6378 | 53.7652 | 61.2438 | 51.3295 | 50.7551 | 51.8859 | -4.3083 | -9.7936 | -1.8793 |
| >1.0 | 2912 | 14837 | 0.0944 | 0.0862 | 0.1016 | 0.1601 | 0.1448 | 0.1743 | 62.3757 | 61.4494 | 63.4267 | 57.7519 | 57.5209 | 57.9538 | -4.6238 | -5.6548 | -3.5035 |
