# Gate differences, compact (2016 test year)

From `results/paper/gate_intervals.csv` ('mean' rows). Each cell: seed-mean RMSE_A - RMSE_B in W/m2 (negative = uncertainty gate better), and in brackets the number of seeds whose paired day-block 95% interval excludes zero, of 5. Columns are target budgets.

| comparison | 10% target | 25% target | 50% target |
|---|---|---|---|
| uncertainty - random | -2.23 (5) | -2.64 (5) | -2.50 (5) |
| uncertainty - learned | -0.27 (0) | -0.18 (0) | -0.27 (1) |
