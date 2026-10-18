post hoc, not pre-specified

# A10: target vs realised escalation rate on 2016

Share of the 7,457 evaluation issue times escalated, seed mean and range over 5 seeds. Source: results/test/gates/summary_test.csv (escalate_to = cloud, row_set = primary). Thresholds were fitted on the 2015 gate-fit population, so realised rates on 2016 can differ from the target; random and oracle hit the target by construction, fixed_interval by its per-day schedule.

| gate | budget_target | realised_rate_mean | realised_rate_min | realised_rate_max | realised_minus_target_mean |
|---|---|---|---|---|---|
| fixed_interval | 0.1000 | 0.0774 | 0.0774 | 0.0774 | -0.0226 |
| learned | 0.1000 | 0.1053 | 0.0999 | 0.1084 | 0.0053 |
| oracle | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0000 |
| random | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0000 |
| uncertainty | 0.1000 | 0.0994 | 0.0960 | 0.1029 | -0.0006 |
| variability | 0.1000 | 0.1055 | 0.1055 | 0.1055 | 0.0055 |
| fixed_interval | 0.2500 | 0.2316 | 0.2316 | 0.2316 | -0.0184 |
| learned | 0.2500 | 0.2476 | 0.2396 | 0.2602 | -0.0024 |
| oracle | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0000 |
| random | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0000 |
| uncertainty | 0.2500 | 0.2303 | 0.2277 | 0.2324 | -0.0197 |
| variability | 0.2500 | 0.2362 | 0.2362 | 0.2362 | -0.0138 |
| fixed_interval | 0.5000 | 0.4875 | 0.4875 | 0.4875 | -0.0125 |
| learned | 0.5000 | 0.4939 | 0.4718 | 0.5356 | -0.0061 |
| oracle | 0.5000 | 0.4999 | 0.4999 | 0.4999 | -0.0001 |
| random | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| uncertainty | 0.5000 | 0.4770 | 0.4690 | 0.4880 | -0.0230 |
| variability | 0.5000 | 0.4986 | 0.4986 | 0.4986 | -0.0014 |
