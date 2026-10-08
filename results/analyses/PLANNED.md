# Planned analyses A1 to A7 on the 2016 test run

Run with `./run.sh analyses` (`scripts/planned_analyses.py`, settings in `configs/analyses.yaml`).
- Everything reads `results/test/` and the saved checkpoints. Nothing was retrained, retuned or
  rerun.
- Gate decisions per issue time were recomputed by inference with the saved 2015-refit gate
  networks and thresholds. They reproduce `results/test/gates/sweep_test.csv` exactly: 300 of
  300 gate x budget x seed combinations (`planned_summary.json`, `decision_check`).
- The analysis settings (routing gate and budget, outage fractions, calibration bins, gate pairs,
  sky-regime rule at 30-min resolution) were fixed after the test run. None of them enters the
  frozen claims.

| file | analysis |
|---|---|
| `a1_gate_oracle_gap.csv` | A1: each gate's RMSE gap to the oracle, and the share of the random-to-oracle gap it closes |
| `a2_per_horizon.csv` | A2: tiers and gates per horizon; share kept per horizon at 25% |
| `a3_routing_hour_month.csv`, `a3_routing_sky_regime.csv` | A3: escalation share by local hour and month, and by daily sky regime (uncertainty gate, 25%) |
| `a4_link_outage.csv`, `a4_link_outage_summary.csv` | A4: share kept when a fraction of escalations is refused, either independently or as whole days |
| `a5_gate_differences.csv`, `a5_gate_differences_summary.csv` | A5: paired day-block bootstrap of RMSE_A - RMSE_B for gate pairs |
| `a6_input_ablation.csv` | A6: the input ablation available from the frozen run, with validation values next to it |
| `a7_uncertainty_calibration.csv` | A7: predicted against realised mean absolute edge error, in deciles; Spearman and fit in `planned_summary.json` |

**A6 limitation.** The planned A6 (cloud tier without satellite, cloud tier without NWP) needs
two extra cloud trainings. It was not done, because nothing is retrained after the freeze. For
2016 only trees_ground against trees on all inputs is available; this is claim 1.

**A3 sky regimes.** The classes use the same thresholds as the audit, but on 30-min block kt, so
the variability cut is indicative only. No 2016 day fell in the "overcast" class.
