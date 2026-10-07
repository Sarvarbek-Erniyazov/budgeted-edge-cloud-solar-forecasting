# Plan: 8 to 18 October 2026

Source of truth for topic, split, gates and metrics: the decision notes
(claude/ICONI2026_decision_notes.md in the project). This file refines the schedule.
Nothing has been run; no number in this file is a result.

## 1. Day by day

| Date | Work | Done when |
|---|---|---|
| Thu 8 | Data audit: load every CSV, schema, time zone, gaps, daylight filter, clear-sky index, seasonal mix of sky conditions; read the horizons from Target_intra-day.csv; freeze the split | `results/audit/` written; `split.provisional: false`; clear-sky and split tests pass |
| Fri 9 | Reproduce the benchmark's persistence and linear models with its own scripts; train edge and cloud tiers (float) on 2014 | Benchmark numbers reproduced on development data; **go/no-go** recorded in `results/go_no_go.json` |
| Sat 10 | Out-of-sample tier predictions for 2015; six gates; budget sweep on validation | Sweep CSV with target and realised rates; gate and budget tests pass |
| Sun 11 | int8 quantisation of edge tier and learned gate; size, latency, float-to-int8 change | `results/footprint.json` |
| Mon 12 | **Freeze**: tag the commit, set `protocol.frozen: true`, run 2016 once with five seeds; bootstrap intervals; ramp subset | `results/test/` written from the tagged commit; no code change after the tag except figures |
| Tue 13 | Core figures and table; extended analyses A1 to A5 | `figures/` in vector and PNG |
| Wed 14 | Write the two pages; trim the abstract | Word file at 2 pages |
| Thu 15 | Clean-clone re-run; every number checked against `results/`; draft to Prof. Lim | Checklist in `docs/number_trace.md` |
| Fri 16 | Buffer: comments; extended analyses A6 to A7 | |
| Sat 17 | Final edit; form abstract updated | |
| Sun 18 | Submit in the morning | |

## 2. Checkpoints

- **Go/no-go, 9 Oct.** On validation (July to December 2015) the cloud tier must beat the
  edge tier by about 5% RMSE, averaged over horizons (suggested threshold). The result is
  written to `results/go_no_go.json`. If it fails, try these in order, on validation only,
  and stop at the first that passes. The test year stays locked throughout.
  1. Input check: confirm the cloud tier really receives the satellite tiles and all four
     NAM nodes, correctly time-aligned and with no future values.
  2. Longer satellite window: more past tiles, plus simple tile differences as a motion cue.
  3. Richer NWP features: all four nodes, cloud cover and irradiance fields, and the
     forecast valid at each target time, not only the latest run.
  4. Stronger cloud model: a gradient-boosted tree model on the same engineered features,
     alone or averaged with the network.
  5. Horizon focus: if the gap exists only at the longer horizons, restrict the study to
     those and say so.
  Deadline: noon on 10 Oct. If nothing passes, the paper reports that ground sensing is
  nearly sufficient for this task at this site, with the escalation curve as evidence.
  Every variant tried is logged in `results/go_no_go.json`, including the ones that failed.
- **Freeze, 12 Oct.** Before the test year is opened: split, features, model sizes, gate
  definitions, thresholds, budget grid, metrics, ramp definition and bootstrap settings are
  fixed and committed under a tag. The test year is run once.

## 3. Analyses

Core (mandatory, in the paper): six gates; budget sweep; ramp subset; quantisation;
latency; day-block bootstrap intervals.

Extended, recommended, in priority order (cheap and informative first):

| # | Analysis | Why | Extra cost |
|---|---|---|---|
| A1 | Gate-to-oracle gap per budget | Shows how much headroom a better gate has | none; from the sweep |
| A2 | Per-horizon breakdown | The value of escalation should grow with horizon | none |
| A3 | Routing maps: hour by month, and by sky regime | The interpretable picture; portfolio figure | none |
| A4 | Link-outage test | Graceful degradation when escalation is refused | post-hoc on saved predictions |
| A5 | Paired day-block bootstrap of gate differences | Says which gate differences are real | reuses the bootstrap |
| A6 | Cloud-tier input ablation: no satellite, no NWP | Says which context is worth requesting | two extra cloud trainings |
| A7 | Calibration of the uncertainty gate | Explains why that gate works or fails | small |

Dropped:
- Per-season results as a headline: one test year gives each season too few days; kept
  only as a descriptive table.
- Separate Diebold-Mariano tests: redundant with A5.
- Sky-image cloud tier: 49 GB of images and a reported timestamp offset.
- Latency on a real microcontroller: no board is available. Report MCU-class footprint and latency measured on a PC CPU, and claim nothing about real hardware.

Comparisons: the benchmark's own persistence and linear models, under the same protocol.
CAPE is related work only; no numeric comparison with papers on other data.

## 4. Risks

| Risk | Response |
|---|---|
| Edge-to-cloud gap too small | Go/no-go rule above |
| Learned gate no better than the variability rule | Report it; the rule is cheaper to deploy |
| Leakage through the split or the gate | Guard in `src/escal/splits.py`; gates fitted on out-of-sample predictions; tests |
| Gate-fitting half-year lacks cloudy days | Checked in the audit before the split is frozen |
| Benchmark numbers do not reproduce | Stop and resolve before training anything |
| Two pages too tight | One figure and one table; extended set stays in the repository |
| Public repository during double-blind review | Accepted by the author on 7 Oct; see section 5 |

## 5. Public repository

The repository is public from the start, with real author names, by the author's decision
of 7 Oct 2026. Trade-off, noted once: ICONI review is double-blind, and a public repository
with the same title can reveal authorship to a reviewer who searches for it.

Keep out of the repository regardless: data and checkpoints (not committed), absolute paths
containing a user name, and any result that has no file in `results/`.

After the short paper, the extended analyses in section 3 are the basis for a full paper
(3 to 6 pages). Keep each one as its own script, config and results folder.

## 6. Amendments before Stage 5 (2026-10-07)

Recorded before any tier was trained. Decisions 1 to 7 are the author's, after reviewing
`results/audit/AUDIT.md` at Checkpoint A. The implementation details below them were chosen by
the assistant and are also fixed before training.

1. **Satellite latency.** The primary setting is `satellite.availability_lag_minutes = 15`: only
   frames stamped at or before t-15 min are used. A 0-minute variant is also trained and
   reported, labelled "benchmark convention (sensitivity)". The go/no-go is judged on the
   15-minute setting only.
   *Reason:* a GOES image is not available at its nominal time stamp. The benchmark's feature
   includes the frame stamped exactly t, which is optimistic.
2. **NAM.** Interpolate linearly in time between valid times of the same run, which fills the
   3-hourly part (leads 36 to 45 h), and add a binary flag for interpolated values. Runs are
   never mixed. All NAM fields are standardised, and NAM dwsw divided by the benchmark
   `ghi_clear_h` is added as a feature.
   *Reason:* hourly bracketing left 14 to 18% of daylight targets, at late-afternoon valid
   times, without NWP.
3. **Evaluation rows.** Both tiers are scored on exactly the same issue times.
   - Primary set: daylight issue times (elevation_h >= 5 degrees) where a satellite frame is
     available under the 15-minute rule.
   - Also reported: all daylight issue times, with the cloud tier receiving a satellite-missing
     flag.
   - The satellite availability rate on validation is reported before training.
   *Reason:* a like-for-like comparison, and an honest picture of the cases where the cloud
   tier has no satellite input.
4. **Targets and scoring.** Train on the benchmark's kt target and score in W/m2 with the
   benchmark's `ghi_clear_h`, exactly as the anchors do.
   *Reason:* tiers and anchors stay comparable.
5. **Anchors as a check.** Each tier is reported next to its linear anchor on the same rows:
   edge against lasso_endo, cloud against lasso_exo. If a tier does not beat its anchor, the
   report says so plainly.
   *Reason:* a learned tier that loses to a linear model is a warning sign.
6. **Fallback ladder, step 5.** The wording changes from "longer horizons" to "the horizons
   where the gap exists".
   *Reason:* the anchors (Stage 3) show the satellite gain is largest at 30 to 60 min.
   This change was made after seeing the linear anchors and before training any tier. The
   primary criterion is unchanged: about 5% RMSE gain, averaged over all six horizons.
7. **Per-horizon gaps.** The edge-to-cloud gap is always reported per horizon as well as
   averaged.

### Implementation details fixed before training (assistant's choices)
- **Satellite availability:** a frame is "available" at issue time t if one is stamped in
  [t-lag-60 min, t-lag]; the 60 min is `tiers.sat_max_age_minutes`.
- **NAM valid time** for horizon h is the midpoint of the target window, t+h-15 min. The run
  used is the newest with reftime + 6 h <= t whose valid times bracket that time. A value is
  flagged as interpolated when the bracketing valid times are more than 1 h apart. NaN after
  standardisation becomes 0, with a missing flag.
- **Edge inputs** (ground sensors and clear-sky only):
  - intra-day B/V/L features for GHI and DNI;
  - intra-hour (5-min) B/V/L features at t, verified to be backward-looking;
  - 30-min means of the site weather sensors over (t-30 min, t];
  - solar elevation and benchmark `ghi_clear_h` per horizon;
  - sine and cosine of day of year and hour.
  The network is an MLP with 6 outputs (kt per horizon), small enough for int8.
- **Cloud inputs:** the edge inputs, plus the newest available GOES-15 tile (10 x 10) with its
  age and a missing flag, plus NAM at each horizon (4 nodes, all fields, dwsw/clear, flags).
  The network fuses a small CNN for the tile with an MLP for the tabular inputs.
- **Training:**
  - loss: mean squared error on kt, masked where elevation_h < 5;
  - predictions clipped to [0, 1.2], the benchmark target range, for both tiers;
  - early stopping on the last 30 days of models_train, with targets that cross the slice
    boundary purged;
  - hyper-parameters in `configs/base.yaml` under `tiers`, not tuned.
- **Go/no-go statistic:** gain = 1 - RMSE_cloud / RMSE_edge. Each tier's RMSE is computed per
  horizon and averaged over horizons, then averaged over the five seeds. The check passes if
  gain >= `go_no_go.min_rmse_gain` (0.05).
- **Ladder mechanics** (validation only, in order, stop at the first pass):
  - Each step builds on the best cloud variant so far, measured by validation RMSE averaged over
    horizons.
  - The edge tier is not changed.
  - Step 1 runs automated input checks:
    - tiles present and stamped at or before t-lag;
    - all four NAM nodes present;
    - NAM reftime + 6 h <= t;
    - no feature built from data after t.
    A fault found is fixed and the step is re-scored; otherwise the step is logged as failed,
    with an unchanged gap.
  - Step 2: the last 4 tiles within 120 min, plus 3 successive tile differences.
  - Step 3 adds, on top:
    - NAM valid at t-15 min and at t+h-15 min ± 1 h;
    - the NAM error at issue time (NAM dwsw/clear minus the measured B(ghi_kt|30min)).
  - Step 4: gradient-boosted trees (scikit-learn HistGradientBoosting, one model per horizon) on
    the same engineered features, with the tile summarised as mean, std and centre 4 x 4 mean
    per frame. They are scored alone and averaged with the network.
  - Step 5: with the best variant, keep the horizons whose per-horizon gain is >= 5%. The step
    passes if their averaged gain is >= 5%, and the study is then restricted to those horizons
    and says so.
