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

## 7. Amendments before Stage 6 (2026-10-08)

Recorded before any Stage 6 computation. Decisions are the author's, after Checkpoint B.

**Standing decisions**
- The cloud tier going forward is the step-4 average: gradient-boosted trees on the step-3
  inputs, averaged with the step-3 network.
  *Reason:* it is the first ladder step that passed (gain 0.0631, `results/go_no_go.json`).
- The go/no-go verdict and the chosen cloud tier do not change, whatever Stage 6a shows. No new
  edge tier is chosen; the edge tier stays the base MLP.
  *Reason:* changing them after seeing the control would be tuning on validation.

**Stage 6a: model-type control** (validation only, same rows and seeds as Stage 5)
- Train the same gradient-boosted tree model on four input sets:
  - edge inputs only (trees_ground);
  - ground + satellite;
  - ground + NAM;
  - all inputs.
- Write `results/tiers/control.csv` with RMSE per horizon and averaged, per seed, for:
  - the edge network;
  - the four tree models;
  - the step-4 average.
- Report two effects, with mean, median and range over seeds:
  - (a) the input effect: trees on all inputs against trees_ground;
  - (b) the model effect: trees_ground against the edge network.
  *Reason:* Stage 5 passed only with trees. This separates "the cloud inputs help" from "trees
  beat the MLP".

**Stage 6b: gates and budget sweep** (only after Checkpoint B2)
- **Unit of decision:** one issue time. An escalation returns the cloud forecast for all six
  horizons, and the budget is the share of issue times escalated.
  *Reason:* one request per issue time is the deployable unit.
- **Predictions:** both tiers' predictions on gate_fit and validation are out of sample (both
  were trained on 2014). They are saved once to `results/tiers/predictions/`, and gates are
  built only from those files.
- **Gate inputs:** on-device information only (ground history, clear-sky features, the edge
  forecast). Nothing from satellite, NAM or the cloud forecast.
  *Reason:* the gate runs on the device before any request is made.
- **Six gates:**
  - random: expected curve over many draws;
  - fixed-interval;
  - variability threshold on recent clear-sky-index variability;
  - uncertainty: a small on-device model predicting the edge tier's absolute error;
  - learned: a small int8-compatible network predicting edge error minus cloud error, summed
    over horizons;
  - oracle: escalate the issue times with the largest true benefit.
- **Fitting:** the learned and uncertainty gates are fitted on gate_fit only. Each gate's
  threshold for every budget is set on gate_fit and stored in configs. Gates are evaluated on
  validation, with the realised escalation rate reported next to each target.
- **Metrics** per budget and gate:
  - RMSE, MAE, skill over smart persistence;
  - share of gain retained = (RMSE_edge - RMSE_gate) / (RMSE_edge - RMSE_cloud). The oracle can
    exceed 100%.
- **Ramp subset:** defined in the config before anything is computed on it.
- **Intervals:** day-block bootstrap with the settings in the config. Mean, median and range
  over the five seeds.
- **Tests:**
  - budget accounting (realised against target rate);
  - gate inputs contain no cloud-side column;
  - thresholds come from gate_fit only.
- **For the freeze, not done now:** whether gates are refitted on all of 2015 before the test
  run. This is to be proposed with a reason at Checkpoint C.

**Stage 6a implementation details** (assistant's choices, fixed before computing)
- **Rows:** the primary rows of Stage 5 (daylight cells with a satellite frame under the
  15-minute rule). All daylight cells are reported as a second row set.
- **Tree model:** the same HistGradientBoosting set-up and settings as ladder step 4 (one model
  per horizon, iterations chosen on the early-stop slice). The tile is summarised the same way.
- **Input sets:**
  - ground: the edge inputs;
  - satellite: the newest tile and its age/present flags, with the base frame settings;
  - NAM: the step-3 NAM block, including the step-3 extras;
  - all: exactly the step-4 input set.
- **Effects** use RMSE per horizon, averaged over horizons:
  - effect = 1 - RMSE_A / RMSE_B, positive when A is better, also given in W/m2;
  - (a) A = trees all inputs, B = trees_ground;
  - (b) A = trees_ground, B = the edge network, seed by seed.
- **Seeds:** the tree models are deterministic, so their five seeds are identical. The spread of
  effect (b) comes from the edge network seeds only. This is stated in the report.
- **Re-use:** the edge network and the step-3 network are not retrained. Their saved Stage 5
  validation predictions are re-used (training is deterministic). The trees on all inputs are
  recomputed and checked against Stage 5.

## 8. Amendments after Stage 6a (2026-10-08)

Recorded before any further work. Decisions 1 to 6 are the author's. The details below them are
the assistant's operational choices, also fixed before computing.

1. **Finding carried forward (validation only).** With the model type held fixed, the input
   effect of satellite and NAM is about zero (-0.27%, `results/tiers/control_effects.json`).
   The cloud tier's gain over the edge network is a model-capacity effect.
   - The study now measures escalation from a small on-device network to a stronger cloud
     model.
   - The input ablation is a main result, not an extended one.
   - The result is reported as found, and no further cloud variants are searched.
2. **Tiers unchanged.** Edge is the base MLP; cloud is the step-4 average. Neither is retuned.
3. **Footprint comparison before the gates** (`results/tiers/footprint_trees.json`):
   - trees_ground: tree count, depth, node count, serialised size, single-row CPU inference time;
   - edge network: parameter count, float32 size, estimated int8 size;
   - two size-capped ground-only tree models, at about 10x and 100x the edge int8 size, with
     validation RMSE per horizon;
   - a plain statement of whether a ground-only tree model of on-device size matches the cloud
     tier.
4. **Stage 6b as specified in section 7**, plus one reference curve: escalating to trees_ground
   instead of the cloud tier.
5. **Every report** gives the paired day-block bootstrap interval for edge against cloud RMSE,
   and says whether it includes zero.
6. **Commits:** commit and push after the footprint step and after the gates.

### Operational details (assistant's choices, fixed before computing)

**Footprint**
- Caps are multiples of the edge network's estimated int8 size: `footprint.cap_multiples`
  [10, 100].
- Tree size on device is estimated as node count x 12 bytes (`footprint.bytes_per_node`):
  a 2-byte feature index, a 4-byte threshold (or leaf value), two 2-byte child indices and
  2 bytes of padding. The pickle size is reported as well, but it includes sklearn overhead.
- A capped model uses `max_leaf_nodes` = 8 (10x cap) or 31 (100x cap). Each of the six horizon
  models gets one sixth of the node budget, giving a maximum of floor(budget / (2L-1)) trees.
  The early-stop slice of models_train then picks the best number of trees up to that maximum.
  Everything else is as in step 4.
- Edge int8 size estimate: int8 weights (1 byte), int32 biases (4 bytes), and 8 bytes per
  tensor for scale and zero point.
- **"Matches the cloud tier"** means the paired day-block bootstrap 95% interval of
  RMSE_trees - RMSE_cloud (averaged over horizons, primary validation rows) lies at or below
  zero, or includes zero, for the cloud seed in question. Reported per cloud seed.
- Single-row CPU time: the median of 200 repeats on one validation row, through all six
  horizon models. It measures Python and sklearn call overhead on a PC CPU, not
  microcontroller latency.

**Bootstrap** (`bootstrap` in config): day blocks are local standard-time days (08 UTC to
08 UTC), with 2,000 resamples, random seed 0 and 95% percentile intervals. Every model in a
comparison is resampled with the same days, which makes the comparison paired.

**Stage 6b**
- **Prediction files** (`results/tiers/predictions/`):
  - per seed, the edge and cloud kt forecasts at every gate_fit and validation issue time;
  - trees_ground, which is deterministic, so a single set;
  - the truth (ghi, clear-sky, elevation, smart persistence, satellite availability);
  - the on-device inputs.
  The gate_fit rows are issue times in gate_fit whose targets stay inside it.
- **Evaluation population:** validation issue times in the Stage 5 primary set (satellite
  available, at least one daylight horizon). Budgets are shares of these issue times. The
  threshold population is the matching gate_fit issue times.
- **Gate inputs:** the edge inputs (ground history, weather, clear-sky and calendar terms)
  plus the edge kt forecast for the six horizons. A test forbids any column containing sat,
  nam, cloud or sif.
- **Gates:**
  - random: the mean over 200 draws, with the interval from the expected squared error;
  - fixed-interval: escalate when floor((i+1)b) > floor(ib), with i counting issue times
    within each day, so the rate is exactly b;
  - variability: score = `ih_V(ghi_kt|30min)`, the variability of the 5-min clear-sky index
    over the last 30 min;
  - uncertainty: a Linear+ReLU MLP (one hidden layer of 32) predicting the mean absolute edge
    error over daylight horizons (W/m2 / 100);
  - learned: the same network shape predicting the benefit B = sum over daylight horizons of
    (e_edge^2 - e_cloud^2) / 1e4, the total reduction in squared error from escalating;
  - oracle: the true B on validation, top share b.
- **Fitting the uncertainty and learned gates:** on gate_fit, with its last 30 days used only
  for early stopping, MSE loss, one gate per seed.
- **Thresholds:** for each score-based gate and budget, the (1-b) quantile of its scores on the
  gate_fit population. They are written to `configs/gate_thresholds.yaml` and read back for
  validation.
- **Blending:** an escalated issue time takes the cloud forecast (or trees_ground, for the
  reference curve) for all six horizons.
- **Metrics:** per horizon, then averaged as in Stage 5. Share of gain retained =
  (RMSE_edge - RMSE_gate) / (RMSE_edge - RMSE_cloud), always relative to the cloud tier's gain,
  so the trees_ground curve is on the same scale.
- **Ramp subset** (`ramp` in config): a ramp cell is an (issue time, horizon) pair whose
  30-min target kt differs from the previous 30-min window's kt by at least 0.25 in absolute
  value. For the 30-min horizon, the previous window is the last observed block,
  `B(ghi_kt|30min)`. Gate metrics are reported on ramp cells as a separate row set.
- **Intervals:** at the report budgets (0.10, 0.25, 0.50), day-block bootstrap intervals for
  each gate's RMSE and share retained, per seed. They are summarised as mean, median and range
  over seeds.
