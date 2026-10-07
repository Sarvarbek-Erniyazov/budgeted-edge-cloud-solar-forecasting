# Freeze proposal (approved with amendments 2026-10-08): NOT FROZEN

The author approved this proposal on 2026-10-08 with seven amendments, which are folded in
below and listed in `docs/PLAN.md` section 10. The freeze itself has not happened.
- `protocol.frozen` is false.
- `ESCAL_UNLOCK_TEST` is not set.
- No 2016 row has been loaded.
- **The repository cannot run the test year yet.** Section e lists the work needed before the
  freeze tag can be created.

Sources for every value below:
- `configs/base.yaml` and `docs/PLAN.md` (sections 6 to 9);
- `results/audit/AUDIT.md`;
- `results/go_no_go.json` and `results/tiers/`;
- `results/gates/`;
- `results/footprint/`.

---

## a. Fixed items

### Data and split
- **Data:** the Folsom dataset (Zenodo record 2826939). Time stamps are UTC.
- **Task:** the intra-day GHI task: horizons 30, 60, 90, 120, 150 and 180 min; issue times
  every 30 min (the row stamps of `Target_intra-day.csv`).
  - The target for horizon h is the 1-min GHI averaged over (t+h-30 min, t+h].
  - Scoring is in W/m2 with the benchmark's `ghi_clear_h`, on cells with elevation_h >= 5
    degrees.
- **Split** (`split`, `provisional: false`):

  | period | dates |
  |---|---|
  | models_train | 2014-01-01 to 2014-12-31; its last 30 days are used only for early stopping |
  | gate_fit | 2015-01-01 to 2015-06-30 |
  | validation | 2015-07-01 to 2015-12-31 |
  | test | 2016-01-01 to 2016-12-31 |

  Issue times whose furthest target (t+180 min) crosses the end of their period are dropped.
- **Gate refit for the test run** (`PLAN.md` section 9): the uncertainty and learned gates and
  all score thresholds are refitted on gate_fit + validation (all of 2015). Their early-stop
  slice is the last 30 days of 2015. The architecture, inputs and budgets are unchanged.

### Features
- **Edge inputs (96 columns),** all built only from data up to t:
  - intra-day B/V/L features for GHI and DNI (36);
  - intra-hour (5-min) B/V/L features at t (36);
  - site-weather means over (t-30 min, t] (7), plus a missing flag;
  - `elevation_h` (6);
  - `ghi_clear_h`/1000 (6);
  - sine and cosine of day of year and hour of day (4).
  They are standardised with models_train training-row statistics.
- **Satellite:**
  - GOES-15 10 x 10 tiles; a frame may be used only if it is stamped at or before t-15 min
    (`satellite.availability_lag_minutes: 15`);
  - "available" means a frame stamped in [t-75 min, t-15 min] (`tiers.sat_max_age_minutes: 60`);
  - the cloud tier uses the newest frame (pixels / 255), its age and a present flag.
- **NAM:**
  - the four node files, all eight fields, at valid time t+h-15 min;
  - the run used is the newest one with reftime + 6 h <= t whose valid times bracket that time;
  - values are interpolated linearly within that run, up to a 3-hour gap, with an
    interpolation flag; runs are never mixed;
  - dwsw / `ghi_clear_h` is added, clipped to [0, 2], and left missing where clear-sky is below
    10 W/m2;
  - the step-3 extras: NAM at t-15 min, NAM dwsw and cloud cover at valid time ± 1 h, and the
    NAM error at issue time;
  - all fields are standardised, and missing values become 0 with a flag.
- **Kt clip:** [0, 1.2] for every tier's predictions.

### Tiers and checkpoints (trained on models_train only)

| tier | definition | checkpoints |
|---|---|---|
| **edge** | MLP 96 → 64 → 64 → 6 (ReLU), 10,758 parameters, fp32. Masked kt MSE, AdamW (lr 1e-3, weight decay 1e-4), batch 256, at most 300 epochs, patience 20 | `checkpoints/tiers/base/edge_seed{0..4}.pt` |
| **cloud** | 0.5 × trees + 0.5 × step-3 network. Trees: HistGradientBoosting (one per horizon, learning rate 0.05, 31 leaves, at most 500 iterations, iterations chosen on the early-stop slice) on the step-3 inputs plus a tile summary. Network: the step-3 CloudNet | `checkpoints/tiers/step3/cloud_seed{0..4}.pt`; tree models **not yet saved** (section e) |
| **trees_ground** | the same tree set-up on the edge inputs only | **not yet saved** (section e) |
| **trees_ground, 10x cap** | 8 leaves, at most 103 trees per horizon (473 chosen); about 85 KB estimated, 87,183 B as a compiled object | **not yet saved** (section e) |
| **trees, all inputs** (claim 1) | the step-4 trees alone | **not yet saved** (section e) |

- **Edge tier, fp32 and int8:** the fp32 edge network is the primary edge tier, and every
  claim is judged on it. The int8 edge network (`checkpoints/footprint/edge_seed{s}.int8.onnx`,
  `results/footprint/measured.json`) is a **secondary row, reported next to fp32 in every table
  that shows the edge tier**.
- **Checkpoint SHA-256 hashes** (first 16 hex digits; the files are not in git):

  | file | edge_seed | step3 cloud_seed |
  |---|---|---|
  | seed 0 | e8897aa00dce9b90 | f74ab655aa0b9043 |
  | seed 1 | c8c23ed5f02dbd87 | d218ca5fbf493e08 |
  | seed 2 | 2ed41103739b9fe1 | f716561a5d9b9e0a |
  | seed 3 | 3951bf917a3752c9 | 5fe6f27a5d6f2fec |
  | seed 4 | 8aca79f44357b0e7 | 7fd91cc7b86fece9 |

  The full hashes are to be written to `docs/checkpoint_hashes.txt` at the freeze (section e).
  The gate checkpoints in `checkpoints/gates/` are the gate_fit versions. The test run
  replaces them with the 2015 refit.

### Gates
- **Unit and inputs:**
  - one decision per issue time; an escalation returns the cloud forecast for all six
    horizons;
  - gate inputs are the 96 edge inputs plus the edge kt forecast (102 columns, on-device
    only);
  - a test forbids any column containing sat, nam, cloud or sif.
- **The six gates:**
  - **random:** an exact count of round(b·n) issue times; the mean over 200 draws;
  - **fixed-interval:** escalate when floor((i+1)b) > floor(ib), with i counted within each
    day;
  - **variability:** score = `ih_V(ghi_kt|30min)`;
  - **uncertainty:** an MLP (one hidden layer of 32, ReLU) predicting the mean absolute edge
    error over daylight horizons / 100;
  - **learned:** the same shape, predicting the sum over daylight horizons of
    (e_edge^2 - e_cloud^2) / 1e4;
  - **oracle:** the top-b issue times by true benefit, a headroom figure only.
- **Thresholds:** the (1-b) quantile of each gate's scores on the fitting population (2015
  issue times with an available satellite frame and at least one daylight horizon). They are
  written to `configs/gate_thresholds.yaml` and read back.
- **Budgets:** targets 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0;
  reported at 0.10, 0.25 and 0.50.
- **Reference curve:** the same gate decisions, escalating to trees_ground.

### Evaluation
- **Rows:**
  - **primary:** 2016 daylight cells (elevation_h >= 5 degrees) at issue times with an
    available satellite frame under the 15-minute rule;
  - **secondary:** all 2016 daylight cells;
  - **ramp:** cells with |kt(t,h) - kt(t,h-30 min)| >= 0.25, where kt(t,0) is
    `B(ghi_kt|30min)` (`ramp.delta_kt`).
- **Reference models**, scored on the same 2016 rows:
  - smart persistence;
  - lasso_endo and lasso_exo, fitted on models_train only exactly as in Stage 3 (LassoCV 10-fold,
    kt clipped to [0, 1]).
  lasso_exo needs the benchmark satellite feature, so it is scored on the primary cells where
  that feature exists. It is reported with its cell count, next to every model on those same
  cells.
- **Metrics:**
  - RMSE and MAE per horizon, averaged over horizons;
  - skill = 1 - RMSE / RMSE(smart persistence), where smart persistence is
    `B(ghi_kt|30min)` × `ghi_clear_h`;
  - share kept = (RMSE_edge - RMSE_gate) / (RMSE_edge - RMSE_cloud);
  - gain = 1 - RMSE_cloud / RMSE_edge.
- **Bootstrap:** paired day-block bootstrap. Blocks are local standard-time days (08 to 08
  UTC) built only from days with selected cells; 2,000 resamples; seed 0; 95% percentile
  intervals.
- **Seeds:** 0, 1, 2, 3, 4. Every result is reported per seed, with mean, median and range.
  The tree models are deterministic, so their spread over seeds is zero.

---

## b. Claims to be tested on 2016

Every claim is evaluated on the primary rows unless stated. "Interval" means the paired
day-block bootstrap 95% interval. The equivalence margins (the 2% in claim 1, the "3 of 5
seeds" rules) are **proposed here, before 2016 is seen, and need the author's approval**.

**1. Input effect: trees on all inputs against trees_ground**
- *Claim:* with the model type fixed, satellite and NAM add nothing. Validation: -0.27%.
- *Metric:* input effect = 1 - RMSE_all / RMSE_ground (averaged over horizons), plus the
  interval of RMSE_ground - RMSE_all.
- *Counts against:* an input effect of at least +2% **and** an interval of
  RMSE_ground - RMSE_all lying entirely above zero.
- An effect of -2% or less (inputs hurt) does not count against the claim; it is reported.

**2. On-device sufficiency: the 10x-capped trees_ground against the cloud tier**
- *Claim:* a ground-only tree model of about 85 KB (estimated; 87 KB compiled object) matches
  the cloud tier. Validation: 68.78 against 68.66 W/m2; the interval includes zero for 5 of 5
  seeds.
- *Metrics:*
  - d = seed-mean (RMSE_cap10x - RMSE_cloud), with RMSE averaged over horizons, as a share of
    the seed-mean cloud RMSE;
  - the interval of RMSE_cap10x - RMSE_cloud, per cloud seed.
- *Verdict rule* (equivalence margin of 2%, amendment 1):
  - **"matches"** only if |d| <= 2% **and** the interval is not entirely above zero for 3 or
    more of the 5 seeds;
  - **"not distinguishable"** if the interval is not entirely above zero for 3 or more seeds,
    but |d| > 2%;
  - **"worse"**, which counts against the claim, if the interval lies entirely above zero for 3
    or more of the 5 seeds.

**3. Gap between the edge network and the cloud tier**
- *Claim:* the cloud tier beats the edge network. Validation: gain 0.063 (median seed 0.055);
  the interval excludes zero for 4 of 5 seeds.
- *Metric:* gain per seed and seed mean, plus the interval of RMSE_edge - RMSE_cloud per seed.
- *Counts against:* the interval includes zero or lies below zero for 3 or more of the 5
  seeds.
- The gain is also reported against the 5% go/no-go threshold. Falling below 5% is reported,
  but does not by itself refute the claim.

**4. Gate ranking and share kept at 10, 25 and 50% targets, with realised rates**
- *Claim (validation):*
  - uncertainty ≥ learned > variability > random ≈ fixed-interval in share kept;
  - at 25%: uncertainty 0.70, learned 0.61, variability 0.40, random 0.25.
- *Metrics:* share kept and realised rate per gate, budget and seed, with mean, median and
  range.
- *Counts against "an on-device gate beats random":* at the 25% target, the uncertainty gate's
  share kept exceeds random's by less than 0.10 in 3 or more of the 5 seeds.
- *Counts against "the learned gate is not better than uncertainty":* the learned gate's
  seed-mean share kept exceeds the uncertainty gate's by at least 0.10 at 2 or more of the 3
  budgets.
- *Counts against "refitting on all of 2015 fixes the budget shortfall":* for any score-based
  gate (variability, uncertainty, learned), the seed-mean realised rate differs from its
  target by more than 0.05 at any of 10, 25 or 50%.

**5. Ramp subset**
- *Claim (validation):*
  - the cloud tier is barely better than the edge on ramp cells (133.4 against 135.3 W/m2);
  - no gate shows a reliable benefit on ramps.
- *Metrics:* ramp-cell RMSE for edge, cloud and each gate; share kept on ramps; the interval
  of RMSE_edge - RMSE_cloud on ramp cells.
- *Counts against "no reliable cloud advantage on ramps":* the ramp-cell interval of
  RMSE_edge - RMSE_cloud lies entirely above zero for 3 or more of the 5 seeds.
- *Counts against "no gate helps on ramps":* some gate keeps a share of at least 0.5 at the
  25% target with a positive share for all 5 seeds.

---

## c. The single test run

### Commit
- The run uses the commit tagged **`freeze-<YYYY-MM-DD>`**, named with the date the tag is
  created. It does not exist yet.
- It is created only after every item in section e is done and the author approves this file.
- At the time of writing, HEAD is `926972e`. **That commit is not freezable**: it has no test
  path.

### Command sequence (proposed; the `test_*` stages are to be built in section e)
```bash
git checkout freeze-<YYYY-MM-DD>
source .venv/Scripts/activate
pip freeze | diff - requirements.lock.txt        # environment unchanged (pickled trees need sklearn 1.9.1)
./run.sh test                                    # unit tests must pass
sha256sum -c docs/checkpoint_hashes.txt          # checkpoints and tree models unchanged
# protocol.frozen: true is set in configs/base.yaml in the tagged commit itself
export ESCAL_UNLOCK_TEST=1
./run.sh test_predictions   # writes results/test/data_quality.json first, then 2016 predictions
./run.sh test_gates         # refit gates on all of 2015, thresholds, evaluate on 2016
./run.sh test_claims        # claims 1 to 5, smart persistence, lasso_endo, lasso_exo, int8 edge row, bootstrap
unset ESCAL_UNLOCK_TEST
```
All output goes to `results/test/`, together with the console log and the hash of the commit it
ran from.

**Data quality is recorded first.** Before any metric is computed on 2016, `test_predictions`
writes `results/test/data_quality.json`: row counts, gaps, satellite availability and NAM
coverage. The run proceeds whatever these show, and they are reported.

**Dry run.** `./run.sh dryrun` runs the same three stages with validation as the "test" period
and gates fitted on gate_fit only. Its output goes to `results/dryrun/`. It must reproduce
`results/gates/` and the validation claim values exactly before the tag is created.

### Forbidden afterwards
- Retraining, retuning, or changing any setting, feature, threshold, margin or row definition.
- Running the test year again with changed settings or code.
- Choosing between variants on the basis of 2016 results.

If a bug is found after the run, it is reported. Any re-run is labelled as such, with both
results kept.

**Allowed afterwards:** figures, tables and text built from the files in `results/test/`.

**Analyses after the run.** Only A1 to A7 in `docs/PLAN.md` section 3 count as planned. Any
other analysis of the saved 2016 predictions must carry "exploratory" in its file name and in
its text.

---

## d. Still unclear from the audit, and affecting the test run

1. **Satellite latency:**
   - the real delay between a GOES-15 time stamp and availability is unknown, so 15 minutes is
     an assumption;
   - the sensitivity run with 0 minutes moved the base network's gain by about 2 points.
2. **2016 data quality:**
   - 2016 gaps, satellite availability and NAM coverage are unknown until the year is loaded;
     the primary-row count may differ from validation (98.8% availability);
   - whether local 2015-12-31 (missing in development) is also missing at the 2016 boundary is
     unknown.
3. **Benchmark clear-sky and kt:**
   - `ghi_clear_h` comes from an unidentified clear-sky model;
   - how `ghi_kt_h` is built is unclear (it equals ghi/clear in only 37% of rows, cap 1.2);
   - `V(ghi_kt|w)` is undefined.
   All three are used as given, and the same file supplies 2016, so the scoring is consistent,
   but the definitions stay unverified.
4. **NAM fields:**
   - units are not stated;
   - `temperature` reaches 329.6, which is implausible as 2-m air temperature in Kelvin;
   - the fields are standardised, so this matters for interpretation, not for the pipeline.
5. **NAM availability lag:** 6 hours is assumed. Every run used is at least 25.5 hours old, so
   no lag below 25.5 hours changes the inputs.
6. **Low-sun clear-sky excess:** measured/Ineichen is about 1.7 at 5 to 10 degrees. It does not
   enter the models or the scoring (only the audit's sky classes), but it limits the ramp
   definition's meaning at low sun.

## e. Required before the freeze tag (not done)

1. Make the loaders able to read the test period, only when `select_split(..., "test")` is
   unlocked. Today every loader calls `dev_only`. Add a test that the test path raises
   `LockedTestYear` while the protocol is not frozen.
2. Save the fitted tree models (step-4 trees, trees_ground, 10x-capped trees_ground) as
   pickled files with SHA-256 hashes. Check that their development predictions reproduce the
   committed results exactly.
3. Write `test_predictions`, `test_gates` (2015 refit) and `test_claims`.
   - Dry-run them on development data, treating validation as the "test" period with gates fitted
     on gate_fit only.
   - Check that the dry run reproduces `results/gates/` and the validation claims.
4. Write `docs/checkpoint_hashes.txt` (full SHA-256 of every checkpoint and tree file).
5. `requirements.lock.txt` must equal the environment (`pip freeze`), with ziglang added and
   skl2onnx removed. This matters because the pickled tree models depend on the scikit-learn
   version, which is pinned at 1.9.1. Checked on 2026-10-08: identical.
6. The author approves sections a to d (done 2026-10-08, with amendments), checks the dry run,
   and then creates the tag.
