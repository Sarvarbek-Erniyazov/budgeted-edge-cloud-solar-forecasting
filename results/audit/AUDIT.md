# Data audit (development period 2014-2015)

Produced by `./run.sh audit` (`scripts/audit_data.py`). Only rows dated before 2016-01-01 were
loaded: every frame goes through `escal.splits.dev_only` right after its time stamps are parsed,
and issue times whose targets reach 2016 are dropped as well. Every number here is in a file
under `results/audit/`, and the file is named next to it. Where something could not be verified,
this file says "unclear".

Files: `schema.json` (part a), `task_definition.md` and `task_checks.json` (b, c),
`daily_sky.csv`, `sky_mix.csv`, `sky_mix_by_split.csv`, `figures/audit/sky_mix.{png,svg}` (d),
`satellite_nam.json` (e).

## 1. Findings

### Files and time base (`schema.json`)
- All 17 CSVs parse. Two do not have the usual layout: `Folsom_satellite.csv` has no header
  row (a time stamp, then 100 pixel values), and each `Folsom_NAM_*.csv` starts with a title
  line ("12Z NAM forecast for Folsom ...") before its header. `Sat_image_features_intra-day.csv`
  repeats the column name `sat` 100 times; the loader renames these to `sat00` to `sat99`.
- Time stamps carry no time-zone marker. The Zenodo record states "All time stamps are in UTC",
  and the data agree. Median GHI is 0 for UTC hours 02 to 13. On the 40 smoothest days, the GHI
  centroid lies -0.36 ± 1.29 min from pvlib solar transit at the site below
  (`time_zone_evidence`).
- 1-min irradiance and weather: 1,029,600 development rows, with no duplicate time stamps.
  18,720 minutes are missing, on 16 days: 2014-01-30 to 02-05, 2014-02-13 to 02-19, and
  2015-08-23. There are 618 rows with missing values. The last development row is
  2015-12-31 07:59 UTC, so local 2015-12-31 has no development data. Whether this is a gap in
  the file cannot be checked without reading past the test boundary, and that was not done.
- Intra-day files (`Target_`, `Irradiance_features_`, `Sat_image_features_intra-day.csv`):
  15,961 development rows each, with identical time stamps (`task_checks.json`).
  4,955 satellite-feature rows (31%) are entirely NaN. Satellite frames start on 2014-03-13.
- `Folsom_sky_image_features.csv` has irregular time stamps and a 13.9-day gap from 2014-05-21.
  It is not used; sky images are out of scope (docs/PLAN.md).

### Task definition (`task_definition.md`, `task_checks.json`)
- Horizons: 30, 60, 90, 120, 150, 180 min. Issue times fall every 30 min, at UTC hours 13 to 02.
- In the row stamped t, `ghi_h` is the mean of the 1-min GHI over (t+h-30min, t+h]. This
  matches with a median absolute difference of about 3e-7 W/m2 at every horizon; the next-best
  window differs by 1.5 to 2.4 W/m2.
- `elevation_h` is the solar elevation averaged over the minute stamps t+h-29 ... t+h of that
  same window. The RMS difference is about 2e-5 degrees.
- Features use only data up to t. `B(ghi_kt|w)` is the mean of the 30-min block kt values over
  the w minutes ending at t, and `L(ghi_kt|w)` is the block ending at t-(w-30min). Both
  identities hold in 100% of the 14,531 rows checked.
- `V(ghi_kt|w)`: unclear. It is not the standard deviation of the blocks (that matches only
  0.4% of rows, with either ddof), and it is non-zero for a single block, so it is computed
  from finer data.
- `ghi_kt_h`: unclear. It is capped at 1.2, and it equals min(ghi/ghi_clear, 1.2) in only 37%
  of rows.
- `ghi_clear_h`: model unclear. None of the pvlib models tried reproduces it. Mean difference
  from it, over the 6 horizons (`clearsky_ours_minus_benchmark`):
  - Ineichen: -15.6 to -19.0 W/m2;
  - Haurwitz: +11.5 to +16.9 W/m2;
  - Simplified Solis: +10.8 to +13.0 W/m2.
- Smart persistence in the benchmark script is `B(ghi_kt|30min) * ghi_clear_h`, with forecasts
  at `elevation_h` < 5 degrees removed.

### Site coordinates and clear-sky index (`task_checks.json`, `daily_sky.csv`)
- Site: 38.6435 N, -121.1477 E, altitude 54 m (`configs/base.yaml`, `site`).
  - Source: neither the Zenodo record nor the benchmark scripts give the site coordinates.
    The coordinates were therefore fitted so that pvlib solar elevation, averaged over each
    target window, reproduces the benchmark's own `elevation_*` columns. The fit is exact to
    about 2e-5 degrees.
  - Why this answer is unique: an elevation-only fit cannot by itself separate longitude from
    the time convention. A fit at the window midpoint (t+h-15 min) leaves RMS 0.08 degrees and
    one at the window end (t+h) leaves 2.17 degrees; only the window-mean convention fits
    exactly.
  - The independent solar-transit check above agrees to within about 1 minute.
  - Altitude: pvlib's `lookup_altitude` at these coordinates; not verified against a survey.
  - The published coordinates in Pedro et al. (2019) could not be retrieved here (paywalled),
    so they are unverified against this fit.
- Clear-sky model: pvlib Ineichen with the pvlib Linke turbidity climatology. The clear-sky
  index uses a daylight filter of elevation >= 5 degrees and is capped at 1.2, as in the
  benchmark. Tests in `tests/test_clearsky.py`. 0 night minutes carry a kt value, and there
  are 0 non-finite values.
- **Surprise:** our kt is too high at low sun. The daylight median is 1.03 and 13% of daylight
  minutes sit at the 1.2 cap. On clear days, measured GHI divided by Ineichen has these medians
  (`clearsky_ratio_clear_days.csv`):

  | solar elevation | morning | afternoon |
  |---|---|---|
  | 5-10 degrees | 1.68 | 1.69 |
  | above 45 degrees | 1.03 | 1.02 |

  Morning and afternoon agree in every elevation band, which rules out a time shift. The cause
  is unclear: possibly turbidity, or the pyranometer's cosine response.
  - Consequence: kt from either clear-sky model saturates below about 20 degrees. Everything
    evaluated in W/m2 uses the benchmark's `ghi_clear_h`, so that forecasts stay comparable
    with the benchmark.

### Sky conditions (`sky_mix.csv`, `sky_mix_by_split.csv`, `figures/audit/sky_mix.*`)
- Days run from 08 UTC to 08 UTC (local standard-time days). Daily statistics use 10-min kt
  at elevation >= 10 degrees.
- Classes (thresholds in `configs/base.yaml` under `sky`, fixed before the counts were seen):
  - clear: mean kt >= 0.8 and variability < 0.03;
  - overcast: mean kt < 0.5 and variability < 0.03;
  - partly cloudy: everything else.
  Variability is the RMS of successive 10-min kt differences.
- The variability is clearly bimodal (`daily_sky.csv`): clear days sit near 0.01 and disturbed
  days at 0.07 to 0.2, so the 0.03 cut is not delicate.
- By split, among classified days:

  | split | classified days | clear | partly cloudy | overcast |
  |---|---|---|---|---|
  | models_train (2014) | 352 | 117 (33%) | 229 (65%) | 6 (2%) |
  | gate_fit (2015 H1) | 181 | 44 (24%) | 134 (74%) | 3 (2%) |
  | validation (2015 H2) | 182 | 68 (37%) | 113 (62%) | 1 (1%) |

- Surprises:
  - Overcast days by this rule are rare. Steady overcast is uncommon here, and the low-sun kt
    excess lifts the daily mean.
  - "Partly cloudy" is broad: any day with cloud passages lands there, including days that are
    mostly clear.
  - July 2014 has 20 partly-cloudy days.

### Satellite and NAM (`satellite_nam.json`)
- GOES-15 tiles: 30,364 development frames, each 10 x 10 pixels with values 0 to 255. There
  are no missing values and no duplicate time stamps.
  - Cadence is irregular: the modal step is 30 min, frames are mostly on :00 and :30 with some
    on :15 and :45, and there are gaps of up to 5.6 days.
  - 99.8% of frames fall on a minute that exists in the irradiance file.
- The benchmark's satellite feature at t equals the mean of the raw frames stamped in
  (t-30min, t] for 100% of the 11,006 non-empty rows. So no frame stamped after t is used, but
  a frame stamped exactly t is. A GOES image is not available at its nominal scan time; the
  real latency is unclear.
  - Our models will use only frames stamped at or before t-15min
    (`satellite.availability_lag_minutes`, an assumption).
  - With that lag, the newest usable frame at daylight issue times has median age 30 min, and
    82% of issue times have one within 60 min. With no lag the figures are 0 min and 84%.
- NAM, four nodes: 729 runs each, all at 12Z, with no day missing and no missing values. Each
  run has 14 valid times: leads 26 to 36 h hourly, then 39, 42 and 45 h.
  - Fields: dwsw, cloud_cover, precipitation, pressure, wind-u, wind-v, temperature,
    rel_humidity. Units are unclear; the file gives none.
    - dwsw ranges 0 to 1096, consistent with downward shortwave in W/m2 (unverified).
    - cloud_cover ranges 0 to 100.
    - temperature reaches 329.6, which is 56 C if the unit is K. That is implausible for 2-m air
      temperature, so what this field measures is unclear.
- **Which run is available.** At an issue time t, a run is usable if reftime + 6 h <= t (an
  assumed lag, unverified) and its valid times bracket t+h.
  - Under that rule every covered target uses the previous day's 12Z run. The issue time is
    always at least 25.5 h after the reftime, so no assumed lag below 25.5 h changes anything,
    and no future run can enter.
  - The same day's 12Z run never contains same-day valid times. It is the newest available run
    in 52% to 65% of cases, but it cannot be used.
  - Coverage of daylight targets is 82% (180 min) to 86% (30 min). The gaps are late-afternoon
    valid times between 00 and 03 UTC, where runs are only 3-hourly; hourly bracketing fails
    there. This rule is `escal.nwp.nam_at`, tested in `tests/test_nwp.py`.
  - NWP for this task is therefore a 26 to 39 h old forecast. Expect it to add little at
    intra-day horizons.

## 2. Recommendation on the split
- Keep the planned chronological split: models_train 2014, gate_fit 2015-01 to 06, validation
  2015-07 to 12, test 2016.
- gate_fit has 134 partly-cloudy days (74% of classified days), more than any other period, so
  the gate sees plenty of variable skies. Validation has 113 (62%).
- The two halves of 2015 differ in season: gate_fit is the wet half and validation the dry
  half. Gate thresholds fitted on gate_fit may therefore transfer imperfectly to validation;
  report this as a limitation rather than reshuffle.
- The final decision is in section "Split decision" below.
