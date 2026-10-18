# Method facts (read-only check)

Each answer quotes or paraphrases code, config or `docs/FREEZE.md`, with `file:line`. Where none of them states the
fact, the answer is **not stated**; nothing is inferred. Numbers shown as stored values come from the result file
named. "Computed here" marks the only values worked out for this note, with how.

## 1. Edge MLP

| item | fact | source |
|---|---|---|
| architecture | `nn.Linear` + `nn.ReLU` per hidden layer, then a linear head: "MLP 96 → 64 → 64 → 6 (ReLU), 10,758 parameters, fp32" | `src/escal/models.py:8-24`; `configs/base.yaml:72-73`; `docs/FREEZE.md:73` |
| activation | ReLU (hidden layers); no activation on the output layer | `src/escal/models.py:11`, `:21` |
| target | the benchmark clear-sky index `ghi_kt_{h}` for the 6 horizons; mask = daylight (elevation >= 5 degrees and finite kt) | `src/escal/features.py:180`, `:183`, `:186`; `src/escal/models.py:1` ("Both predict the benchmark kt target") |
| output post-processing | predictions clipped to kt [0, 1.2], then multiplied by `ghi_clear_h` for W/m2 | `configs/base.yaml:58`; `scripts/tiers.py:84`; `scripts/gates.py:92`; `docs/FREEZE.md:67` |
| loss | masked MSE on kt: `((pred - y) ** 2 * m).sum() / m.sum()` | `src/escal/train.py:30-31`; `docs/FREEZE.md:73` ("Masked kt MSE") |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, batch 256 | `src/escal/train.py:44`; `configs/base.yaml:67-69` |
| epochs / early stopping | at most 300 epochs; stop after 20 epochs without improvement (> 1e-6) of the early-stop loss; the best-epoch weights are restored | `src/escal/train.py:48`, `:63-70`; `configs/base.yaml:70-71` |
| early-stop slice | the last 30 days of models_train (2014), used only for early stopping | `configs/base.yaml:59`; `docs/FREEZE.md:33` |
| int8: method | **static** quantisation with ONNX Runtime `quantize_static`: QDQ format, weights `QInt8`, activations `QUInt8`, `per_channel=False`, `CalibrationMethod.MinMax` | `scripts/measure_footprint.py:82-85`; `configs/base.yaml:116-119`; `docs/PLAN.md:375-377` |
| int8: export and calibration | PyTorch to ONNX with `torch.onnx.export`, opset 17; calibration rows = models_train only (train + early-stop slice) | `scripts/measure_footprint.py:76-79`, `:217`, `:233-234`; `docs/PLAN.md:378-380` |
| int8: tool versions | onnxruntime 1.30.0; torch 2.6.0+cu124 | `requirements.lock.txt:25`, `:46` |

## 2. Cloud tier

| item | fact | source |
|---|---|---|
| composition | "0.5 × trees + 0.5 × step-3 network" | `docs/FREEZE.md:74` |
| network architecture | `CloudNet`: CNN over the tile stack (Conv2d 3x3, padding 1, ReLU, channels 16 then 32), then the spatial mean over the 10x10 map; an MLP over the tabular inputs (hidden 128, 64, ReLU); concatenation of the two; head Linear(64 + 32 → 64), ReLU, Linear(64 → 6) | `src/escal/models.py:27-42`; `configs/base.yaml:74-77` |
| how the tile enters | the step-3 variant = `base_variant` + `step3`: 1 frame, no differences, 60-min lookback, `nwp_extra: true`; the tile is (N, 1, 10, 10), pixels / 255, newest frame stamped in [t-75, t-15] min, fed to the CNN branch; its age (h) and a present flag go into the tabular branch | `scripts/test_run.py:108-109`; `configs/base.yaml:78-85`; `src/escal/features.py:87-111`; `scripts/tiers.py:56-65`; `docs/FREEZE.md:56` |
| tabular inputs | ground (edge) inputs + NAM features + satellite age/present, standardised with models_train training-row statistics | `scripts/tiers.py:62-63`; `docs/FREEZE.md:78` |
| network loss and training | the same `fit()` as the edge: masked kt MSE, AdamW lr 1e-3, wd 1e-4, batch 256, at most 300 epochs, patience 20, early stop on the last 30 days of models_train | `scripts/tiers.py:68-85`; `src/escal/train.py:30-70` |
| trees | HistGradientBoosting, one model per horizon, learning rate 0.05, 31 leaves, at most 500 iterations, iteration count chosen on the early-stop slice by staged prediction, then refit; inputs = step-3 tabular inputs + a tile summary (per-channel mean, std, and mean of the central 4x4) | `scripts/tiers.py:189-219`; `configs/base.yaml:86-90`; `docs/FREEZE.md:74` |
| averaging ("in kt") | both members predict kt, each clipped to [0, 1.2]; the average is `(trees_all + step3_network) / 2` on the kt arrays (float32), and only then multiplied by `ghi_clear_h`. "In kt" means the average is taken of the clear-sky-index forecasts, not of W/m2 | `scripts/test_run.py:148`, `:152-156`; `docs/FREEZE.md:74`, `:84-86` |

## 3. Satellite, NAM, availability

| item | fact | source |
|---|---|---|
| tile size | GOES-15 "10 x 10 tiles" | `docs/FREEZE.md:53`; `src/escal/features.py:95` |
| 15-min availability lag | confirmed: a frame may be used only if stamped at or before t - 15 min; in code `end = t - lag` and the newest frame stamped at or before `end` is taken. The 15 min is an assumption: "assumed delay from frame time stamp to availability; unverified" | `docs/FREEZE.md:53-54`, `:283-286`; `configs/base.yaml:46`; `src/escal/features.py:91-93` |
| "available" | a frame stamped in [t-75 min, t-15 min] | `docs/FREEZE.md:55`; `configs/base.yaml:60`; `src/escal/features.py:114-116` |
| NAM variables | eight fields: `dwsw`, `cloud_cover`, `precipitation`, `pressure`, `wind-u`, `wind-v`, `temperature`, `rel_humidity`, from the four node files; plus dwsw / `ghi_clear_h` clipped to [0, 2] | `src/escal/features.py:14`, `:129-136`; `docs/FREEZE.md:58`, `:62-63` |
| NAM step-3 extras | NAM at t-15 min, dwsw and cloud cover at valid time ± 1 h, and the NAM error at issue time | `docs/FREEZE.md:64-65`; `src/escal/features.py:145-161` |
| NAM run choice | **not simply the latest run before t**: "the newest run with reftime + lag <= t whose valid times span v", lag = 6 h ("assumed delay from 12Z reference time to availability; unverified"); valid time v = t + h - 15 min; linear interpolation inside that run up to a 3-h gap; runs never mixed | `src/escal/nwp.py:10-23`; `configs/base.yaml:44`, `:62-63`; `src/escal/features.py:131-132`; `docs/FREEZE.md:59-61` |

## 4. Gates

| item | fact | source |
|---|---|---|
| unit, escalation | one decision per issue time; an escalation returns the cloud forecast for all six horizons (`blend`) | `docs/FREEZE.md:113-114`; `src/escal/gates.py:1-2`, `:60-61` |
| input (uncertainty, learned) | the 96 edge inputs plus the edge kt forecast for 6 horizons (102 columns), standardised with the gate's training-row mean and std, missing values set to 0; a check rejects any column containing sat, nam, cloud or sif | `scripts/gates.py:94-97`; `src/escal/gates.py:8-13`; `configs/base.yaml:110`; `docs/FREEZE.md:115-117` |
| output | one score per issue time (network output size 1); escalate if score >= threshold | `scripts/gates.py:46`, `:52`; `src/escal/gates.py:30-33` |
| thresholds | per gate, seed and budget b: the (1-b) quantile of the gate's scores on the fitting population; b = 0 never escalates, b = 1 always | `src/escal/gates.py:16-27`; `scripts/gates.py:107`; `docs/FREEZE.md:128-130` |
| random | exactly round(b·n) issue times drawn without replacement; result = mean over 200 draws | `src/escal/gates.py:54-57`; `scripts/gates.py:136-148`; `configs/base.yaml:108`; `docs/FREEZE.md:119` |
| fixed-interval rule | escalate when floor((i+1)b) > floor(ib), i counting issue times within each day in time order (day = local 08-08 UTC block) | `src/escal/gates.py:36-43`; `scripts/gates.py:76`, `:151`; `docs/FREEZE.md:120-121` |
| variability rule | score = `ih_V(ghi_kt\|30min)` (missing set to 0); threshold source: the (1-b) quantile of that feature on the fitting population, like the other score gates. FREEZE.md notes `V(ghi_kt\|w)` itself is undefined in the benchmark | `configs/base.yaml:109`; `scripts/gates.py:104`, `:107`; `docs/FREEZE.md:122`, `:295` |
| uncertainty gate | predicts the mean absolute edge error over daylight horizons, in W/m2 / 100; target = `nanmean(\|y - E\|)` over daylight horizons / 100 | `scripts/gates.py:101-102`, `:105`; `configs/base.yaml:113`; `docs/FREEZE.md:123-124` |
| learned gate | predicts the escalation benefit: target = sum over daylight horizons of (e_edge^2 - e_cloud^2) / 1e4 | `src/escal/gates.py:64-67`; `scripts/gates.py:103`, `:106`; `configs/base.yaml:114`; `docs/FREEZE.md:125-126` |
| gate loss | MSE: `fit()` with an all-ones mask, i.e. the masked MSE of item 1 reduces to plain MSE | `scripts/gates.py:47-49`; `src/escal/train.py:30-31` |
| gate network size | `EdgeNet(n_in, 1, [32])`: 102 → 32 (ReLU) → 1, "one hidden layer of 32, ReLU" | `scripts/gates.py:46`; `configs/base.yaml:111`; `docs/FREEZE.md:123` |
| gate optimiser, epochs | the shared `fit()`: AdamW lr 1e-3, wd 1e-4, batch 256, at most 300 epochs, patience 20 | `src/escal/train.py:44-70`; `configs/base.yaml:67-71` |
| training split (test run) | gate_fit + validation = all of 2015 (`fit_parts=("gate_fit", "val")`), rows with a satellite frame and >= 1 daylight horizon | `scripts/test_run.py:47`; `scripts/gates.py:68-69`; `docs/FREEZE.md:39-41` |
| early stopping | the last 30 days of the fitting period (last 30 days of 2015 in the test run); 6,874 training rows and 462 early-stop rows | `scripts/gates.py:78-81`; `configs/base.yaml:112`; `docs/FREEZE.md:40-41`; `results/test/gates/meta.json:5-6` |
| oracle | the top round(b·n) issue times by true benefit (same benefit as the learned gate's target, from 2016 truth); "a headroom figure only" | `src/escal/gates.py:46-51`, `:64-67`; `scripts/gates.py:152-153`; `docs/FREEZE.md:127` |

## 5. Decided issue times

| item | fact | source |
|---|---|---|
| rule | evaluation issue times with an available satellite frame (15-min rule) and at least one daylight horizon: `pop = sat_available & day.any(1)` | `scripts/gates.py:68`, `:70`; `results/test/gates/meta.json:2` |
| count | 7,457 | `results/test/gates/meta.json:3` |
| context | 2016 has 8,068 issue times, 7,891 with at least one daylight horizon | `results/test/data_quality.json:7-8` |
| non-decided daylight issue times | 434 (7,891 - 7,457; computed here from `results/test/predictions/truth.parquet`: `part == "test"`, any `day_*`, `sat_available` false) | computed here |
| what happens to them | they are outside the gate population: no gate decides them and no gate metric includes them; their cells enter only the "secondary" row set (all 2016 daylight cells), which is used for tier comparisons. What forecast a deployed system would issue for them: **not stated** | `scripts/gates.py:84`; `docs/FREEZE.md:136-139`; `scripts/test_run.py:247` |
| also dropped | issue times whose furthest target (t+180 min) crosses the end of their period | `docs/FREEZE.md:38` |

## 6. Latency

| item | fact | source |
|---|---|---|
| tool, networks | ONNX Runtime 1.30.0, `CPUExecutionProvider` | `scripts/measure_footprint.py:45-50`; `requirements.lock.txt:25` |
| tool, capped trees | generated C compiled with `python -m ziglang cc -O2` (zig 0.16.0), timed in a compiled benchmark with `QueryPerformanceCounter` (timer resolution 0.1 µs) | `configs/base.yaml:124`; `scripts/measure_footprint.py:181-205`, `:321-329`; `results/footprint/measured.json` (`trees_ground_cap10x.generated_c`) |
| CPU model | **not stated** (the record says only "a PC CPU") | `results/footprint/measured.json` (`note`); `docs/PLAN.md:391` |
| threads | 1 intra-op and 1 inter-op thread, sequential execution | `scripts/measure_footprint.py:47-49`; `configs/base.yaml:122`; `docs/PLAN.md:388` |
| warm-up, repeats | 200 warm-up runs, then 5,000 timed runs | `configs/base.yaml:120-121`; `scripts/measure_footprint.py:58-67`; `docs/PLAN.md:389` |
| statistic | median (also p95) of per-run wall time (`perf_counter_ns`) | `scripts/measure_footprint.py:64-68`; `docs/PLAN.md:389` |
| input | single row: ONNX timings repeat the first validation row; the C benchmark cycles through validation rows (`i % n`) | `scripts/measure_footprint.py:221`, `:246-247`, `:289-290`, `:197` |
| which models | edge and gates: seed 0 only. Gates timed are the gate_fit versions (`checkpoints/gates`), not the 2015-refit gates used on 2016 | `scripts/measure_footprint.py:34`, `:245`, `:270`, `:288` |
| stored values | edge fp32 7.5 µs, edge int8 9.6 µs, uncertainty gate int8 8.6 µs, learned gate int8 8.5 µs, capped trees C 3.5 µs (medians) | `results/footprint/measured.json` |

## 7. Footprint

All sizes are file sizes on disk in bytes; the paper uses decimal kB (1 kB = 1,000 B) (`results/paper/paper_table.md`, table note).

| number | what it is | stored value | source |
|---|---|---|---|
| 43.8 kB | the edge network's **fp32 ONNX file** (`edge_seed0.onnx`) | 43,846 B (`edge_network.per_seed[0].fp32_onnx_bytes`; same for all seeds) | `scripts/measure_footprint.py:232-240`; `scripts/core_figures.py:137`; `results/footprint/measured.json` |
| 15.2 kB | the edge network's **int8 ONNX file** (`edge_seed0.int8.onnx`) | 15,177 B (`int8_onnx_bytes`) | `scripts/measure_footprint.py:232-240`; `scripts/core_figures.py:139` |
| 6.1 kB | one gate's **int8 ONNX file** (uncertainty and learned gates are the same size); gate_fit versions | 6,067 B (`gates.per_seed[0].{uncertainty,learned}.int8_onnx_bytes`) | `scripts/measure_footprint.py:271-275`; `scripts/core_figures.py:150` |
| 87.2 kB | the **compiled object file** `trees_cap10x.o` of the generated C for the 10x-capped ground-only trees (static node arrays + the `ensemble`/`trees_predict` functions), compiled with `-c -O2` | 87,183 B (`trees_ground_cap10x.generated_c.object_bytes`); for comparison: static data 71,920 B, C source 222,240 B, 12-B/node estimate 85,140 B | `scripts/measure_footprint.py:149-178`, `:322`, `:334`; `scripts/core_figures.py:141`; `docs/FREEZE.md:76` |
| not counted | whether runtime libraries (ONNX Runtime, C runtime) or feature-computation code are included: **not stated**; the numbers above are the model files only | | |

## 8. Bootstrap

| item | fact | source |
|---|---|---|
| block | one local standard-time day, 08 UTC to 08 UTC; only days with selected cells | `src/escal/bootstrap.py:8-11`, `:46-50`; `configs/base.yaml:21`; `docs/FREEZE.md:155-156` |
| resamples, seed | 2,000 resamples, numpy `default_rng(0)`; each resample draws n_days days with replacement (multinomial day multiplicities) | `src/escal/bootstrap.py:25-29`; `configs/base.yaml:22-23` |
| paired how | per-day, per-horizon sums of squared errors and cell counts are formed for both models; each resample applies the **same** day multiplicities to both, recomputes RMSE per horizon, averages over horizons, and takes the difference RMSE_a - RMSE_b | `src/escal/bootstrap.py:14-22`, `:32-36`, `:45-56` |
| interval | 95% percentile interval of the resampled differences; "includes zero" if lo <= 0 <= hi | `src/escal/bootstrap.py:39-42`; `configs/base.yaml:24` |

## 9. A4 link-outage test (post-test planned analysis)

| item | fact | source |
|---|---|---|
| purpose | "Graceful degradation when escalation is refused", post hoc on saved predictions | `docs/PLAN.md:57` |
| gates, budgets, outage fractions | uncertainty, learned, variability, random; targets 10, 25, 50%; outage fraction q = 0, 0.10, 0.25, 0.50, 1.0 | `configs/analyses.yaml:16-19` |
| decisions | the gate decisions are unchanged (recomputed from the saved 2015-refit gates and thresholds, checked against `sweep_test.csv`); an escalation that is refused gets the edge forecast | `scripts/planned_analyses.py:34-37`, `:114`, `:20-25`; `scripts/analysis_common.py` (`verify_decisions`) |
| independent outages | each escalation refused independently with probability q; RMSE computed as an expectation (escalation weight `esc * (1 - q)`), no random draw | `scripts/planned_analyses.py:116`, `:20-25` |
| day-block outages | each local day (08-08 UTC) is down with probability q; 200 random draws (rng seed 0), RMSE averaged over draws | `scripts/planned_analyses.py:107-108`, `:117-121`; `configs/analyses.yaml:20-21` |
| metric | share kept = (RMSE_edge - RMSE) / (RMSE_edge - RMSE_cloud) per seed; summary = mean over 5 seeds | `scripts/planned_analyses.py:124-125`, `:128` |
| the 0.514 | uncertainty gate, 25% target, 25% of days down: `share_kept_day_block` = 0.5143 (independent outages: 0.5122; no outage: 0.6878; 100% outage: 0) | `results/analyses/a4_link_outage_summary.csv` (row uncertainty, 0.25, 0.25); `docs/number_trace.md:198` |
| status | planned analysis A1-A7, fixed after the test run; not part of the frozen claims | `configs/analyses.yaml:1-2`; `results/analyses/PLANNED.md` |

## 10. Ramp cell (claim 5)

| item | fact | source |
|---|---|---|
| definition (frozen) | "**ramp:** cells with \|kt(t,h) - kt(t,h-30 min)\| >= 0.25, where kt(t,0) is `B(ghi_kt\|30min)` (`ramp.delta_kt`)." | `docs/FREEZE.md:140-141` |
| variable | the benchmark clear-sky index of the **target** (observed), `kt_{h}` for each horizon, from the truth file; not a forecast | `scripts/test_run.py:223`; `src/escal/features.py:180` |
| window / step | the change between consecutive horizons 30 min apart (each kt is the mean over its 30-min target window). For h = 30 min the previous value is `B(ghi_kt\|30min)`, the measured mean kt over the 30 min before the issue time | `scripts/test_run.py:224`; `configs/base.yaml:106`; `docs/FREEZE.md:26` |
| threshold | absolute change >= 0.25 in kt | `configs/base.yaml:106`; `scripts/test_run.py:225` |
| which cells | primary cells only (daylight cells at the 7,457 issue times with a satellite frame): `ramp = prim & (...)` | `scripts/test_run.py:220-221`, `:225`; same rule in `scripts/gates.py:72-74` |
| count | 2,551 ramp cells in 2016 | `results/test/claims_test.json` (`ramp_cells`); `results/test/gates/meta.json:7` |

## 11. Was the A4 link-outage analysis pre-specified?

- **Listed before the test run, as a name only.** At tag `freeze-2026-10-08`, `docs/FREEZE.md:274` (current file: `:275`)
  says: "**Analyses after the run.** Only A1 to A7 in `docs/PLAN.md` section 3 count as planned." At the same tag,
  `docs/PLAN.md:57` lists: "| A4 | Link-outage test | Graceful degradation when escalation is refused | post-hoc on saved
  predictions |". The A4 row has been in `docs/PLAN.md` since `ad925df` (2026-10-07 21:38:47 +0900), before the
  freeze commit `3c07703` (2026-10-08 09:38:55 +0900) and the run (2026-10-08 00:39:24 UTC).
- **Setup fixed after the run.** The gates, budgets, outage fractions (0, 0.10, 0.25, 0.50, 1.0), day-block draws (200)
  and seed are in `configs/analyses.yaml:16-21`, whose header says "Fixed on 2026-10-08 AFTER the test run; none of
  these settings enters the frozen claims." (`configs/analyses.yaml:1-2`). FREEZE.md does not describe the A4 setup
  (no "outage" or "link" in the file).
- So A4 counts as planned under FREEZE.md's rule, but its design (and so the 0.514) was fixed after 2016 was seen.

## 12. CPU model

**Not recorded.** Searched `results/footprint/measured.json`, `results/tiers/footprint_trees.json`,
`results/tiers/FOOTPRINT.md`, `logs/footprint/run.log`, `logs/tiers/footprint.log`, `results/test/console.log` and
all of `logs/` and `results/` for CPU, processor, Intel, AMD, Ryzen, Core(TM) and GHz. The only record is
`measured.json` `note`: "Latency: single row, one thread, ONNX Runtime or compiled C on a PC CPU (CPU provider
CPU)." The "cpu" hits in `logs/footprint/run.log` are ONNX Runtime warning URLs; the others are byte matches inside
parquet files. `logs/` is not in git (only `logs/.gitkeep` is tracked).
