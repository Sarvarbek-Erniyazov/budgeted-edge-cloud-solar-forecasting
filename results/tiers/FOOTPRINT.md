# Footprint: ground-only trees against the edge network (validation)

Run with `./run.sh footprint` (`scripts/footprint_trees.py`). All numbers are in
`results/tiers/footprint_trees.json`; the rules are in `docs/PLAN.md` section 8.

## Sizes

| model | size | other |
|---|---|---|
| edge network | 10,758 parameters; float32 43,032 B; **int8 about 11,208 B** (estimate) | single-row CPU time 0.019 ms (PyTorch) |
| trees_ground (uncapped) | 517 trees, max depth 18, 31,537 nodes; compact about 378 KB (estimate), pickle 2.9 MB | 4.3 ms (sklearn) |
| trees_ground, 10x cap (112,080 B) | 473 trees (max 103 per horizon), 8 leaves per tree, max depth 7, 7,095 nodes; **compact about 85 KB (estimate)**, pickle 1.5 MB | 4.0 ms |
| trees_ground, 100x cap (1,120,800 B) | the cap did not bind: early stopping chose the same 517 trees as uncapped | 4.3 ms |

- Compact size = nodes x 12 bytes. This is an estimate for a plain array encoding, not a
  measured on-device file.
- CPU times are single-row times on a PC through Python. They are not microcontroller latency.

## Validation RMSE (W/m2), primary rows, 20,125 cells; seed mean where seeds apply

| model | 30min | 60min | 90min | 120min | 150min | 180min | mean |
|---|---|---|---|---|---|---|---|
| edge network | 51.88 | 64.05 | 72.89 | 79.26 | 83.35 | 88.26 | 73.28 |
| cloud (step-4 average) | 46.77 | 60.08 | 68.66 | 73.82 | 78.70 | 83.91 | 68.66 |
| trees_ground | 44.33 | 58.82 | 69.16 | 74.58 | 80.97 | 87.12 | 69.16 |
| trees_ground, 10x cap | 44.96 | 59.14 | 68.70 | 75.15 | 79.93 | 84.83 | 68.78 |
| trees_ground, 100x cap | 44.33 | 58.82 | 69.16 | 74.58 | 80.97 | 87.12 | 69.16 |

## Does a ground-only tree model of on-device size match the cloud tier?
**Yes, on validation.** The 10x-capped ground-only trees (about 85 KB estimated, 7.6 times the
edge network's int8 size) have an averaged RMSE of 68.78 W/m2, against 68.66 for the cloud tier.

The paired day-block bootstrap 95% interval of RMSE_trees - RMSE_cloud includes zero for all
five cloud seeds:

| cloud seed | point (W/m2) | 95% interval |
|---|---|---|
| 0 | +0.06 | -2.38 to +2.54 |
| 1 | +0.06 | -2.37 to +2.57 |
| 2 | +0.67 | -1.75 to +3.08 |
| 3 | +0.29 | -2.27 to +2.85 |
| 4 | -0.45 | -2.87 to +2.02 |

Uncapped trees_ground also matches the cloud tier on all five seeds, with points from -0.07 to
+1.05 W/m2.

Caveats:
- "On-device size" here is a size estimate. Nothing was run on a microcontroller.
- The tree model is about 7.6 times larger than the int8 edge network. Whether 85 KB fits a
  target device is a hardware question this study does not answer.

## Edge vs cloud, paired day-block bootstrap (RMSE_edge - RMSE_cloud, W/m2)

| seed | point | 95% interval | includes zero? |
|---|---|---|---|
| 0 | 4.34 | 1.23 to 7.69 | no |
| 1 | 3.49 | 0.64 to 6.34 | no |
| 2 | 3.98 | 0.97 to 7.04 | no |
| 3 | 8.78 | 5.59 to 12.00 | no |
| 4 | 2.55 | -0.01 to 5.08 | **yes** |

For one of five seeds (seed 4), the edge-cloud interval includes zero.

Correction (2026-10-08): an earlier version of this file built bootstrap day blocks from all
development days, including days with no validation cells. Day blocks are now built only from
days that have selected cells (`src/escal/bootstrap.py`, `tests/test_bootstrap.py`). With that
correction, the seed-4 edge-cloud interval moved from 0.04 to 5.14 to -0.01 to 5.08, so it now
includes zero. The other intervals moved slightly, and no conclusion changed.

## Measured footprint (Stage 7, 2026-10-08)
Run with `./run.sh measure` (`scripts/measure_footprint.py`). The numbers are in
`results/footprint/measured.json`; the rules are in `docs/PLAN.md` section 9.

**All latency figures are on a PC CPU: single row, one thread, ONNX Runtime 1.30 or compiled C.
Nothing was run on a microcontroller.**

- **int8:** ONNX Runtime static quantisation (QDQ, int8 weights, uint8 activations, MinMax),
  calibrated on models_train rows only.
- **Latency:** 200 warm-up runs, then 5,000 timed runs; median and 95th percentile. Measured on
  seed 0. Sizes are identical across seeds.

### Sizes and latency

| model | estimate | measured file | latency median / p95 (µs) |
|---|---|---|---|
| edge network, fp32 ONNX | 43,032 B (parameters x 4) | 43,846 B | 7.5 / 7.7 |
| edge network, int8 ONNX | 11,208 B | **15,177 B** | 9.6 / 10.0 |
| uncertainty gate, fp32 / int8 ONNX | n/a | 13,832 / **6,067 B** | 6.7 / 6.9 (fp32); 8.6 / 9.5 (int8) |
| learned gate, fp32 / int8 ONNX | n/a | 13,832 / **6,067 B** | 6.8 / 7.0 (fp32); 8.5 / 8.7 (int8) |
| trees_ground 10x cap, estimate | 85,140 B (7,095 nodes x 12 B) | n/a | n/a |
| trees_ground 10x cap, ONNX (one file, all horizons) | n/a | 218,912 B | 9.5 / 10.1 |
| trees_ground 10x cap, generated C (`zig cc -O2`) | n/a | **object 87,183 B**; node arrays 71,920 B; source 222,240 B | **3.5 / 7.2** |

- The ONNX files carry graph metadata. ONNX Runtime's TreeEnsemble format stores every node with
  several float and int attributes, so the tree ONNX file is about 2.6 times the node-array
  estimate. The compiled C object (87 KB) is close to the estimate (85 KB). The C node arrays
  use 10 bytes per node, because the indices are int16, against the 12 assumed in the estimate.
- int8 runs slower than fp32 on this PC for these tiny networks: the quantise and dequantise
  steps cost more than they save. This says nothing about microcontroller latency.
- **Export note:** skl2onnx 1.20 failed on sklearn 1.9 HistGradientBoosting (a boolean passed
  as an integer attribute). The trees were therefore exported with a small custom
  TreeEnsembleRegressor builder. skl2onnx was removed from the environment.

### Accuracy before and after
**Edge network**, validation primary rows, RMSE in W/m2 (seed mean):

| | 30min | 60min | 90min | 120min | 150min | 180min | mean |
|---|---|---|---|---|---|---|---|
| fp32 | 51.88 | 64.05 | 72.89 | 79.26 | 83.35 | 88.26 | 73.28 |
| int8 | 53.05 | 64.92 | 73.88 | 80.36 | 84.07 | 89.04 | **74.22** |

- int8 costs +0.94 W/m2 on average (seed range +0.67 to +1.43). That is about a fifth of the
  4.6 W/m2 edge-cloud gap, so a quantised edge widens the gap somewhat.
- The ONNX fp32 export matches PyTorch to 5e-7 in kt.

**Gates**, validation, with thresholds recomputed on gate_fit from each version's own scores.
Values are seed means.

| gate | fp32-int8 score correlation | 10%: rate / share kept | 25%: rate / share kept | 50%: rate / share kept |
|---|---|---|---|---|
| uncertainty fp32 | 0.9967 to 0.9987 | 0.101 / 0.42 | 0.241 / 0.70 | 0.433 / 0.87 |
| uncertainty int8 | | 0.100 / 0.44 | 0.244 / 0.71 | 0.440 / 0.86 |
| learned fp32 | 0.9995 to 0.9997 | 0.074 / 0.36 | 0.209 / 0.61 | 0.378 / 0.83 |
| learned int8 | | 0.074 / 0.36 | 0.210 / 0.60 | 0.388 / 0.82 |

int8 leaves the gates' behaviour essentially unchanged.

**Trees (10x cap):**
- ONNX: maximum difference from sklearn 4.8e-7 in kt; RMSE 68.7825 against 68.7825 (sklearn).
- Generated C: maximum difference 9.5e-7; RMSE 68.7826 against 68.7825.
