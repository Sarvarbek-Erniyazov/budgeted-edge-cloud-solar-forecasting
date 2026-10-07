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
