# Claim criteria: frozen rules against the stored 2016 values

Read-only check for the Protocol section. Rule text is quoted verbatim from `docs/FREEZE.md` section b
(lines 163-226) unless marked otherwise; where FREEZE.md splits one sentence over list bullets, the bullets are
joined with a space and nothing else is changed; values and flags are read from `results/test/claims_test.json` (and, where
noted, `results/test/gates/sweep_test.csv`). Where FREEZE.md does not state a rule, the table says **not stated**;
nothing is inferred. Rates and shares are fractions.

General conditions quoted from FREEZE.md:
- "Every claim is evaluated on the primary rows unless stated. "Interval" means the paired day-block bootstrap 95%
  interval. The equivalence margins (the 2% in claim 1, the "3 of 5 seeds" rules) are **proposed here, before 2016 is
  seen, and need the author's approval**." (Approved on 2026-10-08 per the file header: "The author (Sarvarbek
  Erniyazov) approved sections a to d with the seven amendments in `docs/PLAN.md` section 10".)
- Bootstrap (section a, Evaluation): "paired day-block bootstrap. Blocks are local standard-time days (08 to 08 UTC)
  built only from days with selected cells; 2,000 resamples; seed 0; 95% percentile intervals."
- Metrics (section a): "share kept = (RMSE_edge - RMSE_gate) / (RMSE_edge - RMSE_cloud)"; "gain = 1 - RMSE_cloud / RMSE_edge."

| claim | claim text (verbatim) | sub-test / metric (verbatim) | decision rule (verbatim) | stored 2016 value | flag in `claims_test.json` |
|---|---|---|---|---|---|
| 1 | "with the model type fixed, satellite and NAM add nothing. Validation: -0.27%." | "input effect = 1 - RMSE_all / RMSE_ground (averaged over horizons), plus the interval of RMSE_ground - RMSE_all." | *Counts against:* "an input effect of at least +2% **and** an interval of RMSE_ground - RMSE_all lying entirely above zero." | input effect 0.0417 (`claim1_input_effect.input_effect`); RMSE_ground - RMSE_all 3.08 W/m2, interval [0.70, 5.58], 345 days (`interval_rmse_ground_minus_all`) | `claim1_input_effect.counts_against` = **True** |
| 1 | (same) | (inputs hurt) | "An effect of -2% or less (inputs hurt) does not count against the claim; it is reported." | effect is +0.0417, so this case does not arise | no flag stored |
| 2 | "a ground-only tree model of about 85 KB (estimated; 87 KB compiled object) matches the cloud tier. Validation: 68.78 against 68.66 W/m2; the interval includes zero for 5 of 5 seeds." | "d = seed-mean (RMSE_cap10x - RMSE_cloud), with RMSE averaged over horizons, as a share of the seed-mean cloud RMSE; the interval of RMSE_cap10x - RMSE_cloud, per cloud seed." | *Verdict rule* "(equivalence margin of 2%, amendment 1): **"matches"** only if \|d\| <= 2% **and** the interval is not entirely above zero for 3 or more of the 5 seeds; **"not distinguishable"** if the interval is not entirely above zero for 3 or more seeds, but \|d\| > 2%; **"worse"**, which counts against the claim, if the interval lies entirely above zero for 3 or more of the 5 seeds." | d = 0.0372 (`relative_difference_d`), above the 0.02 margin; interval entirely above zero for 4 of 5 cloud seeds (`seeds_entirely_above_zero` = 4; seed 4 interval [-0.11, 4.32] includes zero) | `claim2_on_device_sufficiency.verdict` = **"worse"**; `counts_against` = **True** |
| 3 | "the cloud tier beats the edge network. Validation: gain 0.063 (median seed 0.055); the interval excludes zero for 4 of 5 seeds." | "gain per seed and seed mean, plus the interval of RMSE_edge - RMSE_cloud per seed." | *Counts against:* "the interval includes zero or lies below zero for 3 or more of the 5 seeds." | seed-mean gain 0.0788 (`gain_mean_of_seed_rmse`), per seed 0.057 to 0.113; intervals including zero or below: 0 of 5 (`seeds_including_zero_or_below` = 0) | `claim3_edge_vs_cloud_fp32.counts_against` = **False** |
| 3 | (same) | 5% go/no-go threshold | "The gain is also reported against the 5% go/no-go threshold. Falling below 5% is reported, but does not by itself refute the claim." | seed-mean gain 0.0788 >= 0.05 | `below_5pct_threshold` = **False** (not a verdict flag) |
| 4 | "*Claim (validation):* uncertainty ≥ learned > variability > random ≈ fixed-interval in share kept; at 25%: uncertainty 0.70, learned 0.61, variability 0.40, random 0.25." | "share kept and realised rate per gate, budget and seed, with mean, median and range." | The ranking itself: **not stated** (no decision rule is given for the ordering or for "≈"; the 25% values are validation references). | 2016 share kept at 25% (seed mean, `claim4_gates.table`): uncertainty 0.6878, learned 0.6493, variability 0.3948, random 0.2406, fixed-interval 0.2297 | no flag stored |
| 4a | "an on-device gate beats random" | uncertainty vs random share kept at 25% | *Counts against:* "at the 25% target, the uncertainty gate's share kept exceeds random's by less than 0.10 in 3 or more of the 5 seeds." | uncertainty minus random share kept per seed: 0.383, 0.518, 0.436, 0.393, 0.507 (`uncertainty_minus_random_share_at_25pct_per_seed`); below 0.10 in 0 of 5 seeds | `claim4_gates.against_gate_beats_random` = **False** |
| 4b | "the learned gate is not better than uncertainty" | learned vs uncertainty seed-mean share kept at 10, 25, 50% | *Counts against:* "the learned gate's seed-mean share kept exceeds the uncertainty gate's by at least 0.10 at 2 or more of the 3 budgets." | learned minus uncertainty: -0.0464 (10%), -0.0385 (25%), -0.0489 (50%) (`learned_minus_uncertainty_seed_mean`); at least 0.10 at 0 of 3 budgets | `claim4_gates.against_learned_not_better` = **False** |
| 4c | "refitting on all of 2015 fixes the budget shortfall" | seed-mean realised rate vs target, score-based gates | *Counts against:* "for any score-based gate (variability, uncertainty, learned), the seed-mean realised rate differs from its target by more than 0.05 at any of 10, 25 or 50%." | realised minus target (`realised_minus_target`): variability +0.0055 / -0.0138 / -0.0014; uncertainty -0.0006 / **-0.0197** / -0.0230; learned +0.0053 / -0.0024 / -0.0061 (10 / 25 / 50%); largest \|difference\| 0.0230, below 0.05 | `claim4_gates.against_refit_fixes_shortfall` = **False** |
| 5 | "*Claim (validation):* the cloud tier is barely better than the edge on ramp cells (133.4 against 135.3 W/m2); no gate shows a reliable benefit on ramps." | "ramp-cell RMSE for edge, cloud and each gate; share kept on ramps; the interval of RMSE_edge - RMSE_cloud on ramp cells." | (rows 5a, 5b) | ramp cells: 2,551 (`ramp_cells`) | see 5a, 5b |
| 5a | "no reliable cloud advantage on ramps" | interval of RMSE_edge - RMSE_cloud on ramp cells, per seed | *Counts against:* "the ramp-cell interval of RMSE_edge - RMSE_cloud lies entirely above zero for 3 or more of the 5 seeds." | entirely above zero for 4 of 5 seeds (`intervals_rmse_edge_minus_cloud_ramp`, lo > 0; seed 3 interval [-0.34, 8.49] includes zero); 179 days | `claim5_ramp.against_no_cloud_advantage` = **True** |
| 5b | "no gate helps on ramps" | share kept on ramp cells at 25% | *Counts against:* "some gate keeps a share of at least 0.5 at the 25% target with a positive share for all 5 seeds." Whether the 0.5 applies to the seed mean or to each seed: **not stated**. Whether the oracle counts as "some gate": **not stated** in this rule (section a calls the oracle "a headroom figure only"). | seed-mean ramp share kept at 25% (`gate_share_at_25pct_ramp`): uncertainty **0.4752**, learned 0.4394, fixed-interval 0.2552, random 0.2454, variability 0.0767; `all_seeds_positive` = True for all five | `claim5_ramp.against_no_gate_helps` = **False** |

## The three points asked about

**Claim 2, equivalence margin.** "matches" means, verbatim: "\|d\| <= 2% **and** the interval is not entirely above
zero for 3 or more of the 5 seeds". The margin is 2% of the seed-mean cloud RMSE (d is "a share of the seed-mean
cloud RMSE"). 2016: d = 0.0372 is outside the margin, and the interval is entirely above zero for 4 of 5 seeds, so
the "worse" branch applies (it needs 3 or more), not "matches" or "not distinguishable". Stored verdict "worse",
counts_against True.

**Claim 4, budget-shortfall tolerance.** Verbatim: the claim counts against if "the seed-mean realised rate differs
from its target by more than 0.05 at any of 10, 25 or 50%". The tolerance is 0.05 on the rate (rates are fractions
of issue times). The uncertainty gate at the 25% target has a seed-mean realised rate of 0.2303
(`claim4_gates.table`), a difference of -0.0197 (`realised_minus_target.uncertainty@0.25`). |-0.0197| <= 0.05, so it
passes. The largest difference of any score-based gate at any report budget is 0.0230 (uncertainty at 50%), also
within 0.05. Flag False.

**Claim 5, "no gate helps" threshold and the 0.48.** Verbatim threshold: "some gate keeps a share of at least 0.5 at
the 25% target with a positive share for all 5 seeds". The 0.48 is the uncertainty gate's seed-mean ramp share kept
at 25%, 0.4752 (`gate_share_at_25pct_ramp.uncertainty.share_mean`), the highest of the five non-oracle gates. It is
below 0.5, so no gate meets the rule, and the flag is False. Two things FREEZE.md does not settle:
- It does not state whether "a share of at least 0.5" is the seed mean or applies per seed. The stored flag uses the
  seed mean (`scripts/test_run.py` line 332). Per seed (`results/test/gates/sweep_test.csv`, `row_set = ramp`,
  `escalate_to = cloud`, target 0.25, `share_retained`), the uncertainty gate has 0.515, 0.457, 0.640, 0.366, 0.398 and
  the learned gate 0.386, 0.425, 0.644, 0.369, 0.374: three single seeds exceed 0.5, the seed means do not.
- It does not state in this rule whether the oracle is excluded; the stored result notes "oracle excluded (headroom
  only)" (`claim5_ramp.note`). The oracle's seed-mean ramp share is 2.29.

## Implementation check (not part of FREEZE.md)

The flags in `claims_test.json` come from `scripts/test_run.py` lines 273 to 332. Each matches the quoted rule as
written: claim 1 `eff >= 0.02 and lo > 0`; claim 2 `"worse" if above >= 3`, else `"matches" if |d| <= 0.02`, else
`"not distinguishable"`; claim 3 `incl >= 3`; claim 4a `sum(x < 0.10) >= 3`; claim 4b `sum(v >= 0.10) >= 2`; claim 4c
`any(|v| > 0.05)`; claim 5a `sum(lo > 0) >= 3`; claim 5b `share_mean >= 0.5 and all_seeds_positive` over non-oracle
gates. The only choices beyond the text are the two marked "not stated" under claim 5b.
