# EXPLORATORY: claims 1 to 3 by half-year of 2016

**Exploratory, not a planned analysis** (`docs/FREEZE.md` section c). The January-June /
July-December split was **chosen after seeing the 2016 results**, to examine why validation
(July to December 2015) and test disagree on claims 1 and 2. The frozen verdict rules are applied
to each half for description only. The full-year verdicts in `results/test/claims_test.json`
stand. Numbers are in `results/analyses/exploratory_halfyear_claims.json`
(`scripts/exploratory_halfyear_claims.py`), which reads `results/test/` only.

| | Jan-Jun 2016 (H1) | Jul-Dec 2016 (H2) | validation, Jul-Dec 2015 |
|---|---|---|---|
| primary cells | 20,901 | 18,754 | 20,125 |
| satellite available (share of daylight issue times) | 0.961 | 0.927 | 0.988 |
| **Claim 1** input effect (trees all vs ground) | +2.79%; interval of RMSE_ground - RMSE_all 2.49 [-0.93, 6.34] | **+8.75%**; 4.56 [2.08, 6.87] | -0.27%; -0.19 [-2.60, 2.28] |
| **Claim 2** capped ground trees vs cloud, d | 2.04%; 0 of 5 seeds above zero: "not distinguishable" | **9.75%; 5 of 5 above zero: "worse"** | 0.18%; 0 of 5: "matches" |
| **Claim 3** edge vs cloud gain | 0.056; 0 of 5 seeds include zero | 0.152; 0 of 5 include zero | 0.063; 1 of 5 |
| cloud RMSE (W/m2, seed mean) | 86.50 | 47.07 | 68.66 |
| edge RMSE (W/m2, seed mean) | 91.61 | 55.53 | 73.28 |

## What this shows (descriptive only)
- **Not a seasonal-mix artefact.** The 2016 half that matches the validation season (H2) shows
  the largest input effect and the clearest failure of claim 2. A full year containing the wet
  season does not explain the disagreement.
- **The same months in different years behaved very differently.** In H2 2016, errors are about
  30% lower than in H2 2015 (cloud 47.1 against 68.7 W/m2). In that clearer half-year, the cloud
  inputs reduce error by a larger fraction.
- One plausible reading, untested here: the value of satellite and NWP context, relative to a
  ground-only model, depends on the year's weather. One validation half-year was too little to
  estimate it.
