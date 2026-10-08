# Budgeted Edge-to-Cloud Escalation for Solar Irradiance Forecasting

<!-- Badges: add once they are true. Suggested: Python version, PyTorch, License: MIT, tests (GitHub Actions), status: work in progress -->

> **Status:** the protocol was frozen on 2026-10-08 (tag `freeze-2026-10-08`, [docs/FREEZE.md](docs/FREEZE.md)), and the 2016 test year was evaluated once from that tag. **Of the five pre-specified claims, two were refuted and one was partly refuted on the test year (claim 5: the cloud tier is reliably better on ramp cells for 4 of 5 seeds).** The paper is in preparation.

## Summary

A low-power device at a photovoltaic site forecasts solar irradiance from its own ground sensors. A small on-device gate decides when to forward the request to a cloud model that also uses satellite imagery and numerical weather prediction. This repository measures how much of the cloud model's accuracy gain survives as the share of forwarded requests falls from 100% to 0%, on the public Folsom benchmark. It is a measurement study on one site, not a new architecture.

## Figures

- Two-tier architecture: [figures/core/architecture.svg](figures/core/architecture.svg)
- RMSE against escalation budget for all gates, 2016: [figures/core/budget_sweep.svg](figures/core/budget_sweep.svg)
- Routing map, hour by month (uncertainty gate, 25% budget): [figures/core/routing_map.svg](figures/core/routing_map.svg)

## Reproduce

```bash
py -3.11 -m venv .venv && source .venv/Scripts/activate   # Windows, Git Bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
./run.sh test        # unit tests
./run.sh download    # Folsom non-image files, about 0.52 GB, MD5-checked
./run.sh audit       # data audit
bash scripts/install_hooks.sh   # optional: pre-commit guard against leaked local paths
```

Every stage has its own `./run.sh` command (`./run.sh` lists them). The test-year sequence is in
[docs/FREEZE.md](docs/FREEZE.md), section c. The exact environment is in `requirements.lock.txt`.

## Results (2016 test year, single run from the frozen tag)

The table uses 39,655 daylight forecast cells; RMSE and MAE are in W/m², averaged over the six horizons
(30 to 180 min) and over five seeds.

| Setting | RMSE | MAE | Skill vs smart persistence | Escalation rate (target → realised) |
|---|---|---|---|---|
| Edge only (on-device MLP, fp32) | 76.71 | 45.36 | 0.130 | 0% |
| Edge only, int8 (secondary row) | 77.21 | 46.47 | 0.124 | 0% |
| Uncertainty gate, 10% budget | 73.90 | 44.06 | 0.162 | 10% → 9.9% |
| Uncertainty gate, 25% budget | 72.61 | 43.29 | 0.177 | 25% → 23.0% |
| Uncertainty gate, 50% budget | 71.26 | 41.44 | 0.193 | 50% → 47.7% |
| Cloud only (trees + network, satellite and NWP) | 70.66 | 40.40 | 0.200 | 100% |
| Smart persistence (reference) | 89.43 | 49.90 | 0 | n/a |

At a 25% budget, the uncertainty gate keeps 69% of the cloud tier's RMSE gain, against 24% for random escalation.
The other gates, the linear anchors, model sizes and CPU latency are in
[results/paper/core_table.md](results/paper/core_table.md). Every number traces to a file; see
[docs/number_trace.md](docs/number_trace.md).

**Pre-specified claims** ([docs/FREEZE.md](docs/FREEZE.md), [results/test/claims_test.json](results/test/claims_test.json)):

| # | Claim (from validation) | 2016 result |
|---|---|---|
| 1 | Satellite and NWP add nothing when the model type is held fixed | **Refuted:** they reduce error by 4.2% |
| 2 | A ground-only tree model of about 85 KB matches the cloud tier | **Refuted:** it is worse by 3.7%, with the interval above zero for 4 of 5 seeds |
| 3 | The cloud tier beats the on-device network | Supported: 7.9% lower RMSE, with the interval excluding zero for all seeds |
| 4 | On-device gates beat random; the learned gate is no better than the uncertainty gate; realised rates match targets | Supported |
| 5 | Ramps: no reliable cloud advantage, and no gate helps | Mixed: the cloud tier is reliably better on ramps (refutes the first part); no gate reaches the 0.5 share (the second part holds) |

Note (post-freeze): 'about 85 KB' is the 12-bytes-per-node estimate (85,140 B = 85.1 kB decimal); the compiled object the paper reports is 87,183 B = 87.2 kB decimal; the paper reports decimal kB.

An exploratory follow-up (split chosen after seeing the results) finds that the disagreement with validation
is not a seasonal artefact: see [results/analyses/exploratory_halfyear_claims.md](results/analyses/exploratory_halfyear_claims.md).

## Method in brief

- **Data:** Folsom dataset (Pedro, Larson and Coimbra, 2019), Zenodo record 2826939, CC-BY 4.0.
- **Split:** chronological; 2016 is the test year and is locked in code until the protocol is frozen.
- **Gates:** random, fixed-interval, variability threshold, uncertainty, learned, oracle.
- **Plan and checkpoints:** `docs/PLAN.md`.

## Limitations

- One site, so no claim about fleets, virtual power plants or transfer to other sites.
- On-device cost is reported as model footprint and latency measured on a PC CPU. Nothing was run on a microcontroller.
- Two development conclusions did not hold on the test year (claims 1 and 2). One validation half-year was too short to estimate the value of satellite and NWP inputs.
- The cloud tier is trained here; it is not a state-of-the-art forecaster.

## Layout

`configs/` experiment YAMLs · `src/escal/` library · `scripts/` stage entry points · `tests/` · `results/` raw metrics · `figures/` · `docs/`

## Citation

Manuscript in preparation. See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The Folsom dataset is CC-BY 4.0 and is not redistributed here.

## Authors

Sarvarbek Erniyazov and Chang Gyoon Lim, Department of Computer Engineering, Chonnam National University, Yeosu, South Korea.

## Acknowledgement

This work was supported by Industry-academia-research collabo R&D Program (RS-2026-25529976) funded by the Ministry of SMEs and Startups (MSS, Korea) and Jeollanam-do ('2026 R&D supporting program' operated by Jeonnam Technopark)
