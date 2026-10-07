# Budgeted Edge-to-Cloud Escalation for Solar Irradiance Forecasting

<!-- Badges: add once they are true. Suggested: Python version, PyTorch, License: MIT, tests (GitHub Actions), status: work in progress -->

> **Status: work in progress.** No experiment has been run yet. The results table below is empty until the protocol is frozen and the test year is evaluated.

## Summary

A low-power device at a photovoltaic site forecasts solar irradiance from its own ground sensors. A small on-device gate decides when to forward the request to a cloud model that also uses satellite imagery and numerical weather prediction. This repository measures how much of the cloud model's accuracy gain survives as the share of forwarded requests falls from 100% to 0%, on the public Folsom benchmark. It is a measurement study on one site, not a new architecture.

## Figures

<!-- Fig. 1: two-tier architecture (figures/architecture.svg) -->
<!-- Fig. 2: accuracy versus escalation budget for all gates (figures/budget_sweep.svg) -->
<!-- Fig. 3: routing map, hour by month (figures/routing_map.svg) -->

## Reproduce

```bash
py -3.11 -m venv .venv && source .venv/Scripts/activate   # Windows, Git Bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
./run.sh test        # unit tests
./run.sh download    # Folsom non-image files, about 0.52 GB, MD5-checked
./run.sh audit       # data audit
```

Further stages are added to `run.sh` as they are built. The exact environment is in `requirements.lock.txt`.

## Results

| Setting | RMSE | MAE | Skill vs smart persistence | Escalation rate |
|---|---|---|---|---|
| Edge only | | | | 0% |
| Gate at 10% / 25% / 50% budget | | | | |
| Cloud only | | | | 100% |

Filled from `results/` after the test-year run. Every number traces to a file there.

## Method in brief

- **Data:** Folsom dataset (Pedro, Larson and Coimbra, 2019), Zenodo record 2826939, CC-BY 4.0.
- **Split:** chronological; 2016 is the test year and is locked in code until the protocol is frozen.
- **Gates:** random, fixed-interval, variability threshold, uncertainty, learned, oracle.
- **Plan and checkpoints:** `docs/PLAN.md`.

## Limitations

- One site, so no claim about fleets, virtual power plants or transfer to other sites.
- On-device cost is reported as model footprint and latency measured on a PC CPU. Nothing was run on a microcontroller.
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
