"""Stage: benchmark persistence and linear anchors on development data (fit models_train, eval validation)."""
from __future__ import annotations

import argparse
from pathlib import Path

from escal.benchmark import load_intra_day, run_anchors
from escal.data import load_config

OUT = Path("results/anchors")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    OUT.mkdir(parents=True, exist_ok=True)
    df, endo, exo = load_intra_day(cfg)
    res = run_anchors(df, endo, exo, cfg)
    res.to_csv(OUT / "intra_day.csv", index=False, float_format="%.4f")
    summ = (res.groupby(["target", "model"], sort=False)[["RMSE", "MAE", "MBE", "skill"]]
            .agg(["mean", "std"]))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(OUT / "intra_day_mean_over_horizons.csv", float_format="%.4f")
    print(res.to_string())
    print(summ.to_string())


if __name__ == "__main__":
    main()
