"""Stage 6b step 1: save out-of-sample tier predictions on gate_fit and validation, once.
Both tiers were trained on 2014, so these predictions are out of sample. Gates read only these files.
Writes results/tiers/predictions/{truth,on_device_inputs,edge,cloud,trees_ground}.parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from footprint_trees import cloud_kt  # noqa: E402
from tiers import CKPT, OUT, Run, gbt_on  # noqa: E402

from escal.data import load_config  # noqa: E402
from escal.features import standardise  # noqa: E402

PRED = OUT / "predictions"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    run = Run(cfg)
    PRED.mkdir(parents=True, exist_ok=True)
    keep = run.df["part"].isin(["gate_fit", "val"]).values
    ts = run.df.loc[keep, "timestamp"].reset_index(drop=True)
    hz, tg, seeds = run.hz, run.tg, cfg["seeds"]

    truth = pd.DataFrame({"timestamp": ts, "part": run.df.loc[keep, "part"].values,
                          "sat_available": run.avail[keep],
                          "B(ghi_kt|30min)": run.df.loc[keep, "B(ghi_kt|30min)"].values})
    for j, h in enumerate(hz):
        truth[f"ghi_{h}"] = tg["ghi"][keep, j]
        truth[f"kt_{h}"] = run.df.loc[keep, f"ghi_kt_{h}"].values
        truth[f"clear_{h}"] = tg["clear"][keep, j]
        truth[f"sp_{h}"] = tg["sp"][keep, j]
        truth[f"day_{h}"] = tg["day"][keep, j]
    truth.to_parquet(PRED / "truth.parquet", index=False)

    dev = run.ground.loc[keep].reset_index(drop=True)
    dev.insert(0, "timestamp", ts)
    dev.to_parquet(PRED / "on_device_inputs.parquet", index=False)

    def long(arrs: dict) -> pd.DataFrame:
        frames = []
        for s, a in arrs.items():
            f = pd.DataFrame(a[keep], columns=[f"kt_{h}" for h in hz])
            f.insert(0, "seed", s)
            f.insert(0, "timestamp", ts)
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    long({s: np.load(CKPT / "base" / f"edge_seed{s}_pred_kt.npy") for s in seeds}).to_parquet(PRED / "edge.parquet", index=False)
    long(dict(zip(seeds, cloud_kt(run)))).to_parquet(PRED / "cloud.parquet", index=False)
    tgk = gbt_on(run, standardise(run.ground, run.tr), seeds[0])     # deterministic: one set
    long({"all": tgk}).to_parquet(PRED / "trees_ground.parquet", index=False)
    print("saved", {p.name: p.stat().st_size for p in PRED.iterdir()}, "rows", int(keep.sum()),
          truth["part"].value_counts().to_dict())


if __name__ == "__main__":
    main()
