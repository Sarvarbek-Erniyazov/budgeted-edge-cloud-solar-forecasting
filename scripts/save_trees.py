"""Freeze preparation (FREEZE.md section e, item 2): fit and save the tree models used in the test run,
with the standardisation statistics, and verify they reproduce the committed development predictions
exactly. Development data only. Writes checkpoints/trees/ and results/freeze/trees_verification.json."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tiers import CKPT, Run, gbt_on, tile_summary  # noqa: E402

from escal.data import load_config  # noqa: E402
from escal.evaluate import common_rows  # noqa: E402
from escal.features import apply_standardiser, fit_standardiser  # noqa: E402

TREES = Path("checkpoints/trees")
PRED = Path("results/tiers/predictions")
OUT = Path("results/freeze")


def avg_rmse(y, p, sel):
    return float(np.mean([np.sqrt(np.mean((y[sel[:, j], j] - p[sel[:, j], j]) ** 2)) for j in range(y.shape[1])]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    TREES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    run = Run(cfg)
    seeds, hz = cfg["seeds"], run.hz
    s0 = seeds[0]

    g_mu, g_sd = fit_standardiser(run.ground, run.tr)
    Xg = apply_standardiser(run.ground, g_mu, g_sd)
    v3 = {**run.tc["base_variant"], **cfg["ladder"]["step3"]}
    d3 = run.cloud_data(v3, run.lag)
    t_mu, t_sd = fit_standardiser(d3["tab"], run.tr)
    assert np.array_equal(apply_standardiser(d3["tab"], t_mu, t_sd), d3["x"])
    X3 = np.concatenate([d3["x"], tile_summary(d3["tiles"])], axis=1)
    np.savez(TREES / "norms.npz", ground_mu=g_mu.values, ground_sd=g_sd.values, ground_cols=np.array(g_mu.index),
             step3_mu=t_mu.values, step3_sd=t_sd.values, step3_cols=np.array(t_mu.index))

    fp = json.loads(Path("results/tiers/footprint_trees.json").read_text())
    cap = fp["trees"]["trees_ground_cap10x"]
    fits = {"trees_all": (X3, {}), "trees_ground": (Xg, {}),
            "trees_ground_cap10x": (Xg, {"max_iter": cap["max_trees_per_horizon"], "max_leaf_nodes": cap["max_leaf_nodes"]})}
    preds, ver = {}, {}
    for name, (X, kw) in fits.items():
        kt, models = gbt_on(run, X, s0, return_models=True, **kw)
        path = TREES / f"{name}.pkl"
        path.write_bytes(pickle.dumps(models))
        back = pickle.loads(path.read_bytes())
        kt_back = np.clip(np.stack([m.predict(X) for m in back], 1).astype(np.float32), *run.tc["kt_clip"])
        ver[name] = {"file": str(path).replace("\\", "/"), "bytes": path.stat().st_size,
                     "pickle_roundtrip_identical": bool(np.array_equal(kt_back, kt))}
        preds[name] = kt
        print(name, ver[name], flush=True)

    # committed development predictions
    keep = run.df["part"].isin(["gate_fit", "val"]).values
    tg_par = pd.read_parquet(PRED / "trees_ground.parquet")
    ver["trees_ground"]["matches_results/tiers/predictions/trees_ground.parquet"] = bool(
        np.array_equal(tg_par[[f"kt_{h}" for h in hz]].values, preds["trees_ground"][keep]))
    cache = CKPT / "step4_trees_all_pred_kt.npy"
    ver["trees_all"]["matches_cached_step4_trees_predictions_all_rows"] = bool(np.array_equal(np.load(cache), preds["trees_all"]))
    cl = pd.read_parquet(PRED / "cloud.parquet")
    same = []
    for s in seeds:
        net = np.load(CKPT / "step3" / f"cloud_seed{s}_pred_kt.npy")
        ref = cl[cl["seed"].astype(str) == str(s)][[f"kt_{h}" for h in hz]].values
        same.append(bool(np.array_equal(ref, ((preds["trees_all"] + net) / 2)[keep])))
    ver["trees_all"]["cloud_tier_matches_results/tiers/predictions/cloud.parquet_per_seed"] = same
    prim = common_rows({"sp": run.tg["sp"]}, run.primary_rows(), run.tg["day"])
    r = avg_rmse(run.tg["ghi"], run.to_w(preds["trees_ground_cap10x"]), prim)
    ref_fp = fp["validation_rmse_seed_mean"]["trees_ground_cap10x"]["mean"]
    ref_ms = json.loads(Path("results/footprint/measured.json").read_text())["trees_ground_cap10x"]["onnx"]["rmse_sklearn"]
    ver["trees_ground_cap10x"].update({"validation_rmse": r, "footprint_trees.json_rmse": ref_fp,
                                       "measured.json_rmse_sklearn": ref_ms,
                                       "identical_to_both": bool(r == ref_fp == ref_ms)})
    ver["norms"] = {"file": "checkpoints/trees/norms.npz", "ground_columns": int(len(g_mu)), "step3_columns": int(len(t_mu))}
    ok = all(v.get("pickle_roundtrip_identical", True) for v in ver.values()) and \
        ver["trees_ground"]["matches_results/tiers/predictions/trees_ground.parquet"] and \
        ver["trees_all"]["matches_cached_step4_trees_predictions_all_rows"] and all(same) and \
        ver["trees_ground_cap10x"]["identical_to_both"]
    ver["all_identical"] = bool(ok)
    (OUT / "trees_verification.json").write_text(json.dumps(ver, indent=2))
    print(json.dumps(ver, indent=1))
    if not ok:
        raise SystemExit("saved trees do NOT reproduce the committed development predictions")


if __name__ == "__main__":
    main()
