"""Footprint of ground-only trees against the edge network, and size-capped ground-only trees.
Validation only. Writes results/tiers/footprint_trees.json (see docs/PLAN.md section 8)."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from tiers import CKPT, OUT, Run, gbt_on, tile_summary  # noqa: E402

from escal.bootstrap import paired_diff  # noqa: E402
from escal.data import load_config  # noqa: E402
from escal.evaluate import common_rows, score  # noqa: E402
from escal.features import standardise  # noqa: E402
from escal.models import EdgeNet, n_params  # noqa: E402

CACHE = CKPT / "step4_trees_all_pred_kt.npy"


def tree_stats(models, X_row, repeats: int, bytes_per_node: int) -> dict:
    preds = [p for m in models for p in m._predictors]          # one list of trees per iteration
    nodes = [tree.nodes for it in preds for tree in it]
    n_nodes = int(sum(len(n) for n in nodes))
    for m in models:
        m.predict(X_row)                                        # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for m in models:
            m.predict(X_row)
        times.append(time.perf_counter() - t0)
    return {"horizon_models": len(models), "trees": int(sum(m.n_iter_ for m in models)),
            "trees_per_horizon": [int(m.n_iter_) for m in models],
            "max_depth": int(max(n["depth"].max() for n in nodes)),
            "leaves": int(sum(n["is_leaf"].sum() for n in nodes)), "nodes": n_nodes,
            "compact_size_bytes_estimate": n_nodes * bytes_per_node,
            "pickle_size_bytes": len(pickle.dumps(models)),
            "single_row_cpu_ms_median": float(np.median(times) * 1e3)}


def cloud_kt(run: Run) -> list[np.ndarray]:
    """The step-4 average per seed: trees on step-3 inputs (deterministic, cached) + step-3 network."""
    if CACHE.exists():
        trees = np.load(CACHE)
    else:
        v3 = {**run.tc["base_variant"], **run.cfg["ladder"]["step3"]}
        d = run.cloud_data(v3, run.lag)
        trees = gbt_on(run, np.concatenate([d["x"], tile_summary(d["tiles"])], axis=1), run.cfg["seeds"][0])
        np.save(CACHE, trees)
    return [(trees + np.load(CKPT / "step3" / f"cloud_seed{s}_pred_kt.npy")) / 2 for s in run.cfg["seeds"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    fc = cfg["footprint"]
    run = Run(cfg)
    seeds = cfg["seeds"]
    Xg = standardise(run.ground, run.tr)
    row = Xg[np.nonzero(run.va)[0][:1]]

    # edge network
    net = EdgeNet(Xg.shape[1], len(run.hz), run.tc["edge"]["hidden"])
    net.load_state_dict(torch.load(CKPT / "base" / "edge_seed0.pt", map_location="cpu"))
    w = [p for n, p in net.named_parameters() if n.endswith("weight")]
    b = [p for n, p in net.named_parameters() if n.endswith("bias")]
    int8 = (sum(p.numel() for p in w) + fc["int8_bias_bytes"] * sum(p.numel() for p in b)
            + fc["int8_bytes_per_tensor_qparams"] * (len(w) + len(b)))
    net.eval()
    xr = torch.as_tensor(row)
    with torch.no_grad():
        net(xr)
        times = []
        for _ in range(fc["timing_repeats"]):
            t0 = time.perf_counter()
            net(xr)
            times.append(time.perf_counter() - t0)
    edge_fp = {"parameters": n_params(net), "float32_bytes": n_params(net) * 4, "int8_bytes_estimate": int(int8),
               "int8_estimate_rule": "int8 weights + int32 biases + 8 bytes per tensor for scale/zero point",
               "single_row_cpu_ms_median": float(np.median(times) * 1e3)}

    # trees_ground and the capped ground-only trees
    tg_kt, tg_models = gbt_on(run, Xg, seeds[0], return_models=True)
    fp = {"trees_ground": tree_stats(tg_models, row, fc["timing_repeats"], fc["bytes_per_node"])}
    preds = {"trees_ground": tg_kt}
    for mult, leaves in zip(fc["cap_multiples"], fc["cap_max_leaf_nodes"]):
        cap = mult * int8
        per_h = cap / fc["bytes_per_node"] / len(run.hz)
        max_iter = int(per_h // (2 * leaves - 1))
        kt, models = gbt_on(run, Xg, seeds[0], max_iter=max_iter, max_leaf_nodes=leaves, return_models=True)
        name = f"trees_ground_cap{mult}x"
        st = tree_stats(models, row, fc["timing_repeats"], fc["bytes_per_node"])
        fp[name] = {"cap_bytes": int(cap), "max_leaf_nodes": leaves, "max_trees_per_horizon": max_iter,
                    "within_cap": st["compact_size_bytes_estimate"] <= cap, **st}
        preds[name] = kt
        print(name, json.dumps(fp[name]), flush=True)

    # validation RMSE per horizon, primary rows; all on identical cells
    clouds = cloud_kt(run)
    edges = [np.load(CKPT / "base" / f"edge_seed{s}_pred_kt.npy") for s in seeds]
    models = {**{f"edge_network|{s}": run.to_w(p) for s, p in zip(seeds, edges)},
              **{f"cloud_step4_average|{s}": run.to_w(p) for s, p in zip(seeds, clouds)},
              **{f"{k}|na": run.to_w(v) for k, v in preds.items()}}
    sel = common_rows({**models, "sp": run.tg["sp"]}, run.primary_rows(), run.tg["day"])
    sc = score(models, sel, run.tg["ghi"], run.tg["sp"], run.hz)
    rmse = {}
    for r in sc:
        name = r["model"].split("|")[0]
        rmse.setdefault(name, {}).setdefault(r["horizon"], []).append(r["RMSE"])
    rmse = {n: {h: float(np.mean(v)) for h, v in d.items()} for n, d in rmse.items()}

    ts, y = run.df["timestamp"], run.tg["ghi"]
    match = {}
    for name in preds:
        per_seed = []
        for s, c in zip(seeds, clouds):
            r = paired_diff(y, run.to_w(preds[name]), run.to_w(c), sel, ts, cfg)
            per_seed.append({"cloud_seed": s, **r, "matches_cloud": r["lo"] <= 0})
        match[name] = {"rmse_trees_minus_cloud": per_seed,
                       "matches_cloud_seeds": int(sum(p["matches_cloud"] for p in per_seed))}
    edge_cloud = [{"seed": s, **paired_diff(y, run.to_w(e), run.to_w(c), sel, ts, cfg)}
                  for s, e, c in zip(seeds, edges, clouds)]

    out = {"rows": "validation primary rows (Stage 5)", "cells": int(sel.sum()),
           "edge_network": edge_fp, "trees": fp,
           "validation_rmse_seed_mean": rmse,
           "match_rule": "matches if the 95% paired day-block bootstrap interval of RMSE_trees - RMSE_cloud has lo <= 0",
           "match_vs_cloud": match,
           "paired_bootstrap_edge_minus_cloud_rmse": edge_cloud,
           "timing_note": "single-row CPU time on a PC through Python/sklearn or PyTorch; not microcontroller latency"}
    (OUT / "footprint_trees.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("edge_network", "validation_rmse_seed_mean")}, indent=1))
    print(json.dumps({n: m["matches_cloud_seeds"] for n, m in match.items()}))
    print(json.dumps(edge_cloud))


if __name__ == "__main__":
    main()
