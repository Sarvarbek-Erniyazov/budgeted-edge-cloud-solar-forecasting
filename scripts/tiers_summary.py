"""Per-seed robustness of the go/no-go gain; reads results/go_no_go.json, writes results/tiers/summary.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from escal.data import load_config


def main() -> None:
    cfg = load_config()
    thr = cfg["go_no_go"]["min_rmse_gain"]
    g = json.loads(Path("results/go_no_go.json").read_text())
    out = {"threshold": thr, "note": "gain per seed pairs edge seed s with cloud seed s; the verdict uses the seed mean",
           "variants": {}}
    for v in g["variants"] + [{**g["sensitivity_benchmark_convention"], "name": "sens_lag0"}]:
        s = np.array(v["gain_avg_per_seed"])
        out["variants"][v["name"]] = {
            "gain_avg_seed_mean_rmse": v["gain_avg"], "gain_per_seed": s.tolist(),
            "median_seed_gain": float(np.median(s)), "min_seed_gain": float(s.min()),
            "seeds_passing": int((s >= thr).sum()), "n_seeds": int(len(s)),
            "edge_rmse_per_horizon": v["edge_rmse_per_horizon"], "cloud_rmse_per_horizon": v["cloud_rmse_per_horizon"]}
    Path("results/tiers/summary.json").write_text(json.dumps(out, indent=2))
    for k, v in out["variants"].items():
        print(f"{k:22s} mean {v['gain_avg_seed_mean_rmse']:+.4f} median {v['median_seed_gain']:+.4f} "
              f"min {v['min_seed_gain']:+.4f} pass {v['seeds_passing']}/{v['n_seeds']}")


if __name__ == "__main__":
    main()
