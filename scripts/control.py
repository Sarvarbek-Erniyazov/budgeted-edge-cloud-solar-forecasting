"""Stage 6a: model-type control. Same tree model on four input sets, against the edge network.
Validation only, Stage 5 rows and seeds. Writes results/tiers/control.csv and control_effects.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tiers import OUT, Run, gbt_on, tile_summary  # noqa: E402

from escal.data import load_config  # noqa: E402
from escal.evaluate import common_rows, score  # noqa: E402
from escal.features import satellite, standardise  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    cc = cfg["control"]
    run = Run(cfg)
    seeds = cfg["seeds"]

    v0 = run.tc["base_variant"]
    tiles, meta, _ = satellite(run.base, run.lag, v0["sat_frames"], v0["sat_lookback_minutes"], v0["sat_diffs"])
    k = v0["sat_frames"]
    meta_df = pd.DataFrame(meta, columns=[f"sat_age{i}" for i in range(k)] + [f"sat_present{i}" for i in range(k)],
                           index=run.df.index)
    nam, _ = run.nwp(cc["nam_extra"])
    summ = tile_summary(tiles)
    sets = {"trees_ground": ([run.ground], False), "trees_ground_sat": ([run.ground, meta_df], True),
            "trees_ground_nam": ([run.ground, nam], False), "trees_all": ([run.ground, nam, meta_df], True)}

    models = {}
    for name, (parts, with_tiles) in sets.items():
        X = standardise(pd.concat(parts, axis=1), run.tr)
        if with_tiles:
            X = np.concatenate([X, summ], axis=1)
        models[name] = []
        for s in seeds:
            models[name].append((s, run.to_w(gbt_on(run, X, s))))
            print(f"{name} seed {s} done ({X.shape[1]} features)", flush=True)
    rp = cc["reuse_predictions"]
    edge = [(s, run.to_w(np.load(rp["edge"].format(seed=s)))) for s in seeds]
    net3 = [(s, run.to_w(np.load(rp["step3_network"].format(seed=s)))) for s in seeds]
    models = {"edge_network": edge, **models,
              "step4_average": [(s, (a + b) / 2) for (s, a), (_, b) in zip(models["trees_all"], net3)]}

    rows = {"primary": run.primary_rows(), "all_daylight": run.va}
    recs, effects = [], {}
    for rs in cc["row_sets"]:
        flat = {f"{n}|{s}": p for n, lst in models.items() for s, p in lst}
        flat["smart_persistence|na"] = run.tg["sp"]
        sel = common_rows(flat, rows[rs], run.tg["day"])
        for r in score(flat, sel, run.tg["ghi"], run.tg["sp"], run.hz):
            name, seed = r.pop("model").split("|")
            recs.append({"row_set": rs, "model": name, "seed": seed, **r})

        def avg_rmse(p):
            return float(np.mean([np.sqrt(np.mean((run.tg["ghi"][sel[:, j], j] - p[sel[:, j], j]) ** 2))
                                  for j in range(len(run.hz))]))
        r = {n: np.array([avg_rmse(p) for _, p in lst]) for n, lst in models.items()}

        def stats(x):
            return {"per_seed": np.round(x, 4).tolist(), "mean": float(x.mean()), "median": float(np.median(x)),
                    "min": float(x.min()), "max": float(x.max())}
        effects[rs] = {
            "definition": "effect = 1 - RMSE_A / RMSE_B (RMSE per horizon, averaged over horizons); "
                          "positive means A is better; W/m2 = RMSE_B - RMSE_A",
            "input_effect (A = trees_all, B = trees_ground)": {
                "relative": stats(1 - r["trees_all"] / r["trees_ground"]), "W_m2": stats(r["trees_ground"] - r["trees_all"])},
            "model_effect (A = trees_ground, B = edge_network)": {
                "relative": stats(1 - r["trees_ground"] / r["edge_network"]),
                "W_m2": stats(r["edge_network"] - r["trees_ground"])},
            "avg_rmse_per_seed": {n: np.round(x, 3).tolist() for n, x in r.items()},
            "trees_identical_across_seeds": bool(all(np.ptp(r[n]) == 0 for n in sets)),
        }
    pd.DataFrame(recs).to_csv(OUT / "control.csv", index=False, float_format="%.4f")
    (OUT / "control_effects.json").write_text(json.dumps(effects, indent=2))
    print(json.dumps({rs: {k: v for k, v in e.items() if k != "definition"} for rs, e in effects.items()}, indent=1))


if __name__ == "__main__":
    main()
