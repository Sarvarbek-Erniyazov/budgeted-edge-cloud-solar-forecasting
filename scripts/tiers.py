"""Stage 5: edge and cloud tiers, evaluation on validation, go/no-go and the fallback ladder.

Trains on models_train only (early stopping on its last days), evaluates on validation only.
Writes results/tiers/ and results/go_no_go.json; checkpoints/tiers/ and logs/tiers/ are not committed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from escal.benchmark import anchor_predict
from escal.data import horizon_minutes, load_config
from escal.evaluate import common_rows, score
from escal.features import load_base, nwp, sat_available, satellite, standardise, targets
from escal.models import CloudNet, EdgeNet, n_params
from escal.splits import _bounds
from escal.train import fit, predict, set_seed

OUT = Path("results/tiers")
CKPT = Path("checkpoints/tiers")
LOGS = Path("logs/tiers")


class Run:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tc = cfg["tiers"]
        self.base = load_base(cfg)
        self.df = self.base.df
        self.hz = self.base.hz
        self.tg = targets(self.base, cfg)
        part = self.df["part"].values
        self.tr, self.es, self.va = part == "train", part == "es", part == "val"
        self.lag = cfg["satellite"]["availability_lag_minutes"]
        self.avail = sat_available(self.base, self.lag, self.tc["sat_max_age_minutes"])
        self.avail0 = sat_available(self.base, self.tc["sat_lag_sensitivity"], self.tc["sat_max_age_minutes"])
        self.ground = self.df[self.base.ground_cols]
        self._nwp = {}
        self.records, self.variants = [], []

    # ---------- inputs ----------
    def nwp(self, extra: bool):
        if extra not in self._nwp:
            self._nwp[extra] = nwp(self.base, self.cfg, extra)
        return self._nwp[extra]

    def edge_data(self):
        return {"x": standardise(self.ground, self.tr), "kt": self.tg["kt"], "mask": self.tg["mask"]}

    def cloud_data(self, v: dict, lag: int):
        tiles, meta, stamps = satellite(self.base, lag, v["sat_frames"], v["sat_lookback_minutes"], v["sat_diffs"])
        nam, info = self.nwp(v["nwp_extra"])
        k = v["sat_frames"]
        meta_df = pd.DataFrame(meta, columns=[f"sat_age{i}" for i in range(k)] + [f"sat_present{i}" for i in range(k)],
                               index=self.df.index)
        tab = pd.concat([self.ground, nam, meta_df], axis=1)
        return {"x": standardise(tab, self.tr), "tiles": tiles, "kt": self.tg["kt"], "mask": self.tg["mask"],
                "tab_cols": list(tab.columns), "n_nam": nam.shape[1], "stamps": stamps, "meta": meta,
                "nam_info": info, "nam_raw": nam}

    # ---------- training ----------
    def train_tier(self, kind: str, variant: str, data: dict) -> tuple[list[np.ndarray], int]:
        preds, logs, npar = [], [], 0
        for seed in self.cfg["seeds"]:
            set_seed(seed)                      # initial weights depend on the seed only
            if kind == "edge":
                m = EdgeNet(data["x"].shape[1], len(self.hz), self.tc["edge"]["hidden"])
            else:
                c = self.tc["cloud"]
                m = CloudNet(data["x"].shape[1], len(self.hz), data["tiles"].shape[1], c["hidden"],
                             c["cnn_channels"], c["fusion_hidden"])
            npar = n_params(m)
            d = {k: data[k] for k in ("x", "kt", "mask")}
            d["tiles"] = data.get("tiles") if kind == "cloud" else None
            log = fit(m, d, self.tr, self.es, self.cfg, seed, LOGS / variant / f"{kind}_seed{seed}.json")
            (CKPT / variant).mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), CKPT / variant / f"{kind}_seed{seed}.pt")
            p = np.clip(predict(m, d), *self.tc["kt_clip"])
            np.save(CKPT / variant / f"{kind}_seed{seed}_pred_kt.npy", p)
            preds.append(p)
            logs.append({"seed": seed, "epochs": log["epochs"], "best_es_loss": log["best_es_loss"],
                         "seconds": log["seconds"], "device": log["device"]})
            print(f"  {variant} {kind} seed {seed}: epochs {log['epochs']} es_loss {log['best_es_loss']:.5f} "
                  f"({log['seconds']}s, {log['device']})")
        self.train_logs = getattr(self, "train_logs", {})
        self.train_logs[f"{variant}/{kind}"] = logs
        return preds, npar

    def to_w(self, kt):
        return np.where(self.tg["day"], kt * self.tg["clear"], np.nan)

    # ---------- scoring ----------
    def score_set(self, variant, row_set, rows, models: dict, strict=True):
        """models: name -> list of (seed, W/m2 array). All scored on identical cells."""
        flat = {f"{n}|{s}": p for n, lst in models.items() for s, p in lst}
        flat["smart_persistence|na"] = self.tg["sp"]
        sel = common_rows(flat, rows, self.tg["day"], strict=strict)
        for r in score(flat, sel, self.tg["ghi"], self.tg["sp"], self.hz):
            name, seed = r.pop("model").split("|")
            self.records.append({"variant": variant, "row_set": row_set, "model": name, "seed": seed, **r})
        return sel

    def gain(self, edge, cloud, sel, horizons=None):
        """Seed-mean of per-horizon RMSE for each tier; gain = 1 - cloud/edge (averaged over horizons)."""
        idx = [self.hz.index(h) for h in (horizons or self.hz)]

        def rmse(p, j):
            s = sel[:, j]
            return float(np.sqrt(np.mean((self.tg["ghi"][s, j] - p[s, j]) ** 2)))

        e = np.array([[rmse(p, j) for j in idx] for p in edge])     # seeds x horizons
        c = np.array([[rmse(p, j) for j in idx] for p in cloud])
        hs = [self.hz[j] for j in idx]
        return {"horizons": hs,
                "edge_rmse_per_horizon": dict(zip(hs, e.mean(0).round(3).tolist())),
                "cloud_rmse_per_horizon": dict(zip(hs, c.mean(0).round(3).tolist())),
                "gain_per_horizon": dict(zip(hs, (1 - c.mean(0) / e.mean(0)).round(4).tolist())),
                "edge_rmse_avg": float(e.mean(1).mean()), "cloud_rmse_avg": float(c.mean(1).mean()),
                "gain_avg": float(1 - c.mean(1).mean() / e.mean(1).mean()),
                "gain_avg_per_seed": (1 - c.mean(1) / e.mean(1)).round(4).tolist()}

    def primary_rows(self):
        return self.va & self.avail


def check_inputs(run: Run, data: dict, lag: int) -> dict:
    """Ladder step 1: the cloud tier really receives tiles and all four NAM nodes, aligned, no future values."""
    df, t = run.df, run.df["timestamp"].values
    st = data["stamps"]
    has = ~np.isnat(st)
    future_tiles = int((st[has] > np.repeat(t[:, None], st.shape[1], 1)[has] - np.timedelta64(lag, "m")).sum())
    pv = run.primary_rows()
    nodes = sorted({c.split("_")[0] for c in data["nam_raw"].columns if c.startswith("nam") and c[3].isdigit()})
    # ground features: B(ghi_kt|30min) at t equals the target block ending at t (backward-looking)
    blk = df.set_index("timestamp")["ghi_kt_30min"].reindex(df["timestamp"] - pd.Timedelta(minutes=30)).values
    okb = np.isfinite(blk)
    # weather means end at t: recompute 30 rows directly from the minute file
    from escal.data import read_dev
    wx = read_dev("Folsom_weather.csv", run.cfg).set_index("timestamp")["air_temp"]
    samp = df.sample(30, random_state=0)
    wx_ok = all(np.isclose(wx[(wx.index > ts - pd.Timedelta(minutes=30)) & (wx.index <= ts)].mean(), v, equal_nan=True)
                for ts, v in zip(samp["timestamp"], samp["wx_air_temp"]))
    res = {
        "tiles_stamped_after_t_minus_lag": future_tiles,
        "primary_val_rows_with_tile": float(data["meta"][pv, data["meta"].shape[1] // 2].mean()),
        "tile_std_on_primary_val_rows": float(data["tiles"][pv, 0].std()),
        "nam_nodes_in_inputs": nodes,
        "nam_alignment": data["nam_info"],
        "B30_equals_block_ending_t_share": float((np.abs(df["B(ghi_kt|30min)"].values[okb] - blk[okb]) < 1e-5).mean()),
        "weather_mean_over_t_minus_30_to_t": bool(wx_ok),
        "target_columns_in_inputs": [c for c in data["tab_cols"] if c.startswith(("ghi_", "dni_")) and "kt|" not in c],
    }
    res["passed"] = bool(future_tiles == 0 and len(nodes) == 4 and res["primary_val_rows_with_tile"] == 1.0
                         and res["B30_equals_block_ending_t_share"] == 1.0 and wx_ok
                         and not res["target_columns_in_inputs"]
                         and all(v["reftime_plus_lag_le_t"] for v in data["nam_info"].values()))
    return res


def ablation(run: Run, variant: str, data: dict) -> dict:
    """Does the trained cloud tier use its inputs? Primary-row RMSE with tiles or NAM neutralised."""
    from escal.models import CloudNet
    sel = common_rows({"x": run.tg["sp"]}, run.primary_rows(), run.tg["day"])
    out = {}
    for seed in run.cfg["seeds"][:1]:
        c = run.tc["cloud"]
        m = CloudNet(data["x"].shape[1], len(run.hz), data["tiles"].shape[1], c["hidden"], c["cnn_channels"],
                     c["fusion_hidden"])
        m.load_state_dict(torch.load(CKPT / variant / f"cloud_seed{seed}.pt"))
        m.to(run.tc["device"] if torch.cuda.is_available() else "cpu")
        g0 = len(run.base.ground_cols)
        variants = {"full": data, "tiles_zeroed": {**data, "tiles": np.zeros_like(data["tiles"])}}
        x2 = data["x"].copy()
        x2[:, g0:g0 + data["n_nam"]] = 0.0
        variants["nam_at_train_mean"] = {**data, "x": x2}
        for k, d in variants.items():
            p = run.to_w(np.clip(predict(m, d), *run.tc["kt_clip"]))
            out[k] = float(np.mean([np.sqrt(np.mean((run.tg["ghi"][sel[:, j], j] - p[sel[:, j], j]) ** 2))
                                    for j in range(len(run.hz))]))
    return {"seed": run.cfg["seeds"][0], "rmse_avg_primary": out}


def tile_summary(tiles: np.ndarray) -> np.ndarray:
    return np.concatenate([tiles.mean((2, 3)), tiles.std((2, 3)), tiles[:, :, 3:7, 3:7].mean((2, 3))], axis=1)


def gbt(run: Run, data: dict, seed: int) -> np.ndarray:
    """Ladder step 4: trees on the tabular inputs plus the tile summary."""
    return gbt_on(run, np.concatenate([data["x"], tile_summary(data["tiles"])], axis=1), seed)


def gbt_on(run: Run, X: np.ndarray, seed: int, max_iter: int | None = None, max_leaf_nodes: int | None = None,
           return_models: bool = False):
    """One HistGradientBoosting model per horizon; iterations picked on the es slice.
    max_iter / max_leaf_nodes override the step-4 settings (used only for size-capped models)."""
    s4 = run.cfg["ladder"]["step4"]
    out = np.zeros((len(X), len(run.hz)), np.float32)
    models = []
    for j in range(len(run.hz)):
        m = run.tg["mask"][:, j] > 0
        tr, es = run.tr & m, run.es & m
        model = HistGradientBoostingRegressor(max_iter=max_iter or s4["max_iter"], learning_rate=s4["learning_rate"],
                                              max_leaf_nodes=max_leaf_nodes or s4["max_leaf_nodes"],
                                              early_stopping=False, random_state=seed)
        model.fit(X[tr], run.tg["kt"][tr, j])
        errs = [np.mean((p - run.tg["kt"][es, j]) ** 2) for p in model.staged_predict(X[es])]
        best = int(np.argmin(errs)) + 1
        model.set_params(max_iter=best, warm_start=False)
        model.fit(X[tr], run.tg["kt"][tr, j])
        out[:, j] = model.predict(X)
        models.append(model)
    out = np.clip(out, *run.tc["kt_clip"])
    return (out, models) if return_models else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    OUT.mkdir(parents=True, exist_ok=True)
    run = Run(cfg)
    hz, tg, va = run.hz, run.tg, run.va

    # satellite availability on validation, reported before training
    day_va = va[:, None] & tg["day"]
    avail = {"definition": f"frame stamped in [t-lag-{run.tc['sat_max_age_minutes']}min, t-lag]",
             "validation_daylight_issue_times": int((day_va.any(1)).sum())}
    for name, a in (("lag_15min_primary", run.avail), ("lag_0min_sensitivity", run.avail0)):
        avail[name] = {"overall_share_of_daylight_issue_times": float(a[day_va.any(1)].mean()),
                       "per_horizon_share_of_daylight_cells": {h: float(a[day_va[:, j]].mean()) for j, h in enumerate(hz)},
                       "per_horizon_primary_cells": {h: int((a & day_va[:, j]).sum()) for j, h in enumerate(hz)}}
    avail["training_rows"] = {"train": int(run.tr.sum()), "early_stop": int(run.es.sum()), "validation": int(va.sum())}
    (OUT / "satellite_availability.json").write_text(json.dumps(avail, indent=2))
    print(json.dumps(avail, indent=2))
    if args.prepare_only:
        return

    # anchors on the same rows (fit exactly as in Stage 3: models_train, dropna over endo + satellite)
    s, e = _bounds(cfg, "models_train")
    max_m = max(horizon_minutes(h) for h in hz)
    fit_rows = ((run.df["timestamp"] >= s) & (run.df["timestamp"] + pd.Timedelta(minutes=max_m) < e)).values
    endo = [c for c in run.base.endo_cols if "ghi" in c]
    allf = endo + run.base.sif_cols
    lasso_endo = np.stack([anchor_predict(run.df, fit_rows, endo, allf, "ghi", h, cfg) for h in hz], 1)
    lasso_exo = np.stack([anchor_predict(run.df, fit_rows, allf, allf, "ghi", h, cfg) for h in hz], 1)
    print("anchors done")

    gcfg = cfg["go_no_go"]
    thr = gcfg["min_rmse_gain"]
    prim = run.primary_rows()

    def evaluate(variant, edge_w, cloud_w, extra_models=None, strict=True):
        models = {"edge": list(zip(cfg["seeds"], edge_w)), "cloud": list(zip(cfg["seeds"], cloud_w)),
                  "lasso_endo": [("na", lasso_endo)]}
        models.update(extra_models or {})
        sel = run.score_set(variant, "primary", prim, models, strict)
        run.score_set(variant, "all_daylight", va, models, strict)
        # anchor rows: primary cells where the benchmark satellite feature exists
        m2 = {**models, "lasso_exo": [("na", lasso_exo)]}
        sel_a = run.score_set(variant, "primary_anchor_rows", prim, m2, strict=False)
        return sel, sel_a

    # ---- base variant (primary, 15-min satellite rule) ----
    edge_kt, edge_np = run.train_tier("edge", "base", run.edge_data())
    edge_w = [run.to_w(p) for p in edge_kt]
    v0 = dict(run.tc["base_variant"])
    cdata = run.cloud_data(v0, run.lag)
    cloud_kt, cloud_np = run.train_tier("cloud", "base", cdata)
    cloud_w = [run.to_w(p) for p in cloud_kt]
    sel, sel_a = evaluate("base", edge_w, cloud_w)
    g = run.gain(edge_w, cloud_w, sel)
    params = {"edge": edge_np, "cloud_base": cloud_np}
    run.variants.append({"name": "base", "ladder_step": 0, "satellite_lag_min": run.lag, "config": v0,
                         "cloud_params": cloud_np, **g, "passed": g["gain_avg"] >= thr})
    print(f"base: gain_avg {g['gain_avg']:.4f}")

    # ---- sensitivity: benchmark convention, 0-minute satellite lag (not used for the verdict) ----
    cdata0 = run.cloud_data(v0, run.tc["sat_lag_sensitivity"])
    c0_kt, _ = run.train_tier("cloud", "sens_lag0", cdata0)
    c0_w = [run.to_w(p) for p in c0_kt]
    sel0, _ = evaluate("sens_lag0 benchmark convention (sensitivity)", edge_w, c0_w)
    g0 = run.gain(edge_w, c0_w, sel0)
    sensitivity = {"name": "benchmark convention (sensitivity)", "satellite_lag_min": run.tc["sat_lag_sensitivity"],
                   "scored_on": "primary rows of the 15-min rule", **g0}

    # anchor check (amendment 5), base variant
    def anchor_check(variant, edge_w, cloud_w, sel_a):
        def avg(p, sel_):
            return float(np.mean([np.sqrt(np.mean((tg["ghi"][sel_[:, j], j] - p[sel_[:, j], j]) ** 2))
                                  for j in range(len(hz))]))
        e = float(np.mean([avg(p, sel) for p in edge_w]))
        c = float(np.mean([avg(p, sel_a) for p in cloud_w]))
        le, lx = avg(lasso_endo, sel), avg(lasso_exo, sel_a)
        return {"variant": variant,
                "edge_vs_lasso_endo (primary rows)": {"edge_rmse_avg": e, "lasso_endo_rmse_avg": le, "edge_beats_anchor": e < le},
                "cloud_vs_lasso_exo (primary rows with benchmark satellite feature)": {
                    "rows": int(sel_a.any(1).sum()), "cloud_rmse_avg": c, "lasso_exo_rmse_avg": lx,
                    "cloud_beats_anchor": c < lx}}

    anchors = [anchor_check("base", edge_w, cloud_w, sel_a)]

    # ---- fallback ladder ----
    best = {"name": "base", "cfg": v0, "cloud_w": cloud_w, "data": cdata, "gain": g}
    passed_at = 0 if g["gain_avg"] >= thr else None
    ladder_log = []
    if passed_at is None:
        # step 1: input check
        chk = check_inputs(run, cdata, run.lag)
        abl = ablation(run, "base", cdata)
        ladder_log.append({"step": 1, "name": "input check", "checks": chk, "ablation": abl,
                           "fault_found": not chk["passed"], "gain_avg": g["gain_avg"],
                           "passed": False,
                           "note": "inputs verified; model unchanged, so the gap is unchanged" if chk["passed"]
                           else "FAULT FOUND: stop and fix before continuing"})
        print("step 1:", "checks passed" if chk["passed"] else "FAULT", json.dumps(abl))
        if not chk["passed"]:
            passed_at = "halted"
    for step in (2, 3):
        if passed_at is not None:
            break
        v = {**best["cfg"], **cfg["ladder"][f"step{step}"]}
        name = f"step{step}"
        d = run.cloud_data(v, run.lag)
        kt, npar = run.train_tier("cloud", name, d)
        w = [run.to_w(p) for p in kt]
        sel_s, sel_as = evaluate(name, edge_w, w)
        gs = run.gain(edge_w, w, sel_s)
        ok = gs["gain_avg"] >= thr
        run.variants.append({"name": name, "ladder_step": step, "satellite_lag_min": run.lag, "config": v,
                             "built_on": best["name"], "cloud_params": npar, **gs, "passed": ok})
        ladder_log.append({"step": step, "name": {2: "longer satellite window + tile differences",
                                                   3: "richer NWP"}[step], "built_on": best["name"],
                           "gain_avg": gs["gain_avg"], "gain_per_horizon": gs["gain_per_horizon"], "passed": ok})
        anchors.append(anchor_check(name, edge_w, w, sel_as))
        print(f"{name}: gain_avg {gs['gain_avg']:.4f}")
        if gs["cloud_rmse_avg"] < best["gain"]["cloud_rmse_avg"]:
            best = {"name": name, "cfg": v, "cloud_w": w, "data": d, "gain": gs}
        if ok:
            passed_at = step
    if passed_at is None:
        # step 4: gradient-boosted trees on the best variant's features, alone and averaged with the network
        gb_w = [run.to_w(gbt(run, best["data"], s)) for s in cfg["seeds"]]
        avg_w = [(a + b) / 2 for a, b in zip(gb_w, best["cloud_w"])]
        res4, base4 = {}, best["name"]   # both step-4 models use the features (and network) of base4
        for name, w in (("step4_gbt", gb_w), ("step4_gbt_plus_net", avg_w)):
            sel_s, sel_as = evaluate(name, edge_w, w)
            gs = run.gain(edge_w, w, sel_s)
            ok = gs["gain_avg"] >= thr
            res4[name] = gs
            run.variants.append({"name": name, "ladder_step": 4, "satellite_lag_min": run.lag, "config": best["cfg"],
                                 "built_on": base4, **gs, "passed": ok})
            anchors.append(anchor_check(name, edge_w, w, sel_as))
            print(f"{name}: gain_avg {gs['gain_avg']:.4f}")
            if gs["cloud_rmse_avg"] < best["gain"]["cloud_rmse_avg"]:
                best = {"name": name, "cfg": best["cfg"], "cloud_w": w, "data": best["data"], "gain": gs}
        ladder_log.append({"step": 4, "name": "stronger cloud model (GBT, GBT+network)",
                           "gain_avg": {k: v["gain_avg"] for k, v in res4.items()},
                           "passed": any(v["gain_avg"] >= thr for v in res4.values())})
        if ladder_log[-1]["passed"]:
            passed_at = 4
    focus = None
    if passed_at is None:
        # step 5: horizons where the gap exists (wording amended 2026-10-07, see docs/PLAN.md section 6)
        gp = best["gain"]["gain_per_horizon"]
        keep = [h for h in hz if gp[h] >= cfg["ladder"]["step5"]["min_horizon_gain"]]
        ok = bool(keep)
        focus = {"best_variant": best["name"], "gain_per_horizon": gp, "horizons_kept": keep}
        if keep:
            gs = run.gain(edge_w, best["cloud_w"], common_rows({"x": tg["sp"]}, prim, tg["day"]), keep)
            focus["gain_avg_on_kept_horizons"] = gs["gain_avg"]
            ok = gs["gain_avg"] >= thr
        ladder_log.append({"step": 5, "name": "horizon focus (horizons where the gap exists)", **focus, "passed": ok})
        if ok:
            passed_at = 5

    pd.DataFrame(run.records).to_csv(OUT / "validation.csv", index=False, float_format="%.4f")
    (OUT / "params.json").write_text(json.dumps(params, indent=2))
    (OUT / "train_logs.json").write_text(json.dumps(run.train_logs, indent=2))
    verdict = ("GO" if passed_at == 0 else f"GO via ladder step {passed_at}" if isinstance(passed_at, int)
               else "HALTED: input fault" if passed_at == "halted" else "NO-GO")
    out = {
        "criterion": {"statistic": "1 - RMSE_cloud / RMSE_edge; RMSE per horizon averaged over horizons, then over seeds",
                      "min_rmse_gain": thr, "judged_on": "validation, primary rows, satellite lag 15 min",
                      "seeds": cfg["seeds"]},
        "verdict": verdict,
        "primary": run.variants[0],
        "best_variant": best["name"],
        "satellite_availability": "results/tiers/satellite_availability.json",
        "sensitivity_benchmark_convention": sensitivity,
        "anchor_check": anchors,
        "ladder": ladder_log,
        "variants": run.variants,
        "params": params,
        "results_file": "results/tiers/validation.csv",
    }
    Path("results/go_no_go.json").write_text(json.dumps(out, indent=2, default=str))
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
