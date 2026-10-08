"""Shared loader for the post-test analyses: reads results/test/ and the saved gate checkpoints only.
Gate decisions per issue time are recomputed by inference with the saved 2015-refit networks and the saved
thresholds (nothing is retrained); verify_decisions() checks them against results/test/gates/sweep_test.csv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from escal import gates as G
from escal.data import load_config
from escal.models import EdgeNet
from escal.train import predict, set_seed


def load_cfgs():
    return load_config(), yaml.safe_load(open("configs/analyses.yaml"))


class Test:
    """Arrays over the saved test-run rows (fit + test parts); `ev` selects the 2016 evaluation population."""

    def __init__(self, cfg: dict, acfg: dict):
        self.cfg, self.acfg = cfg, acfg
        P = Path(acfg["pred_dir"])
        self.truth = pd.read_parquet(P / "truth.parquet")
        self.inp = pd.read_parquet(P / "on_device_inputs.parquet")
        t = self.truth
        self.hz = [c[4:] for c in t.columns if c.startswith("ghi_")]
        self.seeds = cfg["seeds"]
        self.y = t[[f"ghi_{h}" for h in self.hz]].values
        self.clear = t[[f"clear_{h}" for h in self.hz]].values
        self.sp = t[[f"sp_{h}" for h in self.hz]].values
        self.day = t[[f"day_{h}" for h in self.hz]].values.astype(bool)
        self.pop = t["sat_available"].values & self.day.any(1)
        self.ev = (t["part"] == "test").values & self.pop
        self.fit = t["part"].isin(["gate_fit", "val"]).values & self.pop
        self.sel = self.day & self.ev[:, None]
        kt = t[[f"kt_{h}" for h in self.hz]].values
        prev = np.column_stack([t["B(ghi_kt|30min)"].values, kt[:, :-1]])
        self.ramp = self.sel & (np.abs(kt - prev) >= cfg["ramp"]["delta_kt"])
        self.ts = t["timestamp"]
        self.local = self.ts + pd.Timedelta(hours=acfg["local_utc_offset_hours"])
        self._p = {}

    def kt(self, name: str, seed) -> np.ndarray:
        if name not in self._p:
            self._p[name] = pd.read_parquet(Path(self.acfg["pred_dir"]) / f"{name}.parquet")
        d = self._p[name]
        return d[d["seed"].astype(str) == str(seed)][[f"kt_{h}" for h in self.hz]].values

    def W(self, name: str, seed) -> np.ndarray:
        return np.where(self.day, self.kt(name, seed) * self.clear, np.nan)

    def gate_scores(self, seed) -> dict:
        gc = self.cfg["gates"]
        ck = Path(self.acfg["gate_ckpt"]) / f"seed{seed}"
        norm = np.load(ck / "norm.npz", allow_pickle=True)
        X = pd.concat([self.inp.iloc[:, 1:], pd.DataFrame(self.kt("edge", seed), columns=[f"edge_kt_{h}" for h in self.hz])], axis=1)
        mu = pd.Series(norm["mu"], index=X.columns)
        sd = pd.Series(norm["sd"], index=X.columns)
        Xs = ((X - mu) / sd).fillna(0).values.astype(np.float32)
        dev = self.cfg["tiers"]["device"] if torch.cuda.is_available() else "cpu"
        out = {"variability": self.inp[gc["variability_feature"]].fillna(0).values}
        for g in ("uncertainty", "learned"):
            set_seed(seed)
            m = EdgeNet(Xs.shape[1], 1, gc["net_hidden"])
            m.load_state_dict(torch.load(ck / f"{g}.pt", map_location=dev))
            m.to(dev)
            out[g] = predict(m, {"x": Xs, "tiles": None})[:, 0]
        return out

    def thresholds(self):
        return yaml.safe_load(open(self.acfg["thresholds"]))["gate_thresholds"]

    def decisions(self, gate: str, b: float, seed, scores: dict, tau: dict) -> np.ndarray:
        """Escalation flags over the evaluation population rows (in time order)."""
        v = np.nonzero(self.ev)[0]
        if gate == "fixed_interval":
            from escal.bootstrap import day_index
            return G.fixed_interval(day_index(self.ts.iloc[v]), b)
        if gate == "oracle":
            E, C = self.W("edge", seed)[v], self.W("cloud", seed)[v]
            return G.oracle(G.benefit(self.y[v], E, C, self.sel[v]), b)
        return G.decide(scores[gate][v], tau[int(seed)][gate][float(b)])


def verify_decisions(T: Test, scores_by_seed: dict) -> dict:
    """Recomputed decisions must reproduce sweep_test.csv (rate and RMSE, 4 dp) for every score gate."""
    sw = pd.read_csv(Path(T.acfg["gates_dir"]) / "sweep_test.csv")
    sw = sw[(sw.escalate_to == "cloud") & (sw.row_set == "primary")]
    tau = T.thresholds()
    v = np.nonzero(T.ev)[0]
    worst, n = 0.0, 0
    for s in T.seeds:
        E, C = T.W("edge", s)[v], T.W("cloud", s)[v]
        for g in ("fixed_interval", "variability", "uncertainty", "learned", "oracle"):
            for b in T.cfg["budget"]["targets"]:
                esc = T.decisions(g, b, s, scores_by_seed[s], tau)
                r = G.avg_metrics(T.y[v], G.blend(E, C, esc), T.sp[v], T.sel[v])["RMSE"]
                ref = sw[(sw.gate == g) & (sw.budget_target == b) & (sw.seed == s)].iloc[0]
                worst = max(worst, abs(round(r, 4) - ref["RMSE"]), abs(round(float(esc.mean()), 4) - ref["realised_rate"]))
                n += 1
    return {"checked": n, "max_abs_diff_4dp": worst, "reproduces_sweep": worst == 0.0}
