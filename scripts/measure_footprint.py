"""Stage 7: measured footprint. int8 ONNX for the edge network and the two fitted gates (calibrated on
models_train only); ONNX and generated C for the 10x-capped ground-only trees. Validation only.
Artifacts go to checkpoints/footprint/ (not committed); numbers to results/footprint/measured.json.
Latency is single-row, one thread, on a PC CPU. Nothing runs on a microcontroller."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).parent))
from footprint_trees import cloud_kt  # noqa: E402
from tiers import CKPT, Run, gbt_on  # noqa: E402

from escal import gates as G  # noqa: E402
from escal.data import load_config  # noqa: E402
from escal.evaluate import common_rows  # noqa: E402
from escal.features import standardise  # noqa: E402
from escal.models import EdgeNet  # noqa: E402

ART = Path("checkpoints/footprint")
OUT = Path("results/footprint")
GCKPT = Path("checkpoints/gates")


class Reader(CalibrationDataReader):
    def __init__(self, X: np.ndarray, name: str):
        self.it = iter([{name: X[i:i + 1]} for i in range(len(X))])

    def get_next(self):
        return next(self.it, None)


def session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run_all(sess, X: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: X[k:k + 4096]})[0] for k in range(0, len(X), 4096)])


def latency(fns: list, x: np.ndarray, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        for f in fns:
            f(x)
    t = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter_ns()
        for f in fns:
            f(x)
        t[i] = time.perf_counter_ns() - t0
    return {"median_us": float(np.median(t) / 1e3), "p95_us": float(np.percentile(t, 95) / 1e3), "runs": runs}


def ort_fn(sess):
    name = sess.get_inputs()[0].name
    return lambda x: sess.run(None, {name: x})


def export_mlp(model: torch.nn.Module, n_in: int, path: Path, opset: int) -> None:
    model.eval()
    torch.onnx.export(model, (torch.zeros(1, n_in),), str(path), input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, opset_version=opset, dynamo=False)


def quantise(fp32: Path, int8: Path, calib: np.ndarray) -> None:
    quantize_static(str(fp32), str(int8), Reader(calib.astype(np.float32), "x"), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=False,
                    calibrate_method=CalibrationMethod.MinMax)


def avg_rmse(y, p, sel):
    return float(np.mean([np.sqrt(np.mean((y[sel[:, j], j] - p[sel[:, j], j]) ** 2)) for j in range(y.shape[1])]))


def rmse_h(y, p, sel):
    return [float(np.sqrt(np.mean((y[sel[:, j], j] - p[sel[:, j], j]) ** 2))) for j in range(y.shape[1])]


def trees_onnx(models, n_feat: int, opset: int):
    """One TreeEnsembleRegressor for all horizons: target h = baseline_h + sum of horizon-h trees.
    Branch rule as in HistGradientBoosting: x <= threshold goes left."""
    a = {k: [] for k in ("nodes_treeids", "nodes_nodeids", "nodes_featureids", "nodes_modes", "nodes_values",
                         "nodes_truenodeids", "nodes_falsenodeids", "target_treeids", "target_nodeids",
                         "target_ids", "target_weights")}
    tid = 0
    for h, mdl in enumerate(models):
        for it in mdl._predictors:
            for tree in it:
                n = tree.nodes
                for k in range(len(n)):
                    leaf = bool(n["is_leaf"][k])
                    a["nodes_treeids"].append(tid)
                    a["nodes_nodeids"].append(k)
                    a["nodes_featureids"].append(0 if leaf else int(n["feature_idx"][k]))
                    a["nodes_modes"].append("LEAF" if leaf else "BRANCH_LEQ")
                    a["nodes_values"].append(0.0 if leaf else float(n["num_threshold"][k]))
                    a["nodes_truenodeids"].append(0 if leaf else int(n["left"][k]))
                    a["nodes_falsenodeids"].append(0 if leaf else int(n["right"][k]))
                    if leaf:
                        a["target_treeids"].append(tid)
                        a["target_nodeids"].append(k)
                        a["target_ids"].append(h)
                        a["target_weights"].append(float(n["value"][k]))
                tid += 1
    base = [float(np.ravel(m._baseline_prediction)[0]) for m in models]
    node = helper.make_node("TreeEnsembleRegressor", ["x"], ["y"], domain="ai.onnx.ml", n_targets=len(models),
                            aggregate_function="SUM", post_transform="NONE", base_values=base, **a)
    g = helper.make_graph([node], "trees_cap10x", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, n_feat])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, len(models)])])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", opset), helper.make_opsetid("ai.onnx.ml", 3)],
                             ir_version=9)   # opset 17 era; ONNX Runtime 1.30 reads IR <= 13


# ---------------- generated C for the trees ----------------
def tree_arrays(model) -> dict:
    feat, thr, left, right, roots = [], [], [], [], []
    for it in model._predictors:
        for tree in it:
            n = tree.nodes
            off = len(feat)
            roots.append(off)
            for k in range(len(n)):
                leaf = bool(n["is_leaf"][k])
                feat.append(-1 if leaf else int(n["feature_idx"][k]))
                thr.append(float(n["value"][k]) if leaf else float(n["num_threshold"][k]))
                left.append(0 if leaf else off + int(n["left"][k]))
                right.append(0 if leaf else off + int(n["right"][k]))
    return {"feat": feat, "thr": thr, "left": left, "right": right, "roots": roots,
            "base": float(np.ravel(model._baseline_prediction)[0])}


def write_c(arrs: list[dict], n_feat: int, path: Path) -> dict:
    lines = ["/* Generated by scripts/measure_footprint.py: 10x-capped ground-only trees, one ensemble per horizon. */",
             "#include <stdint.h>", f"#define N_FEATURES {n_feat}", f"#define N_HORIZONS {len(arrs)}", ""]
    idx_t = "int16_t" if max(max(a["left"] + a["right"] + a["roots"]) for a in arrs) < 32767 else "int32_t"
    data_bytes = 0
    for h, a in enumerate(arrs):
        n = len(a["feat"])
        lines.append(f"static const int16_t F{h}[{n}] = {{{','.join(map(str, a['feat']))}}};")
        lines.append(f"static const float T{h}[{n}] = {{{','.join(repr(np.float32(v).item()) + 'f' for v in a['thr'])}}};")
        lines.append(f"static const {idx_t} L{h}[{n}] = {{{','.join(map(str, a['left']))}}};")
        lines.append(f"static const {idx_t} R{h}[{n}] = {{{','.join(map(str, a['right']))}}};")
        lines.append(f"static const {idx_t} S{h}[{len(a['roots'])}] = {{{','.join(map(str, a['roots']))}}};")
        isz = 2 if idx_t == "int16_t" else 4
        data_bytes += n * (2 + 4 + 2 * isz) + len(a["roots"]) * isz + 4
    lines += ["", "static float ensemble(const float *x, const int16_t *F, const float *T, const void *Lv,",
              f"                      const void *Rv, const {idx_t} *S, int n_trees, float base) {{",
              f"    const {idx_t} *L = (const {idx_t} *)Lv, *R = (const {idx_t} *)Rv;",
              "    float s = base;",
              "    for (int t = 0; t < n_trees; t++) {",
              "        int k = S[t];",
              "        while (F[k] >= 0) k = (x[F[k]] <= T[k]) ? L[k] : R[k];",
              "        s += T[k];",
              "    }",
              "    return s;", "}", "",
              "void trees_predict(const float *x, float *out) {"]
    for h, a in enumerate(arrs):
        lines.append(f"    out[{h}] = ensemble(x, F{h}, T{h}, L{h}, R{h}, S{h}, {len(a['roots'])}, {a['base']!r}f);")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")
    return {"index_type": idx_t, "static_data_bytes": int(data_bytes)}


BENCH = r"""
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
void trees_predict(const float *x, float *out);
static int cmp(const void *a, const void *b) { double d = *(const double *)a - *(const double *)b; return (d > 0) - (d < 0); }
int main(int argc, char **argv) {
    int n = atoi(argv[2]), f = atoi(argv[3]), warm = atoi(argv[5]), runs = atoi(argv[6]);
    float *x = malloc(sizeof(float) * n * f), *y = malloc(sizeof(float) * n * 6);
    FILE *fp = fopen(argv[1], "rb"); fread(x, sizeof(float), (size_t)n * f, fp); fclose(fp);
    for (int i = 0; i < n; i++) trees_predict(x + (size_t)i * f, y + (size_t)i * 6);
    fp = fopen(argv[4], "wb"); fwrite(y, sizeof(float), (size_t)n * 6, fp); fclose(fp);
    LARGE_INTEGER fq, a, b; QueryPerformanceFrequency(&fq);
    double *t = malloc(sizeof(double) * runs); float o[6]; volatile float sink = 0;
    for (int i = 0; i < warm; i++) { trees_predict(x, o); sink += o[0]; }
    for (int i = 0; i < runs; i++) {
        QueryPerformanceCounter(&a); trees_predict(x + (size_t)(i % n) * f, o); QueryPerformanceCounter(&b);
        sink += o[0]; t[i] = (double)(b.QuadPart - a.QuadPart) * 1e6 / (double)fq.QuadPart;
    }
    qsort(t, runs, sizeof(double), cmp);
    printf("{\"median_us\": %.4f, \"p95_us\": %.4f, \"runs\": %d, \"timer_resolution_us\": %.4f}\n",
           t[runs / 2], t[(int)(0.95 * runs)], runs, 1e6 / (double)fq.QuadPart);
    return 0;
}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    q = cfg["quantise"]
    ART.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    run = Run(cfg)
    seeds, hz, tg = cfg["seeds"], run.hz, run.tg
    calib_rows = (run.df["part"].isin(["train", "es"])).values          # models_train only
    prim = common_rows({"sp": tg["sp"]}, run.primary_rows(), tg["day"])
    y = tg["ghi"]
    Xg = standardise(run.ground, run.tr)
    one = Xg[np.nonzero(run.va)[0][:1]].astype(np.float32)
    lat = dict(warmup=q["latency_warmup"], runs=q["latency_runs"])
    out = {"note": "Latency: single row, one thread, ONNX Runtime or compiled C on a PC CPU "
                   f"({ort.get_device()} provider CPU). Nothing ran on a microcontroller.",
           "calibration": "models_train rows only (train + early-stop slice), MinMax, QDQ, int8 weights, uint8 activations"}

    # ---- edge network, all seeds ----
    edge = {"per_seed": []}
    for s in seeds:
        m = EdgeNet(Xg.shape[1], len(hz), run.tc["edge"]["hidden"])
        m.load_state_dict(torch.load(CKPT / "base" / f"edge_seed{s}.pt", map_location="cpu"))
        f32, i8 = ART / f"edge_seed{s}.onnx", ART / f"edge_seed{s}.int8.onnx"
        export_mlp(m, Xg.shape[1], f32, q["opset"])
        quantise(f32, i8, Xg[calib_rows])
        sf, si = session(f32, q["threads"]), session(i8, q["threads"])
        clip = run.tc["kt_clip"]
        pt = np.load(CKPT / "base" / f"edge_seed{s}_pred_kt.npy")
        pf = np.clip(run_all(sf, Xg.astype(np.float32)), *clip)
        pi = np.clip(run_all(si, Xg.astype(np.float32)), *clip)
        rec = {"seed": s, "fp32_onnx_bytes": f32.stat().st_size, "int8_onnx_bytes": i8.stat().st_size,
               "max_abs_kt_diff_torch_vs_onnx_fp32": float(np.abs(pf - pt).max()),
               "rmse_fp32": avg_rmse(y, run.to_w(pf), prim), "rmse_int8": avg_rmse(y, run.to_w(pi), prim),
               "rmse_per_horizon_fp32": rmse_h(y, run.to_w(pf), prim),
               "rmse_per_horizon_int8": rmse_h(y, run.to_w(pi), prim)}
        if s == seeds[0]:
            rec["latency_fp32"] = latency([ort_fn(sf)], one, **lat)
            rec["latency_int8"] = latency([ort_fn(si)], one, **lat)
        edge["per_seed"].append(rec)
        print("edge", s, {k: v for k, v in rec.items() if "per_horizon" not in k}, flush=True)
    out["edge_network"] = edge

    # ---- gates, all seeds ----
    clouds = cloud_kt(run)
    pop = run.avail & tg["day"].any(1)
    gf = (run.df["part"] == "gate_fit").values & pop
    va = (run.df["part"] == "val").values & pop
    selv = tg["day"][va] & True
    gates_out = {"per_seed": []}
    for s, ckt in zip(seeds, clouds):
        norm = np.load(GCKPT / f"seed{s}" / "norm.npz", allow_pickle=True)
        ekt = np.load(CKPT / "base" / f"edge_seed{s}_pred_kt.npy")
        X = np.concatenate([run.ground.values, ekt], axis=1)
        Xs = np.nan_to_num((X - norm["mu"]) / norm["sd"]).astype(np.float32)
        E, C = run.to_w(ekt), run.to_w(ckt)
        Ev, Cv, yv = E[va], C[va], y[va]
        re, rc = avg_rmse(yv, Ev, selv), avg_rmse(yv, Cv, selv)
        rec = {"seed": s}
        for g in ("uncertainty", "learned"):
            m = EdgeNet(Xs.shape[1], 1, cfg["gates"]["net_hidden"])
            m.load_state_dict(torch.load(GCKPT / f"seed{s}" / f"{g}.pt", map_location="cpu"))
            f32, i8 = ART / f"gate_{g}_seed{s}.onnx", ART / f"gate_{g}_seed{s}.int8.onnx"
            export_mlp(m, Xs.shape[1], f32, q["opset"])
            quantise(f32, i8, Xs[calib_rows])
            sf, si = session(f32, q["threads"]), session(i8, q["threads"])
            r = {"fp32_onnx_bytes": f32.stat().st_size, "int8_onnx_bytes": i8.stat().st_size}
            for ver, sess in (("fp32", sf), ("int8", si)):
                sc = run_all(sess, Xs)[:, 0]
                tau = G.fit_thresholds(sc[gf], q["report_budgets"])
                r[ver] = {}
                for b in q["report_budgets"]:
                    esc = G.decide(sc[va], tau[b])
                    rg = avg_rmse(yv, G.blend(Ev, Cv, esc), selv)
                    r[ver][str(b)] = {"realised_rate": float(esc.mean()), "rmse": rg,
                                      "share_retained": G.share_retained(re, rg, rc)}
                r[f"scores_{ver}"] = sc
            r["score_corr_fp32_int8_validation"] = float(np.corrcoef(r["scores_fp32"][va], r["scores_int8"][va])[0, 1])
            del r["scores_fp32"], r["scores_int8"]
            if s == seeds[0]:
                r["latency_fp32"] = latency([ort_fn(sf)], Xs[np.nonzero(va)[0][:1]], **lat)
                r["latency_int8"] = latency([ort_fn(si)], Xs[np.nonzero(va)[0][:1]], **lat)
            rec[g] = r
        gates_out["per_seed"].append(rec)
        print("gates", s, json.dumps({g: {k: rec[g][k] for k in ("int8_onnx_bytes", "score_corr_fp32_int8_validation")}
                                      for g in ("uncertainty", "learned")}), flush=True)
    out["gates"] = gates_out

    # ---- 10x-capped trees_ground: ONNX and generated C ----
    fp = json.loads(Path("results/tiers/footprint_trees.json").read_text())["trees"]["trees_ground_cap10x"]
    kt_sk, models = gbt_on(run, Xg, seeds[0], max_iter=fp["max_trees_per_horizon"],
                           max_leaf_nodes=fp["max_leaf_nodes"], return_models=True)
    raw_sk = np.stack([mdl.predict(Xg) for mdl in models], 1)
    X32 = Xg.astype(np.float32)
    p = ART / "trees_cap10x.onnx"
    onnx.save(trees_onnx(models, Xg.shape[1], q["opset"]), str(p))
    onnx_paths, sessions = [p], [session(p, q["threads"])]
    raw_onnx = run_all(sessions[0], X32)
    kt_onnx = np.clip(raw_onnx, *run.tc["kt_clip"])
    trees = {"estimate_bytes_12_per_node": fp["compact_size_bytes_estimate"], "nodes": fp["nodes"],
             "trees": fp["trees"],
             "onnx": {"files": len(onnx_paths), "total_bytes": int(sum(p.stat().st_size for p in onnx_paths)),
                      "max_abs_diff_vs_sklearn": float(np.abs(raw_onnx - raw_sk).max()),
                      "rmse_sklearn": avg_rmse(y, run.to_w(kt_sk), prim),
                      "rmse_onnx": avg_rmse(y, run.to_w(kt_onnx), prim),
                      "exporter": "custom TreeEnsembleRegressor (ai.onnx.ml v3), all six horizons in one file; skl2onnx 1.20 failed on sklearn 1.9 HistGradientBoosting",
                      "latency_all_horizons": latency([ort_fn(sx) for sx in sessions], X32[np.nonzero(run.va)[0][:1]], **lat)}}
    # generated C
    arrs = [tree_arrays(mdl) for mdl in models]
    cpath = ART / "trees_cap10x.c"
    cinfo = write_c(arrs, Xg.shape[1], cpath)
    (ART / "bench.c").write_text(BENCH)
    cc = [sys.executable if c == "python" else c for c in q["c_compiler"]]   # the venv interpreter
    subprocess.run(cc + ["-c", str(cpath), "-o", str(ART / "trees_cap10x.o")], check=True)
    exe = ART / "bench.exe"
    subprocess.run(cc + [str(ART / "bench.c"), str(cpath), "-o", str(exe)], check=True)
    vrows = np.nonzero(run.va)[0]
    X32[vrows].tofile(ART / "val_x.bin")
    res = subprocess.run([str(exe), str(ART / "val_x.bin"), str(len(vrows)), str(Xg.shape[1]),
                          str(ART / "val_y.bin"), str(q["latency_warmup"]), str(q["latency_runs"])],
                         check=True, capture_output=True, text=True)
    raw_c = np.fromfile(ART / "val_y.bin", dtype=np.float32).reshape(len(vrows), 6)
    kt_c = np.full_like(kt_sk, np.nan)
    kt_c[vrows] = np.clip(raw_c, *run.tc["kt_clip"])
    selc = prim & np.isfinite(kt_c)
    trees["generated_c"] = {"source_bytes": cpath.stat().st_size, "object_bytes": (ART / "trees_cap10x.o").stat().st_size,
                            **cinfo, "compiler": " ".join(q["c_compiler"]) + " (zig " + subprocess.run(cc[:3] + ["version"], capture_output=True, text=True).stdout.strip() + ")",
                            "max_abs_diff_vs_sklearn": float(np.abs(raw_c - raw_sk[vrows]).max()),
                            "rmse_sklearn_same_rows": avg_rmse(y, run.to_w(kt_sk), selc),
                            "rmse_c": avg_rmse(y, run.to_w(kt_c), selc),
                            "latency_single_row": json.loads(res.stdout.strip().splitlines()[-1])}
    out["trees_ground_cap10x"] = trees
    (OUT / "measured.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["trees_ground_cap10x"], indent=1))


if __name__ == "__main__":
    main()
