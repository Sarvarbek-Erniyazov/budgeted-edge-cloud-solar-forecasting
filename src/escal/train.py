"""Training with masked kt MSE and early stopping on the end-of-models_train slice."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # required for deterministic cuBLAS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _t(a, dev):
    return None if a is None else torch.as_tensor(a, device=dev)


def masked_mse(pred, y, m):
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)


def fit(model, data: dict, tr: np.ndarray, es: np.ndarray, cfg: dict, seed: int, log_path: Path) -> dict:
    """data: x (N,F), tiles (N,C,10,10) or None, kt (N,H), mask (N,H). tr/es: boolean row masks."""
    tc = cfg["tiers"]
    dev = tc["device"] if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    model.to(dev)
    X, T = _t(data["x"], dev), _t(data.get("tiles"), dev)
    Y, M = _t(data["kt"], dev), _t(data["mask"], dev)
    itr = torch.as_tensor(np.nonzero(tr)[0], device=dev)
    ies = torch.as_tensor(np.nonzero(es)[0], device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    gen = torch.Generator(device=dev).manual_seed(seed)
    best, best_state, wait, hist = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(tc["max_epochs"]):
        model.train()
        perm = itr[torch.randperm(len(itr), device=dev, generator=gen)]
        tot = 0.0
        for k in range(0, len(perm), tc["batch_size"]):
            b = perm[k:k + tc["batch_size"]]
            loss = masked_mse(model(X[b], None if T is None else T[b]), Y[b], M[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            es_loss = masked_mse(model(X[ies], None if T is None else T[ies]), Y[ies], M[ies]).item()
        hist.append({"epoch": epoch, "train_loss": tot / len(perm), "es_loss": es_loss})
        if es_loss < best - 1e-6:
            best, wait = es_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    log = {"seed": seed, "best_es_loss": best, "epochs": len(hist), "seconds": round(time.time() - t0, 1),
           "device": dev, "history": hist}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log))
    return log


def predict(model, data: dict) -> np.ndarray:
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X = torch.as_tensor(data["x"], device=dev)
        T = None if data.get("tiles") is None else torch.as_tensor(data["tiles"], device=dev)
        out = [model(X[k:k + 4096], None if T is None else T[k:k + 4096]).cpu().numpy()
               for k in range(0, len(X), 4096)]
    return np.concatenate(out)
