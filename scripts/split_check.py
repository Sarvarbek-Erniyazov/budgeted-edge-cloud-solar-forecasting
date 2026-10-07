"""Stage: split decision. Checks gate_fit for variable-sky days against configs split_check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from escal.data import load_config
from escal.splits import check_no_overlap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    check_no_overlap(cfg)
    by = pd.read_csv("results/audit/sky_mix_by_split.csv", index_col="split")
    c = cfg["split_check"]
    g = by.loc["gate_fit"]
    passed = bool(g["partly_cloudy"] >= c["min_variable_days"] and g["partly_cloudy_share"] >= c["min_variable_share"])
    out = {
        "criterion": c,
        "gate_fit": {"classified_days": int(g["classified_days"]), "partly_cloudy_days": int(g["partly_cloudy"]),
                     "partly_cloudy_share": float(g["partly_cloudy_share"])},
        "for_reference": {s: {"partly_cloudy_days": int(by.loc[s, "partly_cloudy"]),
                              "partly_cloudy_share": float(by.loc[s, "partly_cloudy_share"])}
                          for s in by.index if s != "gate_fit"},
        "split": cfg["split"],
        "passed": passed,
        "decision": "keep boundaries" if passed else "move gate_fit/validation boundary inside 2015",
    }
    Path("results/audit/split_decision.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
