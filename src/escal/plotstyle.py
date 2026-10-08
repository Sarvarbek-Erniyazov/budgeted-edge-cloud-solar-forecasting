"""One shared figure style: Okabe-Ito colours (colour-blind safe) with markers and line styles as a second
encoding, thin marks, recessive grid. Every figure is saved as SVG and PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
GATE_STYLE = {   # colour, marker, line style
    "random": ("#999999", "o", (0, (4, 2))),
    "fixed_interval": ("#E69F00", "s", "-"),
    "variability": ("#56B4E9", "^", "-"),
    "uncertainty": ("#0072B2", "D", "-"),
    "learned": ("#D55E00", "v", "-"),
    "oracle": ("#000000", "*", (0, (1, 1.5))),
}
GATE_LABEL = {"random": "Random (expected)", "fixed_interval": "Fixed interval", "variability": "Variability rule",
              "uncertainty": "Uncertainty gate", "learned": "Learned gate", "oracle": "Oracle"}
SEQ_CMAP = "Blues"


def apply() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
        "lines.markersize": 6, "svg.fonttype": "none", "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def save(fig, path_stem: Path) -> list[str]:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("svg", "png"):
        p = path_stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight")
        out.append(str(p).replace("\\", "/"))
    plt.close(fig)
    return out
