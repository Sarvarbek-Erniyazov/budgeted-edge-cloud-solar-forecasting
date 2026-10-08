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


def _outline_hits(path, bb, step_px: float = 0.5) -> bool:
    """True if any point on the drawn outline (sampled every step_px) lies inside bb."""
    import numpy as np
    for poly in path.to_polygons(closed_only=False):
        for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step_px) + 1)
            xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
            if np.any((xs >= bb.x0) & (xs <= bb.x1) & (ys >= bb.y0) & (ys <= bb.y1)):
                return True
    return False


def layout_problems(fig, min_pt: float = 0.0, pad_px: float = 1.0) -> list[str]:
    """Text that touches a patch outline (box, border, arrow) or another text, leaves the figure, or is smaller
    than min_pt."""
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden = set()   # tick labels of ticks outside the view interval are not drawn
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:
                    hidden.update((id(tk.label1), id(tk.label2)))
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    patches = [p for ax in fig.axes for p in ax.patches if isinstance(p, Patch)]
    out = []
    boxes = [(t, t.get_window_extent(r).expanded(1.0, 1.0).padded(pad_px)) for t in texts]
    for t, bb in boxes:
        fb = fig.bbox
        if bb.x0 < fb.x0 or bb.y0 < fb.y0 or bb.x1 > fb.x1 or bb.y1 > fb.y1:
            out.append(f"{t.get_text()!r} extends past the figure edge")
        if t.get_fontsize() < min_pt:
            out.append(f"{t.get_text()!r}: {t.get_fontsize()} pt < {min_pt} pt")
        for p in patches:
            if _outline_hits(p.get_transform().transform_path(p.get_path()), bb):
                out.append(f"{t.get_text()!r} touches a patch outline")
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            if ba.overlaps(bb):
                out.append(f"{a.get_text()!r} overlaps {b.get_text()!r}")
    return out
