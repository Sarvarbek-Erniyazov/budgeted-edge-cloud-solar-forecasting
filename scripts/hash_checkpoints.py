"""Write docs/checkpoint_hashes.txt (sha256sum -c format) for every file the test run loads."""
from __future__ import annotations

import hashlib
from pathlib import Path

FILES = ([f"checkpoints/tiers/base/edge_seed{s}.pt" for s in range(5)]
         + [f"checkpoints/tiers/step3/cloud_seed{s}.pt" for s in range(5)]
         + [f"checkpoints/trees/{n}.pkl" for n in ("trees_all", "trees_ground", "trees_ground_cap10x")]
         + ["checkpoints/trees/norms.npz"]
         + [f"checkpoints/footprint/edge_seed{s}.int8.onnx" for s in range(5)])


def main() -> None:
    lines = [f"{hashlib.sha256(Path(f).read_bytes()).hexdigest()}  {f}" for f in FILES]
    Path("docs/checkpoint_hashes.txt").write_text("\n".join(lines) + "\n", newline="\n")   # LF, for sha256sum -c
    print("\n".join(lines))


if __name__ == "__main__":
    main()
