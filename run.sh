#!/usr/bin/env bash
# One command per stage. Stages are added as they are built.
set -euo pipefail
export PYTHONPATH=src
case "${1:-}" in
  download) python scripts/download_data.py ;;
  test)     python -m pytest -q ;;
  audit)    python scripts/audit_data.py --config configs/base.yaml ;;
  anchors)  python scripts/anchors.py --config configs/base.yaml ;;
  split)    python scripts/split_check.py --config configs/base.yaml ;;
  tiers)    python scripts/tiers.py --config configs/base.yaml "${@:2}"
            [[ "${2:-}" == "--prepare-only" ]] || python scripts/tiers_summary.py ;;
  control)  python scripts/control.py --config configs/base.yaml ;;
  footprint) python scripts/footprint_trees.py --config configs/base.yaml ;;
  *) echo "usage: ./run.sh {download|test|audit|anchors|split|tiers|control|footprint}"; exit 1 ;;
esac
