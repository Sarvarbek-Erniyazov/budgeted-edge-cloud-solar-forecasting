#!/usr/bin/env bash
# One command per stage. Stages are added as they are built.
set -euo pipefail
export PYTHONPATH=src
case "${1:-}" in
  download) python scripts/download_data.py ;;
  test)     python -m pytest -q ;;
  audit)    python scripts/audit_data.py --config configs/base.yaml ;;
  anchors)  python scripts/anchors.py --config configs/base.yaml ;;
  *) echo "usage: ./run.sh {download|test|audit|anchors}"; exit 1 ;;
esac
