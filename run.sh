#!/usr/bin/env bash
# One command per stage. Stages are added as they are built.
set -euo pipefail
case "${1:-}" in
  download) python scripts/download_data.py ;;
  test)     python -m pytest -q ;;
  audit)    python scripts/audit_data.py --config configs/base.yaml ;;   # built on Oct 8
  *) echo "usage: ./run.sh {download|test|audit}"; exit 1 ;;
esac
