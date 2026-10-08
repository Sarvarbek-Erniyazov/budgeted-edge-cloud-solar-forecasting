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
  predictions) python scripts/predictions.py --config configs/base.yaml ;;
  gates)    python scripts/gates.py --config configs/base.yaml ;;
  measure)  python scripts/measure_footprint.py --config configs/base.yaml ;;
  save_trees) python scripts/save_trees.py --config configs/base.yaml ;;
  hashes)   python scripts/hash_checkpoints.py ;;
  refit_smoke) python scripts/refit_smoke.py ;;
  dryrun)   python scripts/test_run.py predictions --dry-run
            python scripts/test_run.py gates --dry-run
            python scripts/test_run.py claims --dry-run ;;
  test_predictions) python scripts/test_run.py predictions ;;
  test_gates)  python scripts/test_run.py gates ;;
  test_claims) python scripts/test_run.py claims ;;
  analyses) python scripts/planned_analyses.py ;;
  exploratory_halfyear) python scripts/exploratory_halfyear_claims.py ;;
  figures)  python scripts/core_figures.py ;;
  trace)    python scripts/number_trace.py ;;
  *) echo "usage: ./run.sh {download|test|audit|anchors|split|tiers|control|footprint|predictions|gates|measure|save_trees|hashes|refit_smoke|dryrun|test_predictions|test_gates|test_claims|analyses|exploratory_halfyear|figures|trace}"; exit 1 ;;
esac
